using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Animation;
using Microsoft.Windows.ApplicationModel.Resources;
using Sodalite.Models;
using Sodalite.Services;
using Sodalite.Views;

namespace Sodalite;

public sealed partial class MainWindow : Window
{
    static readonly ResourceLoader ResourceLoader = new();

    readonly BackendProcessManager _backendProcessManager = new(BackendLocator.BackendProjectPath);
    readonly GenerationPage _generationPage;
    readonly GalleryPage _galleryPage = new();
    readonly SystemMonitorService _systemMonitorService = new();
    readonly DispatcherTimer _systemStatsTimer;
    BackendApiClient? _apiClient;

    public MainWindow()
    {
        InitializeComponent();

        Title = ResourceLoader.GetString("MainWindow_Title");
        AppWindow.SetIcon("Assets/AppIcon.ico");
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(TitleBarGrid);

        Closed += MainWindow_Closed;

        UpdateWebUiToggleUi();

        // GenerationPage は Frame.Navigate ではなく先に生成してイベント購読する。Navigate 後の
        // 購読では遷移直後に発火する初期状態イベントを取りこぼすため。インスタンスは保持し、
        // モデル選択ページから戻る際はこのインスタンスへ復帰させて状態を維持する。
        _generationPage = new GenerationPage { OwnerWindow = this };
        _generationPage.StatusChanged += (_, status) => StatusBarTextBlock.Text = status;
        _generationPage.DeviceInfoChanged += (_, deviceInfo) => StatusBarDeviceInfoTextBlock.Text = deviceInfo;
        _generationPage.BackendReadyChanged += (_, _) => ModelSelectionButton.IsEnabled = true;
        _generationPage.NotifyCurrentStatus();

        _galleryPage.BackRequested += GalleryPage_BackRequested;
        _galleryPage.ReuseParametersRequested += GalleryPage_ReuseParametersRequested;

        RootHost.Children.Add(_generationPage);

        _systemStatsTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
        _systemStatsTimer.Tick += SystemStatsTimer_Tick;
        _systemStatsTimer.Start();

        _ = StartBackendAsync();
    }

    async void SystemStatsTimer_Tick(object? sender, object e)
    {
        SystemStats stats = await _systemMonitorService.GetStatsAsync().ConfigureAwait(true);

        SystemStatsPanel.Visibility = Visibility.Visible;
        CpuUsageTextBlock.Text = $"{stats.CpuUsagePercent:F0}%";
        MemoryUsageTextBlock.Text =
            $"{stats.MemoryUsedGiB:F1} / {stats.MemoryTotalGiB:F1} GiB";

        if (stats.GpuUsagePercent is double gpuPercent)
        {
            GpuStatsPanel.Visibility = Visibility.Visible;
            GpuUsageTextBlock.Text = $"{gpuPercent:F0}%";
        }
        else
        {
            GpuStatsPanel.Visibility = Visibility.Collapsed;
        }

        if (stats.VramUsedGiB is double vramUsedGiB && stats.VramTotalGiB is double vramTotalGiB)
        {
            VramStatsPanel.Visibility = Visibility.Visible;
            VramUsageTextBlock.Text = $"{vramUsedGiB:F1} / {vramTotalGiB:F1} GiB";
        }
        else
        {
            VramStatsPanel.Visibility = Visibility.Collapsed;
        }
    }

    /// <summary>横スライドの移動量(px)。画面幅ぶんではなく控えめに動かして自然に見せる。</summary>
    const double SlideOffset = 48;

    /// <summary>
    /// <paramref name="incoming"/> を <see cref="RootHost"/> に重ねて横スライドで前面に出し、
    /// 完了後に背面のページを取り除く。<paramref name="reverse"/> が true なら逆向き(戻る)に動かす。
    /// </summary>
    async Task SlideToPageAsync(UIElement incoming, bool reverse)
    {
        UIElement? outgoing = RootHost.Children.Count > 0 ? RootHost.Children[^1] : null;

        double fromX = reverse ? -SlideOffset : SlideOffset;
        TranslateTransform incomingTransform = new() { X = fromX };
        incoming.RenderTransform = incomingTransform;
        incoming.Opacity = 0;
        RootHost.Children.Add(incoming);

        Storyboard storyboard = new();
        AddSlide(storyboard, incomingTransform, fromX, 0);
        AddFade(storyboard, incoming, 0, 1);

        if (outgoing is not null)
        {
            TranslateTransform outgoingTransform = new();
            outgoing.RenderTransform = outgoingTransform;
            AddSlide(storyboard, outgoingTransform, 0, reverse ? SlideOffset : -SlideOffset);
            AddFade(storyboard, outgoing, 1, 0);
        }

        await RunStoryboardAsync(storyboard);

        if (outgoing is not null)
        {
            RootHost.Children.Remove(outgoing);
            outgoing.Opacity = 1;
            outgoing.RenderTransform = null;
        }
    }

    static void AddSlide(Storyboard storyboard, TranslateTransform transform, double from, double to)
    {
        DoubleAnimation animation = new()
        {
            From = from,
            To = to,
            Duration = TimeSpan.FromMilliseconds(250),
            EasingFunction = new CubicEase { EasingMode = EasingMode.EaseOut },
        };
        Storyboard.SetTarget(animation, transform);
        Storyboard.SetTargetProperty(animation, "X");
        storyboard.Children.Add(animation);
    }

    static void AddFade(Storyboard storyboard, UIElement target, double from, double to)
    {
        DoubleAnimation animation = new()
        {
            From = from,
            To = to,
            Duration = TimeSpan.FromMilliseconds(250),
        };
        Storyboard.SetTarget(animation, target);
        Storyboard.SetTargetProperty(animation, "Opacity");
        storyboard.Children.Add(animation);
    }

    static Task RunStoryboardAsync(Storyboard storyboard)
    {
        TaskCompletionSource completion = new();
        storyboard.Completed += (_, _) => completion.TrySetResult();
        storyboard.Begin();
        return completion.Task;
    }

    async Task StartBackendAsync()
    {
        // UI スレッドで生成するので、Report は自動的に UI スレッドへマーシャルされる。
        Progress<string> setupProgress = new(ReportEnvironmentSetup);

        try
        {
            int port = await _backendProcessManager
                .StartAsync(AppSettings.LastModelId, AppSettings.WebUiEnabled, onSetupProgress: setupProgress)
                .ConfigureAwait(false);
            _apiClient = new BackendApiClient(port);

            DispatcherQueue.TryEnqueue(() =>
            {
                SetupOverlay.Visibility = Visibility.Collapsed;

                // サーバー自体はここで応答可能になっているが、初回モデルはバックグラウンドで
                // まだ VRAM に展開中のことがある。ギャラリーはモデル不要なので即座に使えるが、
                // モデル切り替え・生成はロード完了を待つ必要があるため、ModelSelectionButton は
                // _generationPage.AttachBackend が IsBackendReady を立てるまで無効のままにする。
                GalleryButton.IsEnabled = true;

                _generationPage.AttachBackend(_apiClient);

                UpdateWebUiToggleUi();
            });
        }
        catch (UvNotFoundException)
        {
            ShowBackendStartupError(ResourceLoader.GetString("MainWindow_UvNotFound"));
        }
        catch (Exception ex)
        {
            ShowBackendStartupError(
                string.Format(ResourceLoader.GetString("MainWindow_BackendFailedToStart"), ex.Message));
        }
    }

    // 初回セットアップ(uv sync)中に呼ばれる。開始合図(空文字)でオーバーレイを表示し、
    // 以降は uv sync のログ行を実況表示する。Progress<string> 経由なので UI スレッドで呼ばれる。
    void ReportEnvironmentSetup(string logLine)
    {
        SetupOverlay.Visibility = Visibility.Visible;

        if (!string.IsNullOrEmpty(logLine))
        {
            SetupLogTextBlock.Text = logLine;
        }
    }

    void ShowBackendStartupError(string message) =>
        DispatcherQueue.TryEnqueue(() =>
        {
            SetupOverlay.Visibility = Visibility.Collapsed;
            RootHost.Children.Clear();
            RootHost.Children.Add(new TextBlock
            {
                Text = message,
                Margin = new Thickness(20),
                TextWrapping = TextWrapping.Wrap,
            });
        });

    async void ModelSelectionButton_Click(object sender, RoutedEventArgs e)
    {
        if (_apiClient is not BackendApiClient apiClient)
        {
            return;
        }

        // GenerationPage はイベント購読済みのインスタンスを保持しており、戻る際にこれへ復帰させて
        // 状態を維持する。モデル選択ページを右から重ねて横スライドで前面に出す。
        ModelSelectionPage page = new();
        page.ModelSwitched += ModelSelectionPage_ModelSwitched;
        page.BackRequested += ModelSelectionPage_BackRequested;
        page.Initialize(apiClient, this, _generationPage.SelectedLoras);
        await SlideToPageAsync(page, reverse: false);
    }

    void ModelSelectionPage_ModelSwitched(object? sender, EventArgs e)
    {
        if (_apiClient is BackendApiClient apiClient)
        {
            _ = _generationPage.RefreshDeviceInfoAsync(apiClient);
        }
    }

    async void ModelSelectionPage_BackRequested(object? sender, EventArgs e) =>
        await SlideToPageAsync(_generationPage, reverse: true);

    async void GalleryButton_Click(object sender, RoutedEventArgs e)
    {
        if (_apiClient is not BackendApiClient apiClient)
        {
            return;
        }

        _galleryPage.Initialize(apiClient);
        await SlideToPageAsync(_galleryPage, reverse: false);
    }

    async void GalleryPage_BackRequested(object? sender, EventArgs e) =>
        await SlideToPageAsync(_generationPage, reverse: true);

    async void GalleryPage_ReuseParametersRequested(object? sender, GalleryImageInfo image)
    {
        _generationPage.ApplyHistoryParameters(image);
        await SlideToPageAsync(_generationPage, reverse: true);
    }

    void SkeletonScreenToggleButton_Click(object sender, RoutedEventArgs e)
    {
        bool isEnabled = _generationPage.ToggleSkeletonScreen();
        SkeletonScreenToggleIcon.Glyph = isEnabled ? "" : "";
        SkeletonScreenToggleButton.Opacity = isEnabled ? 0.7 : 1.0;
    }

    // webui の有効・無効はバックエンドプロセス起動時の --webui フラグでしか切り替えられないため、
    // ここでは次回起動用の設定を保存するだけで、既に起動中のバックエンドには反映しない。
    void WebUiToggleButton_Click(object sender, RoutedEventArgs e)
    {
        AppSettings.WebUiEnabled = !AppSettings.WebUiEnabled;
        UpdateWebUiToggleUi();
        StatusBarTextBlock.Text = ResourceLoader.GetString("MainWindow_WebUiRestartRequired");
    }

    void UpdateWebUiToggleUi()
    {
        bool isEnabled = AppSettings.WebUiEnabled;
        WebUiToggleButton.Opacity = isEnabled ? 0.7 : 1.0;

        // 起動済みの webui は URL を、それ以外はオン・オフの説明を表示する。
        string tooltip = isEnabled && _backendProcessManager.WebUiUrl is string webUiUrl
            ? string.Format(ResourceLoader.GetString("MainWindow_WebUiToggleButton_Url"), webUiUrl)
            : ResourceLoader.GetString(isEnabled ? "MainWindow_WebUiToggleButton_On" : "MainWindow_WebUiToggleButton_Off");
        ToolTipService.SetToolTip(WebUiToggleButton, tooltip);
    }

    async void MainWindow_Closed(object sender, WindowEventArgs args)
    {
        _systemStatsTimer.Stop();
        _apiClient?.Dispose();
        await _backendProcessManager.DisposeAsync();
    }
}
