using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.Windows.ApplicationModel.Resources;
using Sodalite.Services;
using Sodalite.Views;

namespace Sodalite;

public sealed partial class MainWindow : Window
{
    static readonly ResourceLoader ResourceLoader = new();

    readonly BackendProcessManager _backendProcessManager = new(BackendLocator.BackendProjectPath);
    BackendApiClient? _apiClient;

    public MainWindow()
    {
        InitializeComponent();

        Title = ResourceLoader.GetString("MainWindow_Title");
        AppWindow.SetIcon("Assets/AppIcon.ico");
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(TitleBarGrid);

        Closed += MainWindow_Closed;

        GenerationPage generationPage = new();
        generationPage.OwnerWindow = this;
        generationPage.StatusChanged += (_, status) => StatusBarTextBlock.Text = status;
        generationPage.DeviceInfoChanged += (_, deviceInfo) => StatusBarDeviceInfoTextBlock.Text = deviceInfo;
        generationPage.NotifyCurrentStatus();

        RootFrame.Content = generationPage;

        _ = StartBackendAsync();
    }

    async Task StartBackendAsync()
    {
        // UI スレッドで生成するので、Report は自動的に UI スレッドへマーシャルされる。
        Progress<string> setupProgress = new(ReportEnvironmentSetup);

        try
        {
            int port = await _backendProcessManager
                .StartAsync(AppSettings.LastModelId, onSetupProgress: setupProgress)
                .ConfigureAwait(false);
            _apiClient = new BackendApiClient(port);

            DispatcherQueue.TryEnqueue(() =>
            {
                SetupOverlay.Visibility = Visibility.Collapsed;

                if (RootFrame.Content is GenerationPage generationPage)
                {
                    generationPage.AttachBackend(_apiClient);
                }

                ModelSelectionButton.IsEnabled = true;
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
            RootFrame.Content = new TextBlock
            {
                Text = message,
                Margin = new Thickness(20),
                TextWrapping = TextWrapping.Wrap,
            };
        });

    async void ModelSelectionButton_Click(object sender, RoutedEventArgs e)
    {
        if (_apiClient is not BackendApiClient apiClient)
        {
            return;
        }

        if (RootFrame.Content is not GenerationPage generationPage)
        {
            return;
        }

        ModelSelectionDialog dialog = new(apiClient, this, generationPage.SelectedLoras)
        {
            XamlRoot = Content.XamlRoot,
        };
        await dialog.ShowAsync();

        if (dialog.SelectedModelId is not null)
        {
            await generationPage.RefreshDeviceInfoAsync(apiClient);
        }
    }

    async void MainWindow_Closed(object sender, WindowEventArgs args)
    {
        _apiClient?.Dispose();
        await _backendProcessManager.DisposeAsync();
    }
}
