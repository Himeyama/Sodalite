using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices.WindowsRuntime;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.Windows.ApplicationModel.Resources;
using Sodalite.Models;
using Sodalite.Services;
using Sodalite.ViewModels;
using Windows.ApplicationModel.DataTransfer;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.Storage.Streams;
using WinRT.Interop;

namespace Sodalite.Views;

public sealed partial class GenerationPage : Page
{
    const string BrailleSpinnerFrames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";

    static readonly ResourceLoader ResourceLoader = new();

    readonly DispatcherQueueTimer _backendStartingSpinnerTimer;
    readonly GenerationViewModel _viewModel;

    int _backendStartingSpinnerIndex;
    double _skeletonAspectRatio = 1.0;
    bool _isSkeletonScreenEnabled = true;
    (string Prompt, string NegativePrompt)? _promptStateBeforeChatUpdate;

    public event EventHandler<string>? StatusChanged;

    public event EventHandler<string>? DeviceInfoChanged;

    /// <summary>初回モデルのロードが完了し、生成・モデル切り替えが可能になったときに発火する。</summary>
    public event EventHandler? BackendReadyChanged;

    public event EventHandler<bool>? ChatAvailabilityChanged;

    public event EventHandler<string?>? ChatModelChanged;

    /// <summary>FileSavePicker の初期化に使うオーナーウィンドウ。MainWindow から注入する。</summary>
    internal Window? OwnerWindow { get; set; }

    public GenerationPage()
    {
        InitializeComponent();

        _viewModel = new GenerationViewModel(DispatcherQueue);
        _viewModel.PropertyChanged += ViewModel_PropertyChanged;
        ChatPanelControl.AvailabilityChanged += (_, available) => ChatAvailabilityChanged?.Invoke(this, available);
        ChatPanelControl.SelectedModelChanged += (_, model) => ChatModelChanged?.Invoke(this, model);
        ChatPanelControl.PromptUpdateRequested += (_, update) => ApplyChatPromptUpdate(update);

        GenerateButton.IsEnabled = false;

        string backendStartingText = ResourceLoader.GetString("Generation_BackendStarting");
        _backendStartingSpinnerTimer = DispatcherQueue.CreateTimer();
        _backendStartingSpinnerTimer.Interval = TimeSpan.FromMilliseconds(80);
        _backendStartingSpinnerTimer.Tick += (_, _) =>
        {
            _backendStartingSpinnerIndex = (_backendStartingSpinnerIndex + 1) % BrailleSpinnerFrames.Length;
            GenerateButton.Content = $"{BrailleSpinnerFrames[_backendStartingSpinnerIndex]} {backendStartingText}";
        };
        _backendStartingSpinnerTimer.Start();
    }

    /// <summary>現在の状態でイベントを再発火する。購読側がページ生成後に接続した場合の初期同期用。</summary>
    internal void NotifyCurrentStatus()
    {
        StatusChanged?.Invoke(this, _viewModel.StatusText);
        DeviceInfoChanged?.Invoke(this, _viewModel.DeviceInfo);
    }

    internal void AttachBackend(BackendApiClient apiClient) =>
        _ = _viewModel.AttachBackendAsync(apiClient, CancellationToken.None);

    /// <summary>選択済み LoRA コレクション。モデル選択ダイアログがこれを直接編集する。</summary>
    internal ObservableCollection<SelectedLoraViewModel> SelectedLoras => _viewModel.SelectedLoras;

    internal Task RefreshDeviceInfoAsync(BackendApiClient apiClient) =>
        _viewModel.RefreshDeviceInfoAsync(apiClient, CancellationToken.None);

    /// <summary>llama.cpp の稼働確認とモデル一覧取得を開始する。失敗時はチャットを表示しない。</summary>
    internal Task InitializeChatAsync() =>
        ChatPanelControl.InitializeAsync(() => (PromptTextBox.Text, NegativePromptTextBox.Text));

    internal void SetChatOpen(bool isOpen)
    {
        ChatPanelControl.Visibility = isOpen ? Visibility.Visible : Visibility.Collapsed;
        // ChatPanel が非表示なら列幅もなくし、従来どおり設定・結果を広く使う。
        ((ColumnDefinition)((Grid)Content).ColumnDefinitions[0]).Width = isOpen
            ? new GridLength(320)
            : new GridLength(0);
    }

    internal void DisposeChat() => ChatPanelControl.Dispose();

    void ApplyChatPromptUpdate(ChatPromptUpdate update)
    {
        _promptStateBeforeChatUpdate = (PromptTextBox.Text, NegativePromptTextBox.Text);
        PromptTextBox.Text = update.Prompt;
        NegativePromptTextBox.Text = update.NegativePrompt;
        UndoChatPromptButton.IsEnabled = true;
        UndoChatNegativePromptButton.IsEnabled = true;
    }

    void UndoChatPromptButton_Click(object sender, RoutedEventArgs e)
    {
        if (_promptStateBeforeChatUpdate is not (string prompt, string negativePrompt))
        {
            return;
        }

        PromptTextBox.Text = prompt;
        NegativePromptTextBox.Text = negativePrompt;
        _promptStateBeforeChatUpdate = null;
        UndoChatPromptButton.IsEnabled = false;
        UndoChatNegativePromptButton.IsEnabled = false;
    }

    /// <summary>ギャラリーで選択した履歴のパラメータを入力欄・LoRA選択に復元する。
    /// GenerateButton_Click は「コントロール値→ViewModel」の一方向同期のため、ViewModel
    /// プロパティではなくコントロール自体の値を更新する(そうしないと表示に反映されない)。</summary>
    internal void ApplyHistoryParameters(GalleryImageInfo image)
    {
        if (image.Parameters is not GalleryParameters parameters)
        {
            return;
        }

        PromptTextBox.Text = parameters.Prompt;
        NegativePromptTextBox.Text = parameters.NegativePrompt;

        if (parameters.Steps is int steps)
        {
            StepsSlider.Value = steps;
        }

        if (parameters.CfgScale is double cfgScale)
        {
            CfgScaleSlider.Value = cfgScale;
        }

        if (parameters.Width is int width)
        {
            WidthNumberBox.Value = width;
        }

        if (parameters.Height is int height)
        {
            HeightNumberBox.Value = height;
        }

        if (parameters.BatchSize is int batchSize)
        {
            BatchSizeNumberBox.Value = batchSize;
        }

        SeedTextBox.Text = parameters.Seed?.ToString() ?? "";

        if (parameters.Sampler is string sampler && _viewModel.Samplers.Contains(sampler))
        {
            SamplerComboBox.SelectedItem = sampler;
        }

        _viewModel.SelectedLoras.Clear();
        foreach (LoraSelection lora in parameters.Loras)
        {
            _viewModel.SelectedLoras.Add(new SelectedLoraViewModel(lora.ModelId) { Weight = lora.Weight });
        }
    }

    /// <summary>生成中に表示するスケルトンスクリーンのオン・オフを切り替え、切替後の状態を返す。
    /// オフの場合、生成中も直近の生成結果画像を表示し続ける。</summary>
    internal bool ToggleSkeletonScreen()
    {
        _isSkeletonScreenEnabled = !_isSkeletonScreenEnabled;
        UpdateResultAreaVisibility();
        return _isSkeletonScreenEnabled;
    }

    void UpdateResultAreaVisibility()
    {
        bool showSkeleton = _viewModel.IsGenerating && _isSkeletonScreenEnabled;

        ResultImageControl.Visibility = showSkeleton ? Visibility.Collapsed : Visibility.Visible;
        SkeletonScreenGrid.Visibility = showSkeleton ? Visibility.Visible : Visibility.Collapsed;

        if (showSkeleton)
        {
            SkeletonShimmerStoryboard.Begin();
        }
        else
        {
            SkeletonShimmerStoryboard.Stop();
        }
    }

    void ViewModel_PropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        switch (e.PropertyName)
        {
            case nameof(GenerationViewModel.StatusText):
                StatusChanged?.Invoke(this, _viewModel.StatusText);
                break;
            case nameof(GenerationViewModel.ResultImage):
                ResultImageControl.Source = _viewModel.ResultImage;
                break;
            case nameof(GenerationViewModel.IsBackendReady):
                if (_viewModel.IsBackendReady)
                {
                    _backendStartingSpinnerTimer.Stop();
                    GenerateButton.Content = ResourceLoader.GetString("Generation_GenerateButtonLabel");
                    BackendReadyChanged?.Invoke(this, EventArgs.Empty);
                }

                GenerateButton.IsEnabled = _viewModel.IsBackendReady && !_viewModel.IsGenerating;
                break;
            case nameof(GenerationViewModel.IsGenerating):
                GenerateButton.IsEnabled = _viewModel.IsBackendReady && !_viewModel.IsGenerating;
                GenerateButton.Visibility = _viewModel.IsGenerating ? Visibility.Collapsed : Visibility.Visible;
                CancelButton.Visibility = _viewModel.IsGenerating ? Visibility.Visible : Visibility.Collapsed;
                CancelButton.IsEnabled = _viewModel.IsGenerating;
                SelectImageButton.IsEnabled = !_viewModel.IsGenerating;
                ClearImageButton.IsEnabled = !_viewModel.IsGenerating && _viewModel.InitialImage is not null;
                UpdateResultAreaVisibility();
                break;
            case nameof(GenerationViewModel.Samplers):
                SamplerComboBox.ItemsSource = _viewModel.Samplers;
                if (_viewModel.Samplers.Count > 0)
                {
                    SamplerComboBox.SelectedIndex = 0;
                }

                break;
            case nameof(GenerationViewModel.DeviceInfo):
                DeviceInfoChanged?.Invoke(this, _viewModel.DeviceInfo);
                break;
        }
    }

    async void GenerateButton_Click(object sender, RoutedEventArgs e)
    {
        _viewModel.Prompt = PromptTextBox.Text;
        _viewModel.NegativePrompt = NegativePromptTextBox.Text;
        _viewModel.Steps = (int)StepsSlider.Value;
        _viewModel.CfgScale = CfgScaleSlider.Value;
        _viewModel.Width = (int)WidthNumberBox.Value;
        _viewModel.Height = (int)HeightNumberBox.Value;
        _viewModel.BatchSize = (int)BatchSizeNumberBox.Value;
        _viewModel.Sampler = SamplerComboBox.SelectedItem as string ?? _viewModel.Sampler;
        _viewModel.SeedText = SeedTextBox.Text;
        _viewModel.Strength = StrengthSlider.Value;

        _skeletonAspectRatio = (double)_viewModel.Width / _viewModel.Height;
        UpdateSkeletonScreenSize(SkeletonScreenGrid.ActualSize.X, SkeletonScreenGrid.ActualSize.Y);

        await _viewModel.GenerateAsync(CancellationToken.None);
    }

    async void PastePromptButton_Click(object sender, RoutedEventArgs e) =>
        PromptTextBox.Text = await GetClipboardTextAsync();

    async void PasteNegativePromptButton_Click(object sender, RoutedEventArgs e) =>
        NegativePromptTextBox.Text = await GetClipboardTextAsync();

    static async Task<string> GetClipboardTextAsync()
    {
        DataPackageView clipboard = Clipboard.GetContent();
        return clipboard.Contains(StandardDataFormats.Text)
            ? await clipboard.GetTextAsync()
            : string.Empty;
    }

    async void CancelButton_Click(object sender, RoutedEventArgs e)
    {
        CancelButton.IsEnabled = false;
        await _viewModel.CancelAsync(CancellationToken.None);
    }

    void StepsSlider_ValueChanged(object sender, Microsoft.UI.Xaml.Controls.Primitives.RangeBaseValueChangedEventArgs e)
    {
        if (StepsValueTextBlock is not null)
        {
            StepsValueTextBlock.Text = ((int)e.NewValue).ToString();
        }
    }

    void CfgScaleSlider_ValueChanged(object sender, Microsoft.UI.Xaml.Controls.Primitives.RangeBaseValueChangedEventArgs e)
    {
        if (CfgScaleValueTextBlock is not null)
        {
            CfgScaleValueTextBlock.Text = e.NewValue.ToString("F1");
        }
    }

    void StrengthSlider_ValueChanged(object sender, Microsoft.UI.Xaml.Controls.Primitives.RangeBaseValueChangedEventArgs e)
    {
        if (StrengthValueTextBlock is not null)
        {
            StrengthValueTextBlock.Text = e.NewValue.ToString("F2");
        }
    }

    async void SelectImageButton_Click(object sender, RoutedEventArgs e)
    {
        if (OwnerWindow is not Window ownerWindow)
        {
            return;
        }

        FileOpenPicker picker = new();
        InitializeWithWindow.Initialize(picker, WindowNative.GetWindowHandle(ownerWindow));
        picker.SuggestedStartLocation = PickerLocationId.PicturesLibrary;
        picker.FileTypeFilter.Add(".png");
        picker.FileTypeFilter.Add(".jpg");
        picker.FileTypeFilter.Add(".jpeg");
        picker.FileTypeFilter.Add(".webp");
        picker.FileTypeFilter.Add(".bmp");

        StorageFile? file = await picker.PickSingleFileAsync();
        if (file is null)
        {
            return;
        }

        IBuffer imageBuffer = await FileIO.ReadBufferAsync(file);
        _viewModel.InitialImage = Convert.ToBase64String(imageBuffer.ToArray());
        SelectedImageTextBlock.Text = file.Name;
        ClearImageButton.IsEnabled = true;
        StrengthSlider.IsEnabled = true;
        StepsSlider.Value = 40;
    }

    void ClearImageButton_Click(object sender, RoutedEventArgs e)
    {
        _viewModel.InitialImage = null;
        SelectedImageTextBlock.Text = ResourceLoader.GetString("Generation_NoImageSelected/Text");
        ClearImageButton.IsEnabled = false;
        StrengthSlider.IsEnabled = false;
        StepsSlider.Value = 20;
    }

    void ResultImageFlyout_Opening(object? sender, object e)
    {
        bool hasImage = _viewModel.ResultImageBytes is not null;
        CopyImageMenuItem.IsEnabled = hasImage;
        SaveImageMenuItem.IsEnabled = hasImage;
        OpenContainingFolderMenuItem.IsEnabled = _viewModel.ResultImagePath is not null;
    }

    async void CopyImageMenuItem_Click(object sender, RoutedEventArgs e)
    {
        if (_viewModel.ResultImageBytes is not byte[] imageBytes)
        {
            return;
        }

        InMemoryRandomAccessStream stream = new();
        await stream.WriteAsync(imageBytes.AsBuffer());
        stream.Seek(0);

        DataPackage dataPackage = new();
        dataPackage.SetBitmap(RandomAccessStreamReference.CreateFromStream(stream));
        Clipboard.SetContent(dataPackage);
    }

    async void SaveImageMenuItem_Click(object sender, RoutedEventArgs e)
    {
        if (_viewModel.ResultImageBytes is not byte[] imageBytes || OwnerWindow is not Window ownerWindow)
        {
            return;
        }

        FileSavePicker picker = new();
        InitializeWithWindow.Initialize(picker, WindowNative.GetWindowHandle(ownerWindow));
        picker.SuggestedStartLocation = PickerLocationId.PicturesLibrary;
        picker.FileTypeChoices.Add("PNG", [".png"]);
        picker.SuggestedFileName = $"sodalite_{DateTime.Now:yyyyMMdd_HHmmss}";

        StorageFile? file = await picker.PickSaveFileAsync();
        if (file is null)
        {
            return;
        }

        await FileIO.WriteBytesAsync(file, imageBytes);
    }

    void OpenContainingFolderMenuItem_Click(object sender, RoutedEventArgs e)
    {
        if (_viewModel.ResultImagePath is not string imagePath)
        {
            return;
        }

        using Process? process = Process.Start(new ProcessStartInfo
        {
            FileName = "explorer.exe",
            ArgumentList = { $"/select,{imagePath}" },
            UseShellExecute = false,
        });
    }

    void SkeletonScreenGrid_SizeChanged(object sender, SizeChangedEventArgs e) =>
        UpdateSkeletonScreenSize(e.NewSize.Width, e.NewSize.Height);

    /// <summary>
    /// Image(Stretch="Uniform")と同じく、指定した幅・高さの比率を保ったまま
    /// 表示領域に収まる最大サイズでスケルトンのプレースホルダーを中央配置する。
    /// </summary>
    void UpdateSkeletonScreenSize(double containerWidth, double containerHeight)
    {
        if (containerWidth <= 0 || containerHeight <= 0)
        {
            return;
        }

        double width = containerWidth;
        double height = width / _skeletonAspectRatio;
        if (height > containerHeight)
        {
            height = containerHeight;
            width = height * _skeletonAspectRatio;
        }

        SkeletonScreenBorder.Width = width;
        SkeletonScreenBorder.Height = height;
        SkeletonClipGeometry.Rect = new Windows.Foundation.Rect(0, 0, width, height);
        SkeletonShimmerTransform.X = -SkeletonShimmerRectangle.Width;
        SkeletonShimmerAnimation.From = -SkeletonShimmerRectangle.Width;
        SkeletonShimmerAnimation.To = width + SkeletonShimmerRectangle.Width;
    }
}
