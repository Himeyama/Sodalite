using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.InteropServices.WindowsRuntime;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.Windows.ApplicationModel.Resources;
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

    public event EventHandler<string>? StatusChanged;

    public event EventHandler<string>? DeviceInfoChanged;

    /// <summary>FileSavePicker の初期化に使うオーナーウィンドウ。MainWindow から注入する。</summary>
    internal Window? OwnerWindow { get; set; }

    public GenerationPage()
    {
        InitializeComponent();

        _viewModel = new GenerationViewModel(DispatcherQueue);
        _viewModel.PropertyChanged += ViewModel_PropertyChanged;

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
                }

                GenerateButton.IsEnabled = _viewModel.IsBackendReady && !_viewModel.IsGenerating;
                break;
            case nameof(GenerationViewModel.IsGenerating):
                GenerateButton.IsEnabled = _viewModel.IsBackendReady && !_viewModel.IsGenerating;
                ResultImageControl.Visibility = _viewModel.IsGenerating ? Visibility.Collapsed : Visibility.Visible;
                SkeletonScreenGrid.Visibility = _viewModel.IsGenerating ? Visibility.Visible : Visibility.Collapsed;
                if (_viewModel.IsGenerating)
                {
                    SkeletonShimmerStoryboard.Begin();
                }
                else
                {
                    SkeletonShimmerStoryboard.Stop();
                }

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
        _viewModel.Sampler = SamplerComboBox.SelectedItem as string ?? _viewModel.Sampler;
        _viewModel.SeedText = SeedTextBox.Text;

        _skeletonAspectRatio = (double)_viewModel.Width / _viewModel.Height;
        UpdateSkeletonScreenSize(SkeletonScreenGrid.ActualSize.X, SkeletonScreenGrid.ActualSize.Y);

        await _viewModel.GenerateAsync(CancellationToken.None);
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

    void ResultImageFlyout_Opening(object? sender, object e)
    {
        bool hasImage = _viewModel.ResultImageBytes is not null;
        CopyImageMenuItem.IsEnabled = hasImage;
        SaveImageMenuItem.IsEnabled = hasImage;
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
