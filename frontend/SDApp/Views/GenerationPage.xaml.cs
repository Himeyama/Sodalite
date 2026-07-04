using System.ComponentModel;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Navigation;
using SDApp.Services;
using SDApp.ViewModels;

namespace SDApp.Views;

public sealed partial class GenerationPage : Page
{
    const string BrailleSpinnerFrames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";

    readonly DispatcherQueueTimer _backendStartingSpinnerTimer;

    GenerationViewModel? _viewModel;
    int _backendStartingSpinnerIndex;

    public event EventHandler<string>? StatusChanged;

    public event EventHandler<string>? DeviceInfoChanged;

    public GenerationPage()
    {
        InitializeComponent();

        _backendStartingSpinnerTimer = DispatcherQueue.CreateTimer();
        _backendStartingSpinnerTimer.Interval = TimeSpan.FromMilliseconds(80);
        _backendStartingSpinnerTimer.Tick += (_, _) =>
        {
            _backendStartingSpinnerIndex = (_backendStartingSpinnerIndex + 1) % BrailleSpinnerFrames.Length;
            GenerateButton.Content = $"{BrailleSpinnerFrames[_backendStartingSpinnerIndex]} Backend starting...";
        };
    }

    protected override void OnNavigatedTo(NavigationEventArgs e)
    {
        base.OnNavigatedTo(e);

        _viewModel = new GenerationViewModel(DispatcherQueue);
        _viewModel.PropertyChanged += ViewModel_PropertyChanged;

        GenerateButton.IsEnabled = false;
        StatusChanged?.Invoke(this, _viewModel.StatusText);
        _backendStartingSpinnerTimer.Start();
    }

    internal void AttachBackend(BackendApiClient apiClient)
    {
        if (_viewModel is not GenerationViewModel viewModel)
        {
            return;
        }

        _ = viewModel.AttachBackendAsync(apiClient, CancellationToken.None);
    }

    void ViewModel_PropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (_viewModel is not GenerationViewModel viewModel)
        {
            return;
        }

        switch (e.PropertyName)
        {
            case nameof(GenerationViewModel.StatusText):
                StatusChanged?.Invoke(this, viewModel.StatusText);
                break;
            case nameof(GenerationViewModel.ResultImage):
                ResultImageControl.Source = viewModel.ResultImage;
                break;
            case nameof(GenerationViewModel.IsBackendReady):
                if (viewModel.IsBackendReady)
                {
                    _backendStartingSpinnerTimer.Stop();
                    GenerateButton.Content = "Generate";
                }

                GenerateButton.IsEnabled = viewModel.IsBackendReady && !viewModel.IsGenerating;
                break;
            case nameof(GenerationViewModel.IsGenerating):
                GenerateButton.IsEnabled = viewModel.IsBackendReady && !viewModel.IsGenerating;
                GenerationProgressRing.IsActive = viewModel.IsGenerating;
                break;
            case nameof(GenerationViewModel.Samplers):
                SamplerComboBox.ItemsSource = viewModel.Samplers;
                if (viewModel.Samplers.Count > 0)
                {
                    SamplerComboBox.SelectedIndex = 0;
                }

                break;
            case nameof(GenerationViewModel.DeviceInfo):
                DeviceInfoChanged?.Invoke(this, viewModel.DeviceInfo);
                break;
        }
    }

    async void GenerateButton_Click(object sender, RoutedEventArgs e)
    {
        if (_viewModel is not GenerationViewModel viewModel)
        {
            return;
        }

        viewModel.Prompt = PromptTextBox.Text;
        viewModel.NegativePrompt = NegativePromptTextBox.Text;
        viewModel.Steps = (int)StepsSlider.Value;
        viewModel.CfgScale = CfgScaleSlider.Value;
        viewModel.Width = (int)WidthNumberBox.Value;
        viewModel.Height = (int)HeightNumberBox.Value;
        viewModel.Sampler = SamplerComboBox.SelectedItem as string ?? viewModel.Sampler;
        viewModel.SeedText = SeedTextBox.Text;

        await viewModel.GenerateAsync(CancellationToken.None);
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
}
