using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.Windows.ApplicationModel.Resources;
using SDApp.Services;
using SDApp.Views;

namespace SDApp;

public sealed partial class MainWindow : Window
{
    static readonly ResourceLoader ResourceLoader = new();

    readonly BackendProcessManager _backendProcessManager = new(BackendProjectPath);
    BackendApiClient? _apiClient;

    static string BackendProjectPath =>
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "..", "backend"));

    public MainWindow()
    {
        InitializeComponent();

        Title = ResourceLoader.GetString("MainWindow_Title");
        AppWindow.SetIcon("Assets/AppIcon.ico");
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(TitleBarGrid);

        Closed += MainWindow_Closed;

        GenerationPage generationPage = new();
        generationPage.StatusChanged += (_, status) => StatusBarTextBlock.Text = status;
        generationPage.DeviceInfoChanged += (_, deviceInfo) => StatusBarDeviceInfoTextBlock.Text = deviceInfo;
        generationPage.NotifyCurrentStatus();

        RootFrame.Content = generationPage;

        _ = StartBackendAsync();
    }

    async Task StartBackendAsync()
    {
        try
        {
            int port = await _backendProcessManager.StartAsync().ConfigureAwait(false);
            _apiClient = new BackendApiClient(port);

            DispatcherQueue.TryEnqueue(() =>
            {
                if (RootFrame.Content is GenerationPage generationPage)
                {
                    generationPage.AttachBackend(_apiClient);
                }

                ModelSelectionButton.IsEnabled = true;
            });
        }
        catch (Exception ex)
        {
            DispatcherQueue.TryEnqueue(() =>
                RootFrame.Content = new TextBlock
                {
                    Text = string.Format(ResourceLoader.GetString("MainWindow_BackendFailedToStart"), ex.Message),
                    Margin = new Thickness(20),
                });
        }
    }

    async void ModelSelectionButton_Click(object sender, RoutedEventArgs e)
    {
        if (_apiClient is not BackendApiClient apiClient)
        {
            return;
        }

        ModelSelectionDialog dialog = new(apiClient, this) { XamlRoot = Content.XamlRoot };
        await dialog.ShowAsync();

        if (dialog.SelectedModelId is not null && RootFrame.Content is GenerationPage generationPage)
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
