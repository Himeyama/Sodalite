using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using SDApp.Services;
using SDApp.Views;

namespace SDApp;

public sealed partial class MainWindow : Window
{
    readonly BackendProcessManager _backendProcessManager = new(BackendProjectPath);
    BackendApiClient? _apiClient;

    static string BackendProjectPath =>
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "..", "backend"));

    public MainWindow()
    {
        InitializeComponent();

        AppWindow.SetIcon("Assets/AppIcon.ico");
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(TitleBarGrid);

        Closed += MainWindow_Closed;

        RootFrame.Navigate(typeof(GenerationPage));

        if (RootFrame.Content is GenerationPage generationPage)
        {
            generationPage.StatusChanged += (_, status) => StatusBarTextBlock.Text = status;
            generationPage.DeviceInfoChanged += (_, deviceInfo) => StatusBarDeviceInfoTextBlock.Text = deviceInfo;
        }

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
            });
        }
        catch (Exception ex)
        {
            DispatcherQueue.TryEnqueue(() =>
                RootFrame.Content = new TextBlock
                {
                    Text = $"Backend failed to start: {ex.Message}",
                    Margin = new Thickness(20),
                });
        }
    }

    async void MainWindow_Closed(object sender, WindowEventArgs args)
    {
        _apiClient?.Dispose();
        await _backendProcessManager.DisposeAsync();
    }
}
