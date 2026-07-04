using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.Windows.ApplicationModel.Resources;
using SDApp.Models;
using SDApp.Services;
using Windows.Storage.Pickers;
using WinRT.Interop;

namespace SDApp.Views;

sealed partial class ModelSelectionDialog : ContentDialog
{
    static readonly ResourceLoader ResourceLoader = new();

    readonly BackendApiClient _apiClient;
    readonly Window _ownerWindow;

    public string? SelectedModelId { get; private set; }

    public ModelSelectionDialog(BackendApiClient apiClient, Window ownerWindow)
    {
        InitializeComponent();
        _apiClient = apiClient;
        _ownerWindow = ownerWindow;

        Loaded += ModelSelectionDialog_Loaded;
    }

    async void ModelSelectionDialog_Loaded(object sender, RoutedEventArgs e)
    {
        await LoadModelsAsync();
    }

    async Task LoadModelsAsync()
    {
        ErrorInfoBar.IsOpen = false;
        LoadingProgressRing.IsActive = true;
        ModelListView.IsEnabled = false;

        try
        {
            List<ModelInfo> models = await _apiClient.GetModelsAsync(CancellationToken.None).ConfigureAwait(true);
            ModelListView.ItemsSource = models
                .Select(model => new ModelListItem(model.ModelId, model.IsActive, DisplayNameFor(model.ModelId)))
                .ToList();
        }
        catch (Exception ex)
        {
            ModelListView.ItemsSource = new List<ModelListItem>();
            ShowError(ResourceLoader.GetString("ModelSelectionDialog_LoadErrorTitle"), ex.Message);
        }
        finally
        {
            LoadingProgressRing.IsActive = false;
            ModelListView.IsEnabled = true;
        }
    }

    async void ModelListView_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (ModelListView.SelectedItem is not ModelListItem item || item.IsActive)
        {
            return;
        }

        ErrorInfoBar.IsOpen = false;
        ModelListView.IsEnabled = false;
        LoadingProgressRing.IsActive = true;

        try
        {
            ModelInfo result = await _apiClient.SetActiveModelAsync(item.ModelId, CancellationToken.None).ConfigureAwait(true);
            SelectedModelId = result.ModelId;
            Hide();
        }
        catch (Exception ex)
        {
            ModelListView.SelectedItem = null;
            ShowError(ResourceLoader.GetString("ModelSelectionDialog_SwitchErrorTitle"), ex.Message);
        }
        finally
        {
            LoadingProgressRing.IsActive = false;
            ModelListView.IsEnabled = true;
        }
    }

    async void ImportButton_Click(object sender, RoutedEventArgs e)
    {
        FileOpenPicker picker = new();
        InitializeWithWindow.Initialize(picker, WindowNative.GetWindowHandle(_ownerWindow));
        picker.FileTypeFilter.Add(".safetensors");
        picker.FileTypeFilter.Add(".ckpt");

        Windows.Storage.StorageFile? file = await picker.PickSingleFileAsync();
        if (file is null)
        {
            return;
        }

        ErrorInfoBar.IsOpen = false;
        ModelListView.IsEnabled = false;
        LoadingProgressRing.IsActive = true;

        try
        {
            await _apiClient.ImportModelAsync(file.Path, CancellationToken.None).ConfigureAwait(true);
            await LoadModelsAsync();
        }
        catch (Exception ex)
        {
            ShowError(ResourceLoader.GetString("ModelSelectionDialog_ImportErrorTitle"), ex.Message);
        }
        finally
        {
            LoadingProgressRing.IsActive = false;
            ModelListView.IsEnabled = true;
        }
    }

    void ShowError(string title, string message)
    {
        ErrorInfoBar.Title = title;
        ErrorInfoBar.Message = message;
        ErrorInfoBar.IsOpen = true;
    }

    static string DisplayNameFor(string modelId) =>
        Path.Exists(modelId) ? Path.GetFileName(modelId) : modelId;
}

sealed record ModelListItem(string ModelId, bool IsActive, string DisplayName)
{
    public Visibility ActiveVisibility => IsActive ? Visibility.Visible : Visibility.Collapsed;
}
