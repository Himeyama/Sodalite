using System.Collections.ObjectModel;
using System.Collections.Specialized;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.Windows.ApplicationModel.Resources;
using Sodalite.Models;
using Sodalite.Services;
using Sodalite.ViewModels;
using Windows.Storage.Pickers;
using WinRT.Interop;

namespace Sodalite.Views;

sealed partial class ModelSelectionDialog : ContentDialog
{
    static readonly ResourceLoader ResourceLoader = new();

    readonly BackendApiClient _apiClient;
    readonly Window _ownerWindow;
    readonly ObservableCollection<SelectedLoraViewModel> _selectedLoras;

    List<LoraFileInfo> _availableLoras = [];
    bool _isRemovingModel;

    public string? SelectedModelId { get; private set; }

    public ModelSelectionDialog(
        BackendApiClient apiClient,
        Window ownerWindow,
        ObservableCollection<SelectedLoraViewModel> selectedLoras)
    {
        InitializeComponent();
        _apiClient = apiClient;
        _ownerWindow = ownerWindow;
        _selectedLoras = selectedLoras;

        SelectedLoraItemsControl.ItemsSource = _selectedLoras;
        _selectedLoras.CollectionChanged += SelectedLoras_CollectionChanged;
        UpdateLoraEmptyVisibility();

        Loaded += ModelSelectionDialog_Loaded;
        Closed += ModelSelectionDialog_Closed;
    }

    async void ModelSelectionDialog_Loaded(object sender, RoutedEventArgs e)
    {
        await LoadModelsAsync();
        await RefreshAvailableLorasAsync();
    }

    async Task RefreshAvailableLorasAsync()
    {
        try
        {
            _availableLoras = await _apiClient.GetLorasAsync(CancellationToken.None).ConfigureAwait(true);
        }
        catch (Exception ex)
        {
            _availableLoras = [];
            ShowError(ResourceLoader.GetString("LoraSelectionDialog_LoadErrorTitle"), ex.Message);
        }

        BuildLoraFlyoutItems();
    }

    /// <summary>
    /// LoRA 一覧項目を Flyout 先頭に組み直す。末尾の「インポート」項目 (ImportLoraMenuItem) は
    /// XAML で静的に定義済みなので消さず、その手前だけを差し替える。これにより Flyout は
    /// 常に非空で、確実にクリックで開ける。
    /// </summary>
    void BuildLoraFlyoutItems()
    {
        while (AddLoraMenuFlyout.Items.Count > 0 && AddLoraMenuFlyout.Items[0] != ImportLoraMenuItem)
        {
            AddLoraMenuFlyout.Items.RemoveAt(0);
        }

        int insertAt = 0;
        foreach (LoraFileInfo lora in _availableLoras)
        {
            string loraId = lora.LoraId;
            MenuFlyoutItem item = new() { Text = DisplayNameFor(loraId) };
            item.Click += (_, _) => AddSelectedLora(loraId);
            AddLoraMenuFlyout.Items.Insert(insertAt++, item);
        }

        if (_availableLoras.Count > 0)
        {
            AddLoraMenuFlyout.Items.Insert(insertAt, new MenuFlyoutSeparator());
        }
    }

    async void ImportLoraMenuItem_Click(object sender, RoutedEventArgs e) => await ImportLoraAsync();

    void ModelSelectionDialog_Closed(ContentDialog sender, ContentDialogClosedEventArgs args) =>
        _selectedLoras.CollectionChanged -= SelectedLoras_CollectionChanged;

    void SelectedLoras_CollectionChanged(object? sender, NotifyCollectionChangedEventArgs e) =>
        UpdateLoraEmptyVisibility();

    void UpdateLoraEmptyVisibility() =>
        LoraEmptyTextBlock.Visibility = _selectedLoras.Count == 0 ? Visibility.Visible : Visibility.Collapsed;

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
        // 削除ボタン経由で項目が選択状態になったときはモデル切り替えを走らせない。
        if (_isRemovingModel || ModelListView.SelectedItem is not ModelListItem item || item.IsActive)
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
            AppSettings.LastModelId = result.ModelId;

            // モデルを切り替えても選択中の LoRA と重みはそのまま維持する。SD1.5/SDXL の
            // 互換性が合わない LoRA は生成時にバックエンド側でスキップされる。
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

    async void RemoveModelButton_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not FrameworkElement { DataContext: ModelListItem item })
        {
            return;
        }

        ErrorInfoBar.IsOpen = false;
        _isRemovingModel = true;
        ModelListView.IsEnabled = false;
        LoadingProgressRing.IsActive = true;

        try
        {
            await _apiClient.RemoveImportedModelAsync(item.ModelId, CancellationToken.None).ConfigureAwait(true);
            await LoadModelsAsync();
        }
        catch (Exception ex)
        {
            ShowError(ResourceLoader.GetString("ModelSelectionDialog_RemoveErrorTitle"), ex.Message);
        }
        finally
        {
            LoadingProgressRing.IsActive = false;
            ModelListView.IsEnabled = true;
            _isRemovingModel = false;
        }
    }

    void AddSelectedLora(string loraId)
    {
        if (_selectedLoras.Any(lora => lora.LoraId == loraId))
        {
            return;
        }

        _selectedLoras.Add(new SelectedLoraViewModel(loraId));
    }

    async Task ImportLoraAsync()
    {
        FileOpenPicker picker = new();
        InitializeWithWindow.Initialize(picker, WindowNative.GetWindowHandle(_ownerWindow));
        picker.FileTypeFilter.Add(".safetensors");

        Windows.Storage.StorageFile? file = await picker.PickSingleFileAsync();
        if (file is null)
        {
            return;
        }

        try
        {
            LoraFileInfo imported = await _apiClient.ImportLoraAsync(file.Path, CancellationToken.None).ConfigureAwait(true);
            if (!_availableLoras.Any(lora => lora.LoraId == imported.LoraId))
            {
                _availableLoras.Add(imported);
            }

            AddSelectedLora(imported.LoraId);
        }
        catch (Exception ex)
        {
            ShowError(ResourceLoader.GetString("LoraSelectionDialog_ImportErrorTitle"), ex.Message);
        }
    }

    void RemoveLoraButton_Click(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { DataContext: SelectedLoraViewModel lora })
        {
            _selectedLoras.Remove(lora);
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

    /// <summary>
    /// インポート済み(ローカルファイルパス)かつ非アクティブな項目にだけ削除ボタンを出す。
    /// Hugging Face キャッシュのモデルはリポジトリ ID なので対象外。アクティブモデルは
    /// 切り替えてからでないと削除できない。
    /// </summary>
    public Visibility RemoveVisibility =>
        !IsActive && Path.Exists(ModelId) ? Visibility.Visible : Visibility.Collapsed;
}
