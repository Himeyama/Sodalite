using System.Collections.ObjectModel;
using System.Collections.Specialized;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.Windows.ApplicationModel.Resources;
using Sodalite.Models;
using Sodalite.Services;
using Sodalite.ViewModels;
using Windows.Storage;
using Windows.Storage.Pickers;
using WinRT.Interop;

namespace Sodalite.Views;

sealed partial class ModelSelectionPage : Page
{
    static readonly ResourceLoader ResourceLoader = new();

    BackendApiClient _apiClient = null!;
    Window _ownerWindow = null!;
    ObservableCollection<SelectedLoraViewModel> _selectedLoras = null!;

    List<LoraFileInfo> _availableLoras = [];

    /// <summary>モデルの切り替えが成功したときに発火する。オーナー側でデバイス情報を更新する。</summary>
    public event EventHandler? ModelSwitched;

    public event EventHandler<string>? ModelSwitchStatusChanged;

    public event EventHandler? ModelSwitchFinished;

    /// <summary>戻る操作が要求されたときに発火する。オーナー側で前の画面へ復帰させる。</summary>
    public event EventHandler? BackRequested;

    public ModelSelectionPage() => InitializeComponent();

    /// <summary>
    /// 依存を注入して初期ロードを開始する。<see cref="MainWindow"/> の画面切り替え直後に呼ぶ。
    /// </summary>
    /// <param name="apiClient">バックエンド API クライアント。</param>
    /// <param name="ownerWindow">フォルダピッカー初期化用のオーナーウィンドウ。</param>
    /// <param name="selectedLoras">生成ページと共有する選択済み LoRA コレクション。</param>
    public async void Initialize(
        BackendApiClient apiClient,
        Window ownerWindow,
        ObservableCollection<SelectedLoraViewModel> selectedLoras)
    {
        _apiClient = apiClient;
        _ownerWindow = ownerWindow;
        _selectedLoras = selectedLoras;

        SelectedLoraItemsControl.ItemsSource = _selectedLoras;
        _selectedLoras.CollectionChanged += SelectedLoras_CollectionChanged;
        UpdateLoraEmptyVisibility();

        Unloaded += ModelSelectionPage_Unloaded;

        await RefreshDirectoriesAsync();
        await LoadModelsAsync();
        await RefreshAvailableLorasAsync();
    }

    void ModelSelectionPage_Unloaded(object sender, RoutedEventArgs e) =>
        _selectedLoras.CollectionChanged -= SelectedLoras_CollectionChanged;

    void BackButton_Click(object sender, RoutedEventArgs e) =>
        BackRequested?.Invoke(this, EventArgs.Empty);

    async Task RefreshDirectoriesAsync()
    {
        try
        {
            ScanDirectories directories = await _apiClient.GetScanDirectoriesAsync(CancellationToken.None).ConfigureAwait(true);
            ApplyModelDirText(directories.ModelDir);
            ApplyLoraDirText(directories.LoraDir);
        }
        catch (Exception ex)
        {
            ShowError(ResourceLoader.GetString("ModelSelectionDialog_LoadErrorTitle"), ex.Message);
        }
    }

    void ApplyModelDirText(string? modelDir)
    {
        if (string.IsNullOrEmpty(modelDir))
        {
            ModelDirTextBlock.Text = ResourceLoader.GetString("ModelSelectionPage_ModelDirUnset/Text");
        }
        else
        {
            ModelDirTextBlock.Text = modelDir;
        }
    }

    void ApplyLoraDirText(string? loraDir)
    {
        if (string.IsNullOrEmpty(loraDir))
        {
            LoraDirTextBlock.Text = ResourceLoader.GetString("ModelSelectionPage_LoraDirUnset/Text");
        }
        else
        {
            LoraDirTextBlock.Text = loraDir;
        }
    }

    async void ChooseModelDirButton_Click(object sender, RoutedEventArgs e)
    {
        string? folder = await PickFolderAsync();
        if (folder is null)
        {
            return;
        }

        ErrorInfoBar.IsOpen = false;
        LoadingProgressRing.IsActive = true;
        ModelListView.IsEnabled = false;

        try
        {
            ScanDirectories saved = await SaveDirectoriesAsync(modelDir: folder, keepLoraDir: true).ConfigureAwait(true);
            ApplyModelDirText(saved.ModelDir);
            await LoadModelsAsync();
        }
        catch (Exception ex)
        {
            ShowError(ResourceLoader.GetString("ModelSelectionPage_DirectoryErrorTitle"), ex.Message);
        }
        finally
        {
            LoadingProgressRing.IsActive = false;
            ModelListView.IsEnabled = true;
        }
    }

    async void ChooseLoraDirButton_Click(object sender, RoutedEventArgs e)
    {
        string? folder = await PickFolderAsync();
        if (folder is null)
        {
            return;
        }

        ErrorInfoBar.IsOpen = false;

        try
        {
            ScanDirectories saved = await SaveDirectoriesAsync(loraDir: folder, keepModelDir: true).ConfigureAwait(true);
            ApplyLoraDirText(saved.LoraDir);
            await RefreshAvailableLorasAsync();
        }
        catch (Exception ex)
        {
            ShowError(ResourceLoader.GetString("ModelSelectionPage_DirectoryErrorTitle"), ex.Message);
        }
    }

    /// <summary>
    /// 片方のディレクトリだけ更新し、もう片方は現在値を維持して PUT する。バックエンドの
    /// 設定は両ディレクトリを一括で受け取るため、更新しない側の現在値を読み直して渡す。
    /// </summary>
    async Task<ScanDirectories> SaveDirectoriesAsync(
        string? modelDir = null,
        string? loraDir = null,
        bool keepModelDir = false,
        bool keepLoraDir = false)
    {
        ScanDirectories current = await _apiClient.GetScanDirectoriesAsync(CancellationToken.None).ConfigureAwait(true);
        ScanDirectories next = new(
            keepModelDir ? current.ModelDir : modelDir,
            keepLoraDir ? current.LoraDir : loraDir);
        return await _apiClient.SetScanDirectoriesAsync(next, CancellationToken.None).ConfigureAwait(true);
    }

    async Task<string?> PickFolderAsync()
    {
        FolderPicker picker = new();
        InitializeWithWindow.Initialize(picker, WindowNative.GetWindowHandle(_ownerWindow));
        picker.FileTypeFilter.Add("*");

        StorageFolder? folder = await picker.PickSingleFolderAsync();
        return folder?.Path;
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
    /// LoRA ディレクトリのスキャン結果で Flyout を組み直す。空の Flyout はクリックしても開けない
    /// ため、項目が無いときは追加ボタンごと無効化する。
    /// </summary>
    void BuildLoraFlyoutItems()
    {
        AddLoraMenuFlyout.Items.Clear();

        foreach (LoraFileInfo lora in _availableLoras)
        {
            string loraId = lora.LoraId;
            MenuFlyoutItem item = new() { Text = DisplayNameFor(loraId) };
            item.Click += (_, _) => AddSelectedLora(loraId);
            AddLoraMenuFlyout.Items.Add(item);
        }

        AddLoraButton.IsEnabled = _availableLoras.Count > 0;
    }

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
        // ここでは選択＝モデル切り替え。すでにアクティブな項目や、リスト再構築で選択が
        // 外れたケースは無視する。
        if (ModelListView.SelectedItem is not ModelListItem item || item.IsActive)
        {
            return;
        }

        ErrorInfoBar.IsOpen = false;
        ModelListView.IsEnabled = false;
        LoadingProgressRing.IsActive = true;
        ModelLoadStatusTextBlock.Text = "モデルを読み込み中";
        ModelLoadStatusTextBlock.Visibility = Visibility.Visible;
        ModelSwitchStatusChanged?.Invoke(this, $"モデルを切り替え中: {item.DisplayName}");
        using CancellationTokenSource progressCancellation = new();
        Task progressTask = PollModelLoadStageAsync(progressCancellation.Token);

        try
        {
            ModelInfo result = await _apiClient.SetActiveModelAsync(item.ModelId, CancellationToken.None).ConfigureAwait(true);
            AppSettings.LastModelId = result.ModelId;

            // モデルを切り替えても選択中の LoRA と重みはそのまま維持する。SD1.5/SDXL の
            // 互換性が合わない LoRA は生成時にバックエンド側でスキップされる。
            ModelSwitched?.Invoke(this, EventArgs.Empty);
            ModelSwitchStatusChanged?.Invoke(this, $"モデルを切り替えました: {item.DisplayName}");
            BackRequested?.Invoke(this, EventArgs.Empty);
        }
        catch (Exception ex)
        {
            ModelListView.SelectedItem = null;
            ShowError(ResourceLoader.GetString("ModelSelectionDialog_SwitchErrorTitle"), ex.Message);
            ModelSwitchStatusChanged?.Invoke(this, $"モデル切り替えに失敗: {ex.Message}");
        }
        finally
        {
            await progressCancellation.CancelAsync();
            await progressTask;
            LoadingProgressRing.IsActive = false;
            ModelLoadStatusTextBlock.Visibility = Visibility.Collapsed;
            ModelListView.IsEnabled = true;
            ModelSwitchFinished?.Invoke(this, EventArgs.Empty);
        }
    }

    async Task PollModelLoadStageAsync(CancellationToken ct)
    {
        try
        {
            while (!ct.IsCancellationRequested)
            {
                await Task.Delay(TimeSpan.FromSeconds(2), ct);
                try
                {
                    HealthInfo health = await _apiClient.GetHealthAsync(ct);
                    if (health.ModelLoadingStage is string stage)
                    {
                        string status = $"モデルを切り替え中: {stage}";
                        if (health.ModelDownloadSource is string source &&
                            health.ModelDownloadDestination is string destination)
                        {
                            status += $"\nDL 元: {source}\n保存先: {destination}";
                        }
                        ModelLoadStatusTextBlock.Text = status;
                        ModelSwitchStatusChanged?.Invoke(this, status);
                    }
                }
                catch (HttpRequestException)
                {
                    // A temporary health-check failure does not stop the switch.
                }
            }
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
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
}
