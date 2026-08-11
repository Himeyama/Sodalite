using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media.Animation;
using Microsoft.UI.Xaml.Media.Imaging;
using Microsoft.Windows.ApplicationModel.Resources;
using Sodalite.Models;
using Sodalite.Services;
using Sodalite.ViewModels;

namespace Sodalite.Views;

sealed partial class GalleryPage : Page
{
    const int SkeletonItemCount = 12;

    static readonly ResourceLoader ResourceLoader = new();

    readonly GalleryViewModel _viewModel;

    BackendApiClient _apiClient = null!;
    bool _hasLoadedOnce;

    /// <summary>戻る操作が要求されたときに発火する。オーナー側で前の画面へ復帰させる。</summary>
    public event EventHandler? BackRequested;

    /// <summary>選択した履歴のパラメータを生成画面に反映するよう要求する。</summary>
    public event EventHandler<GalleryImageInfo>? ReuseParametersRequested;

    public GalleryPage()
    {
        InitializeComponent();
        _viewModel = new GalleryViewModel(DispatcherQueue);
    }

    public async void Initialize(BackendApiClient apiClient)
    {
        _apiClient = apiClient;

        if (_hasLoadedOnce)
        {
            await RefreshDiffAsync();
        }
        else
        {
            await ReloadAsync();
        }
    }

    async Task ReloadAsync()
    {
        ErrorInfoBar.IsOpen = false;
        EmptyTextBlock.Visibility = Visibility.Collapsed;
        ImagesGridView.ItemsSource = Enumerable.Range(0, SkeletonItemCount)
            .Select(_ => new GallerySkeletonItem())
            .ToList();

        await _viewModel.LoadAsync(_apiClient, CancellationToken.None);

        if (_viewModel.ErrorMessage is string error)
        {
            ErrorInfoBar.Title = ResourceLoader.GetString("GalleryPage_LoadErrorTitle");
            ErrorInfoBar.Message = error;
            ErrorInfoBar.IsOpen = true;
        }

        ImagesGridView.ItemsSource = _viewModel.Images
            .Select(BuildItem)
            .ToList();
        EmptyTextBlock.Visibility = _viewModel.Images.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
        _hasLoadedOnce = true;
    }

    /// <summary>
    /// 既に一覧を表示済みの状態で再度開かれたときに使う。全件を取得し直すが、
    /// サムネイル画像は追加・削除された分だけを組み立てて既存の表示を維持する。
    /// </summary>
    async Task RefreshDiffAsync()
    {
        GalleryDiff? diff = await _viewModel.RefreshAsync(_apiClient, CancellationToken.None);

        if (diff is null)
        {
            if (_viewModel.ErrorMessage is string error)
            {
                ErrorInfoBar.Title = ResourceLoader.GetString("GalleryPage_LoadErrorTitle");
                ErrorInfoBar.Message = error;
                ErrorInfoBar.IsOpen = true;
            }

            return;
        }

        if (diff.Added.Count == 0 && diff.Removed.Count == 0)
        {
            return;
        }

        List<GalleryImageItem> items = ((List<GalleryImageItem>)ImagesGridView.ItemsSource)
            .Where(item => !diff.Removed.Contains(item.Image))
            .ToList();

        foreach (GalleryImageInfo added in diff.Added)
        {
            int insertIndex = _viewModel.Images.IndexOf(added);
            items.Insert(Math.Clamp(insertIndex, 0, items.Count), BuildItem(added));
        }

        ImagesGridView.ItemsSource = null;
        ImagesGridView.ItemsSource = items;
        EmptyTextBlock.Visibility = items.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
    }

    GalleryImageItem BuildItem(GalleryImageInfo image)
    {
        Uri thumbnailUri = new(_apiClient.BaseAddress, image.ImageUrl);
        return new GalleryImageItem(image, thumbnailUri, new BitmapImage(thumbnailUri));
    }

    void BackButton_Click(object sender, RoutedEventArgs e) => BackRequested?.Invoke(this, EventArgs.Empty);

    void SkeletonItemBorder_Loaded(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { Resources: var resources } &&
            resources.TryGetValue("ShimmerStoryboard", out object storyboardObject) &&
            storyboardObject is Storyboard storyboard)
        {
            storyboard.Begin();
        }
    }

    async void ImagesGridView_ItemClick(object sender, ItemClickEventArgs e)
    {
        if (e.ClickedItem is not GalleryImageItem item)
        {
            return;
        }

        GalleryDetailDialog dialog = new(item.Image, item.Thumbnail) { XamlRoot = Content.XamlRoot };

        dialog.ReuseRequested += (_, _) =>
        {
            dialog.Hide();
            ReuseParametersRequested?.Invoke(this, item.Image);
        };
        dialog.DeleteConfirmed += async (_, _) =>
        {
            dialog.Hide();
            await DeleteItemAsync(item);
        };

        await dialog.ShowAsync();
    }

    async Task DeleteItemAsync(GalleryImageItem item)
    {
        List<GalleryImageItem> items = (List<GalleryImageItem>)ImagesGridView.ItemsSource;
        int index = items.IndexOf(item);
        if (index < 0)
        {
            return;
        }

        items.RemoveAt(index);
        ImagesGridView.ItemsSource = null;
        ImagesGridView.ItemsSource = items;
        EmptyTextBlock.Visibility = items.Count == 0 ? Visibility.Visible : Visibility.Collapsed;

        string? error = await _viewModel.DeleteAsync(item.Image, CancellationToken.None);

        if (error is not null)
        {
            items.Insert(Math.Min(index, items.Count), item);
            ImagesGridView.ItemsSource = null;
            ImagesGridView.ItemsSource = items;
            EmptyTextBlock.Visibility = Visibility.Collapsed;

            ErrorInfoBar.Title = ResourceLoader.GetString("GalleryPage_LoadErrorTitle");
            ErrorInfoBar.Message = error;
            ErrorInfoBar.IsOpen = true;
        }
    }
}

sealed record GalleryImageItem(GalleryImageInfo Image, Uri ThumbnailUri, BitmapImage Thumbnail);
