using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using Sodalite.Models;
using Sodalite.Services;

namespace Sodalite.ViewModels;

/// <summary>
/// 呼び出し元 (<see cref="Sodalite.Views.GalleryPage"/>) が常に UI スレッドから呼ぶ前提の
/// ViewModel。<see cref="Images"/> への変更は各メソッドが返る前に同期的に完了する。
/// </summary>
sealed class GalleryViewModel : INotifyPropertyChanged
{
    BackendApiClient? _apiClient;
    bool _isLoading;
    string? _errorMessage;

    public ObservableCollection<GalleryImageInfo> Images { get; } = [];

    public event PropertyChangedEventHandler? PropertyChanged;

    public bool IsLoading
    {
        get => _isLoading;
        set => SetField(ref _isLoading, value);
    }

    public string? ErrorMessage
    {
        get => _errorMessage;
        set => SetField(ref _errorMessage, value);
    }

    public async Task LoadAsync(BackendApiClient apiClient, CancellationToken ct)
    {
        _apiClient = apiClient;
        IsLoading = true;
        ErrorMessage = null;

        try
        {
            List<GalleryImageInfo> images = await apiClient.GetGalleryImagesAsync(ct);
            Images.Clear();
            foreach (GalleryImageInfo image in images)
            {
                Images.Add(image);
            }
        }
        catch (Exception ex)
        {
            ErrorMessage = ex.Message;
        }
        finally
        {
            IsLoading = false;
        }
    }

    /// <summary>
    /// サーバーから最新一覧 (新しい順) を取得し、既存の <see cref="Images"/> との差分だけを
    /// 追加・削除して並び順をサーバーの結果に合わせる。全消去して作り直すことはしない。
    /// 呼び出し元の UI スレッド上で呼ばれる前提で、<see cref="Images"/> への変更は
    /// このメソッドが返る前に同期的に完了する。
    /// </summary>
    public async Task<GalleryDiff?> RefreshAsync(BackendApiClient apiClient, CancellationToken ct)
    {
        _apiClient = apiClient;

        try
        {
            List<GalleryImageInfo> latest = await apiClient.GetGalleryImagesAsync(ct);
            HashSet<string> latestIds = latest.Select(image => image.ImageId).ToHashSet();
            HashSet<string> currentIds = Images.Select(image => image.ImageId).ToHashSet();

            List<GalleryImageInfo> added = latest.Where(image => !currentIds.Contains(image.ImageId)).ToList();
            List<GalleryImageInfo> removed = Images.Where(image => !latestIds.Contains(image.ImageId)).ToList();

            foreach (GalleryImageInfo image in removed)
            {
                Images.Remove(image);
            }

            for (int i = 0; i < latest.Count; i++)
            {
                GalleryImageInfo image = latest[i];
                int currentIndex = Images.IndexOf(image);
                if (currentIndex < 0)
                {
                    Images.Insert(Math.Min(i, Images.Count), image);
                }
                else if (currentIndex != i)
                {
                    Images.Move(currentIndex, Math.Min(i, Images.Count - 1));
                }
            }

            return new GalleryDiff(added, removed);
        }
        catch (Exception ex)
        {
            ErrorMessage = ex.Message;
            return null;
        }
    }

    /// <summary>
    /// 削除に成功したら null、失敗したらエラーメッセージを返す。呼び出し元の UI スレッド上で
    /// 呼ばれる前提で、<see cref="Images"/> への変更はこのメソッドが返る前に同期的に完了する。
    /// </summary>
    public async Task<string?> DeleteAsync(GalleryImageInfo image, CancellationToken ct)
    {
        if (_apiClient is not BackendApiClient apiClient)
        {
            return null;
        }

        int originalIndex = Images.IndexOf(image);
        if (originalIndex < 0)
        {
            return null;
        }

        Images.RemoveAt(originalIndex);

        try
        {
            await apiClient.DeleteGalleryImageAsync(image.ImageId, ct);
            return null;
        }
        catch (Exception ex)
        {
            int insertIndex = Math.Min(originalIndex, Images.Count);
            Images.Insert(insertIndex, image);
            return ex.Message;
        }
    }

    void SetField<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return;
        }

        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}

sealed record GalleryDiff(IReadOnlyList<GalleryImageInfo> Added, IReadOnlyList<GalleryImageInfo> Removed);
