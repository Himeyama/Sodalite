using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace Sodalite.Views;

sealed class GalleryItemTemplateSelector : DataTemplateSelector
{
    public DataTemplate ImageTemplate { get; set; } = null!;

    public DataTemplate SkeletonTemplate { get; set; } = null!;

    protected override DataTemplate SelectTemplateCore(object item) =>
        item is GallerySkeletonItem ? SkeletonTemplate : ImageTemplate;

    protected override DataTemplate SelectTemplateCore(object item, DependencyObject container) =>
        SelectTemplateCore(item);
}

/// <summary>ギャラリー読み込み中に表示するプレースホルダー用のマーカー型。</summary>
sealed class GallerySkeletonItem;
