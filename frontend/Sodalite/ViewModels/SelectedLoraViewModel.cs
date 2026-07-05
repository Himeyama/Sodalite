using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace Sodalite.ViewModels;

/// <summary>生成に適用する 1 つの LoRA。パスは固定、重みだけ UI から変更できる。</summary>
sealed class SelectedLoraViewModel(string loraId) : INotifyPropertyChanged
{
    double _weight = 1.0;

    public event PropertyChangedEventHandler? PropertyChanged;

    public string LoraId { get; } = loraId;

    public string DisplayName =>
        Path.Exists(LoraId) ? Path.GetFileNameWithoutExtension(LoraId) : LoraId;

    public double Weight
    {
        get => _weight;
        set
        {
            if (SetField(ref _weight, value))
            {
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(WeightText)));
            }
        }
    }

    public string WeightText => Weight.ToString("F2");

    bool SetField<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return false;
        }

        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        return true;
    }
}
