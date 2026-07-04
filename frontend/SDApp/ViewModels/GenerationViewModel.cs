using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices.WindowsRuntime;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml.Media.Imaging;
using SDApp.Models;
using SDApp.Services;
using Windows.Storage.Streams;

namespace SDApp.ViewModels;

sealed class GenerationViewModel : INotifyPropertyChanged
{
    readonly BackendApiClient _apiClient;
    readonly DispatcherQueue _dispatcherQueue;

    string _prompt = "";
    string _negativePrompt = "";
    int _steps = 20;
    double _cfgScale = 7.0;
    int _width = 512;
    int _height = 512;
    string _sampler = "euler_a";
    string _seedText = "";
    string _statusText = "Ready";
    bool _isGenerating;
    BitmapImage? _resultImage;
    List<string> _samplers = [];
    string _deviceInfo = "";

    public GenerationViewModel(BackendApiClient apiClient, DispatcherQueue dispatcherQueue)
    {
        _apiClient = apiClient;
        _dispatcherQueue = dispatcherQueue;
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    public string Prompt
    {
        get => _prompt;
        set => SetField(ref _prompt, value);
    }

    public string NegativePrompt
    {
        get => _negativePrompt;
        set => SetField(ref _negativePrompt, value);
    }

    public int Steps
    {
        get => _steps;
        set => SetField(ref _steps, value);
    }

    public double CfgScale
    {
        get => _cfgScale;
        set => SetField(ref _cfgScale, value);
    }

    public int Width
    {
        get => _width;
        set => SetField(ref _width, value);
    }

    public int Height
    {
        get => _height;
        set => SetField(ref _height, value);
    }

    public string Sampler
    {
        get => _sampler;
        set => SetField(ref _sampler, value);
    }

    /// <summary>空欄ならランダムシード(null送信)として扱う。</summary>
    public string SeedText
    {
        get => _seedText;
        set => SetField(ref _seedText, value);
    }

    public string StatusText
    {
        get => _statusText;
        set => SetField(ref _statusText, value);
    }

    public bool IsGenerating
    {
        get => _isGenerating;
        set => SetField(ref _isGenerating, value);
    }

    public BitmapImage? ResultImage
    {
        get => _resultImage;
        set => SetField(ref _resultImage, value);
    }

    public List<string> Samplers
    {
        get => _samplers;
        set => SetField(ref _samplers, value);
    }

    public string DeviceInfo
    {
        get => _deviceInfo;
        set => SetField(ref _deviceInfo, value);
    }

    public async Task InitializeAsync(CancellationToken ct)
    {
        try
        {
            List<string> samplers = await _apiClient.GetSamplersAsync(ct).ConfigureAwait(false);
            HealthInfo health = await _apiClient.GetHealthAsync(ct).ConfigureAwait(false);

            _dispatcherQueue.TryEnqueue(() =>
            {
                Samplers = samplers;
                if (samplers.Count > 0 && !samplers.Contains(Sampler))
                {
                    Sampler = samplers[0];
                }

                DeviceInfo = $"{health.Device} / {health.LoadedModel}";
            });
        }
        catch (Exception ex)
        {
            _dispatcherQueue.TryEnqueue(() => StatusText = $"Error: {ex.Message}");
        }
    }

    public async Task GenerateAsync(CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(Prompt) || IsGenerating)
        {
            return;
        }

        if (!TryParseSeed(SeedText, out long? seed))
        {
            StatusText = "Seed must be an integer.";
            return;
        }

        IsGenerating = true;
        StatusText = "Generating...";

        try
        {
            GenerationRequest request = new(
                Prompt,
                NegativePrompt,
                Steps,
                CfgScale,
                Width,
                Height,
                Sampler,
                seed);

            GenerationResult result = await _apiClient.GenerateTextToImageAsync(request, ct).ConfigureAwait(false);

            if (result.Error is not null)
            {
                _dispatcherQueue.TryEnqueue(() => StatusText = $"Error: {result.Error}");
                return;
            }

            if (result.ImageUrl is not string imageUrl)
            {
                _dispatcherQueue.TryEnqueue(() => StatusText = "No image returned.");
                return;
            }

            byte[] imageBytes = await _apiClient.DownloadImageAsync(imageUrl, ct).ConfigureAwait(false);

            _dispatcherQueue.TryEnqueue(async void () =>
            {
                InMemoryRandomAccessStream stream = new();
                await stream.WriteAsync(imageBytes.AsBuffer());
                stream.Seek(0);

                BitmapImage bitmap = new();
                await bitmap.SetSourceAsync(stream);
                ResultImage = bitmap;
                StatusText = "Done";
            });
        }
        catch (Exception ex)
        {
            _dispatcherQueue.TryEnqueue(() => StatusText = $"Error: {ex.Message}");
        }
        finally
        {
            _dispatcherQueue.TryEnqueue(() => IsGenerating = false);
        }
    }

    static bool TryParseSeed(string seedText, out long? seed)
    {
        if (string.IsNullOrWhiteSpace(seedText))
        {
            seed = null;
            return true;
        }

        if (long.TryParse(seedText, out long parsed))
        {
            seed = parsed;
            return true;
        }

        seed = null;
        return false;
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
