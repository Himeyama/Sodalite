using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices.WindowsRuntime;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml.Media.Imaging;
using Microsoft.Windows.ApplicationModel.Resources;
using Sodalite.Models;
using Sodalite.Services;
using Windows.Storage.Streams;

namespace Sodalite.ViewModels;

sealed class GenerationViewModel : INotifyPropertyChanged
{
    static readonly TimeSpan PollInterval = TimeSpan.FromMilliseconds(400);

    static readonly ResourceLoader ResourceLoader = new();

    readonly DispatcherQueue _dispatcherQueue;
    readonly DispatcherQueueTimer _generatingElapsedTimer;

    BackendApiClient? _apiClient;
    Stopwatch _generatingStopwatch = new();
    Stopwatch _batchStopwatch = new();
    string? _runningJobId;
    int _batchImagesCompleted;
    int _batchTotalImages;
    string _prompt = "";
    string _negativePrompt = "";
    int _steps = 20;
    double _cfgScale = 7.0;
    int _width = 1024;
    int _height = 1024;
    int _batchSize = 1;
    string _sampler = "euler_a";
    string _seedText = "";
    string? _initialImage;
    double _strength = 0.4;
    string _statusText = ResourceLoader.GetString("Generation_BackendStarting");
    bool _isGenerating;
    bool _isBackendReady;
    BitmapImage? _resultImage;
    byte[]? _resultImageBytes;
    string? _resultImagePath;
    List<string> _samplers = [];
    string _deviceInfo = "";

    public ObservableCollection<SelectedLoraViewModel> SelectedLoras { get; } = [];

    public GenerationViewModel(DispatcherQueue dispatcherQueue)
    {
        _dispatcherQueue = dispatcherQueue;

        _generatingElapsedTimer = dispatcherQueue.CreateTimer();
        _generatingElapsedTimer.Interval = TimeSpan.FromMilliseconds(100);
        _generatingElapsedTimer.Tick += (_, _) => UpdateGeneratingStatusText();
    }

    /// <summary>生成中のステータス文言を組み立てる。バッチ枚数が1のときは単純な経過秒数のみ、
    /// 2枚以上のときは現在の番号・現在の画像の経過秒・バッチ全体の累計秒・平均秒を表示する。</summary>
    void UpdateGeneratingStatusText()
    {
        if (_batchTotalImages <= 1)
        {
            StatusText = string.Format(ResourceLoader.GetString("Generation_Generating"), _generatingStopwatch.Elapsed.TotalSeconds);
            return;
        }

        int currentImageNumber = Math.Min(_batchImagesCompleted + 1, _batchTotalImages);
        double currentImageSeconds = _generatingStopwatch.Elapsed.TotalSeconds;
        double totalSeconds = _batchStopwatch.Elapsed.TotalSeconds;
        double averageSeconds = _batchImagesCompleted > 0 ? totalSeconds / _batchImagesCompleted : currentImageSeconds;

        StatusText = string.Format(
            ResourceLoader.GetString("Generation_GeneratingBatch"),
            currentImageNumber,
            _batchTotalImages,
            currentImageSeconds,
            totalSeconds,
            averageSeconds);
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

    public int BatchSize
    {
        get => _batchSize;
        set => SetField(ref _batchSize, value);
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

    /// <summary>選択した元画像の base64。null のときは通常の text-to-image として生成する。</summary>
    public string? InitialImage
    {
        get => _initialImage;
        set => SetField(ref _initialImage, value);
    }

    public double Strength
    {
        get => _strength;
        set => SetField(ref _strength, value);
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

    public bool IsBackendReady
    {
        get => _isBackendReady;
        set => SetField(ref _isBackendReady, value);
    }

    public BitmapImage? ResultImage
    {
        get => _resultImage;
        set => SetField(ref _resultImage, value);
    }

    /// <summary>表示中の画像の元 PNG バイト列。コピー・保存に使う。未生成時は null。</summary>
    public byte[]? ResultImageBytes
    {
        get => _resultImageBytes;
        set => SetField(ref _resultImageBytes, value);
    }

    /// <summary>バックエンドが保存した画像の絶対パス。保存先フォルダを開く操作に使う。未生成時は null。</summary>
    public string? ResultImagePath
    {
        get => _resultImagePath;
        set => SetField(ref _resultImagePath, value);
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

    /// <summary>
    /// サーバー自体は起動済みだが、初回モデルはバックグラウンドでまだ VRAM に展開中のことがある。
    /// 生成には展開済みモデルが要るため、`model_ready` になるまでポーリングしてから使用可能にする。
    /// </summary>
    public async Task AttachBackendAsync(BackendApiClient apiClient, CancellationToken ct)
    {
        _apiClient = apiClient;

        try
        {
            List<string> samplers = await apiClient.GetSamplersAsync(ct).ConfigureAwait(false);
            _dispatcherQueue.TryEnqueue(() =>
            {
                Samplers = samplers;
                if (samplers.Count > 0 && !samplers.Contains(Sampler))
                {
                    Sampler = samplers[0];
                }
            });

            HealthInfo health = await apiClient.GetHealthAsync(ct).ConfigureAwait(false);
            while (!health.ModelReady)
            {
                await Task.Delay(PollInterval, ct).ConfigureAwait(false);
                health = await apiClient.GetHealthAsync(ct).ConfigureAwait(false);
            }

            _dispatcherQueue.TryEnqueue(() =>
            {
                DeviceInfo = $"{health.Device} / {DisplayNameFor(health.LoadedModel)}";
                IsBackendReady = true;
                StatusText = ResourceLoader.GetString("Generation_Ready");
            });
        }
        catch (Exception ex)
        {
            _dispatcherQueue.TryEnqueue(() => StatusText = string.Format(ResourceLoader.GetString("Generation_Error"), ex.Message));
        }
    }

    public async Task RefreshDeviceInfoAsync(BackendApiClient apiClient, CancellationToken ct)
    {
        HealthInfo health = await apiClient.GetHealthAsync(ct).ConfigureAwait(false);
        _dispatcherQueue.TryEnqueue(() => DeviceInfo = $"{health.Device} / {DisplayNameFor(health.LoadedModel)}");
    }

    static string DisplayNameFor(string? modelId) =>
        modelId is not null && Path.Exists(modelId) ? Path.GetFileNameWithoutExtension(modelId) : modelId ?? "";

    /// <summary>画像が1枚完成するたびに発火する。ギャラリー等、他画面への反映に使う。</summary>
    public event EventHandler? ImageCompleted;

    public async Task GenerateAsync(CancellationToken ct)
    {
        if (_apiClient is not BackendApiClient apiClient || string.IsNullOrWhiteSpace(Prompt) || IsGenerating)
        {
            return;
        }

        if (!TryParseSeed(SeedText, out long? seed))
        {
            StatusText = ResourceLoader.GetString("Generation_SeedMustBeInteger");
            return;
        }

        IsGenerating = true;
        _batchImagesCompleted = 0;
        _batchTotalImages = BatchSize;
        StatusText = string.Format(ResourceLoader.GetString("Generation_Generating"), 0.0);

        try
        {
            List<LoraSelection> loras = SelectedLoras
                .Select(lora => new LoraSelection(lora.LoraId, lora.Weight))
                .ToList();

            GenerationRequest request = new(
                Prompt,
                NegativePrompt,
                Steps,
                CfgScale,
                Width,
                Height,
                BatchSize,
                Sampler,
                seed,
                loras,
                InitialImage,
                Strength);

            GenerationResult started = InitialImage is null
                ? await apiClient.StartTextToImageAsync(request, ct).ConfigureAwait(false)
                : await apiClient.StartImageToImageAsync(request, ct).ConfigureAwait(false);
            _runningJobId = started.JobId;

            await PollUntilDoneAsync(apiClient, started.JobId, ct).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _dispatcherQueue.TryEnqueue(() => StatusText = string.Format(ResourceLoader.GetString("Generation_Error"), ex.Message));
        }
        finally
        {
            _runningJobId = null;

            // ConfigureAwait(false) 後はスレッドプール上で実行され得るため、経過時間タイマーを
            // 所有する UI スレッド上でまとめて停止・状態解除する。
            _dispatcherQueue.TryEnqueue(() =>
            {
                _generatingElapsedTimer.Stop();
                _generatingStopwatch.Stop();
                _batchStopwatch.Stop();
                IsGenerating = false;
            });
        }
    }

    /// <summary>実行中の生成ジョブをキャンセルする。次に生成予定だった画像の開始前に反映される。</summary>
    public async Task CancelAsync(CancellationToken ct)
    {
        if (_apiClient is not BackendApiClient apiClient || _runningJobId is not string jobId)
        {
            return;
        }

        StatusText = ResourceLoader.GetString("Generation_Cancelling");
        await apiClient.CancelGenerationJobAsync(jobId, ct).ConfigureAwait(false);
    }

    /// <summary>ジョブが完了するまでポーリングし、完成画像が増えるたびに 1 枚ずつ表示・秒数リセットする。</summary>
    async Task PollUntilDoneAsync(BackendApiClient apiClient, string jobId, CancellationToken ct)
    {
        int lastImagesCompleted = 0;
        _generatingStopwatch = Stopwatch.StartNew();
        _batchStopwatch = Stopwatch.StartNew();
        _dispatcherQueue.TryEnqueue(() => _generatingElapsedTimer.Start());

        while (true)
        {
            GenerationResult result = await apiClient.GetGenerationJobAsync(jobId, ct).ConfigureAwait(false);

            if (result.ImagesCompleted > lastImagesCompleted && result.ImageUrl is string imageUrl)
            {
                lastImagesCompleted = result.ImagesCompleted;
                _batchImagesCompleted = result.ImagesCompleted;
                await ShowCompletedImageAsync(apiClient, imageUrl, result.ImagePath, ct).ConfigureAwait(false);

                // 次の1枚の経過時間として数え直す(バッチ全体の累計はそのまま継続)。
                _generatingStopwatch = Stopwatch.StartNew();
            }

            switch (result.Status)
            {
                case "completed":
                    if (_batchTotalImages <= 1)
                    {
                        // 画像完成のたびにリセットされる _generatingStopwatch ではなく、
                        // バッチ開始からの累計を持つ _batchStopwatch を使う。1枚しかない
                        // バッチでは画像完成直後にリセットが起き、_generatingStopwatch は
                        // ほぼ 0 秒になってしまうため。
                        double doneSeconds = _batchStopwatch.Elapsed.TotalSeconds;
                        _dispatcherQueue.TryEnqueue(() => StatusText = string.Format(ResourceLoader.GetString("Generation_Done"), doneSeconds));
                    }

                    return;
                case "cancelled":
                    _dispatcherQueue.TryEnqueue(() => StatusText = ResourceLoader.GetString("Generation_Cancelled"));
                    return;
                case "failed":
                    _dispatcherQueue.TryEnqueue(() => StatusText = string.Format(ResourceLoader.GetString("Generation_Error"), result.Error));
                    return;
            }

            await Task.Delay(PollInterval, ct).ConfigureAwait(false);
        }
    }

    async Task ShowCompletedImageAsync(BackendApiClient apiClient, string imageUrl, string? imagePath, CancellationToken ct)
    {
        byte[] imageBytes = await apiClient.DownloadImageAsync(imageUrl, ct).ConfigureAwait(false);

        TaskCompletionSource displayed = new();
        _dispatcherQueue.TryEnqueue(async void () =>
        {
            try
            {
                InMemoryRandomAccessStream stream = new();
                await stream.WriteAsync(imageBytes.AsBuffer());
                stream.Seek(0);

                BitmapImage bitmap = new();
                await bitmap.SetSourceAsync(stream);
                ResultImageBytes = imageBytes;
                ResultImagePath = imagePath;
                ResultImage = bitmap;
                ImageCompleted?.Invoke(this, EventArgs.Empty);
            }
            finally
            {
                displayed.TrySetResult();
            }
        });

        await displayed.Task.ConfigureAwait(false);
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
