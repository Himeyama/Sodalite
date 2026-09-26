using System.Net.Http.Json;
using System.Text.Json.Serialization;
using Sodalite.Models;

namespace Sodalite.Services;

sealed class BackendApiClient(int port) : IDisposable
{
    readonly HttpClient _http = new()
    {
        BaseAddress = new Uri($"http://127.0.0.1:{port}"),
        Timeout = TimeSpan.FromMinutes(10),
    };
    readonly HttpClient _modelSwitchHttp = new()
    {
        BaseAddress = new Uri($"http://127.0.0.1:{port}"),
        Timeout = TimeSpan.FromHours(2),
    };

    /// <summary>サムネイル等、相対URLを完全URLに組み立てる呼び出し元向けに公開する。</summary>
    public Uri BaseAddress => _http.BaseAddress!;

    public async Task<GenerationResult> StartTextToImageAsync(GenerationRequest request, CancellationToken ct)
    {
        List<LoraBody> loras = request.Loras?
            .Select(lora => new LoraBody(lora.ModelId, lora.Weight))
            .ToList() ?? [];

        TextToImageBody body = new(
            request.Prompt,
            request.NegativePrompt,
            request.Steps,
            request.CfgScale,
            request.Width,
            request.Height,
            request.BatchSize,
            request.Sampler,
            request.Seed,
            loras);

        HttpResponseMessage response = await _http
            .PostAsJsonAsync("/api/v1/generations/text-to-image", body, ct)
            .ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        return ToGenerationResult(await response.Content
            .ReadFromJsonAsync<GenerationJobDto>(ct)
            .ConfigureAwait(false)
            ?? throw new InvalidOperationException("Empty response from backend."));
    }

    public async Task<GenerationResult> StartImageToImageAsync(GenerationRequest request, CancellationToken ct)
    {
        if (request.InitialImage is not string initialImage)
        {
            throw new InvalidOperationException("An initial image is required for image-to-image generation.");
        }

        List<LoraBody> loras = request.Loras?
            .Select(lora => new LoraBody(lora.ModelId, lora.Weight))
            .ToList() ?? [];
        ImageToImageBody body = new(
            request.Prompt, request.NegativePrompt, request.Steps, request.CfgScale,
            request.Width, request.Height, request.BatchSize, request.Sampler, request.Seed,
            loras, initialImage, request.Strength);

        HttpResponseMessage response = await _http
            .PostAsJsonAsync("/api/v1/generations/image-to-image", body, ct)
            .ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        return ToGenerationResult(await response.Content
            .ReadFromJsonAsync<GenerationJobDto>(ct)
            .ConfigureAwait(false)
            ?? throw new InvalidOperationException("Empty response from backend."));
    }

    public async Task<GenerationResult> GetGenerationJobAsync(string jobId, CancellationToken ct)
    {
        GenerationJobDto dto = await _http
            .GetFromJsonAsync<GenerationJobDto>($"/api/v1/generations/{Uri.EscapeDataString(jobId)}", ct)
            .ConfigureAwait(false)
            ?? throw new InvalidOperationException("Empty response from backend.");

        return ToGenerationResult(dto);
    }

    public async Task CancelGenerationJobAsync(string jobId, CancellationToken ct)
    {
        HttpResponseMessage response = await _http
            .DeleteAsync($"/api/v1/generations/{Uri.EscapeDataString(jobId)}", ct)
            .ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
    }

    static GenerationResult ToGenerationResult(GenerationJobDto dto) => new(
        dto.JobId,
        dto.Status,
        dto.ImagesCompleted,
        dto.TotalImages,
        dto.ImageUrl,
        dto.ImagePath,
        dto.Error);

    public async Task<byte[]> DownloadImageAsync(string imageUrl, CancellationToken ct) =>
        await _http.GetByteArrayAsync(imageUrl, ct).ConfigureAwait(false);

    public async Task<List<string>> GetSamplersAsync(CancellationToken ct) =>
        await _http.GetFromJsonAsync<List<string>>("/api/v1/samplers", ct).ConfigureAwait(false)
        ?? [];

    public async Task<HealthInfo> GetHealthAsync(CancellationToken ct)
    {
        HealthDto dto = await _http.GetFromJsonAsync<HealthDto>("/api/v1/health", ct).ConfigureAwait(false)
            ?? throw new InvalidOperationException("Empty response from backend.");

        return new HealthInfo(dto.Status, dto.Device, dto.LoadedModel, dto.ModelReady,
            dto.ModelError, dto.ModelLoadingStage, dto.ModelDownloadSource, dto.ModelDownloadDestination);
    }

    public async Task<List<ModelInfo>> GetModelsAsync(CancellationToken ct)
    {
        List<ModelDto>? dtos = await _http.GetFromJsonAsync<List<ModelDto>>("/api/v1/models", ct).ConfigureAwait(false);
        return dtos?.Select(dto => new ModelInfo(dto.ModelId, dto.IsActive, dto.SizeOnDiskBytes)).ToList() ?? [];
    }

    public async Task<ModelInfo> SetActiveModelAsync(string modelId, CancellationToken ct)
    {
        HttpResponseMessage response = await _modelSwitchHttp
            .PostAsJsonAsync("/api/v1/models/active", new SetActiveModelBody(modelId), ct)
            .ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            BackendErrorDto? error = await response.Content
                .ReadFromJsonAsync<BackendErrorDto>(ct)
                .ConfigureAwait(false);
            throw new InvalidOperationException(
                error?.Detail ?? $"Model switch failed ({(int)response.StatusCode}).");
        }

        ModelDto dto = await response.Content
            .ReadFromJsonAsync<ModelDto>(ct)
            .ConfigureAwait(false)
            ?? throw new InvalidOperationException("Empty response from backend.");

        return new ModelInfo(dto.ModelId, dto.IsActive, dto.SizeOnDiskBytes);
    }

    public async Task<List<LoraFileInfo>> GetLorasAsync(CancellationToken ct)
    {
        List<LoraDto>? dtos = await _http.GetFromJsonAsync<List<LoraDto>>("/api/v1/loras", ct).ConfigureAwait(false);
        return dtos?.Select(dto => new LoraFileInfo(dto.LoraId, dto.SizeOnDiskBytes)).ToList() ?? [];
    }

    public async Task<List<GalleryImageInfo>> GetGalleryImagesAsync(CancellationToken ct)
    {
        List<GalleryImageDto>? dtos = await _http
            .GetFromJsonAsync<List<GalleryImageDto>>("/api/v1/gallery/images", ct)
            .ConfigureAwait(false);

        return dtos?.Select(ToGalleryImageInfo).ToList() ?? [];
    }

    public async Task DeleteGalleryImageAsync(string imageId, CancellationToken ct)
    {
        HttpResponseMessage response = await _http
            .DeleteAsync($"/api/v1/gallery/images/{Uri.EscapeDataString(imageId)}", ct)
            .ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
    }

    static GalleryImageInfo ToGalleryImageInfo(GalleryImageDto dto) => new(
        dto.ImageId,
        dto.ImageUrl,
        dto.ImagePath,
        dto.CreatedAt,
        dto.Parameters is GalleryParametersDto parameters
            ? new GalleryParameters(
                parameters.Prompt,
                parameters.NegativePrompt,
                parameters.Steps,
                parameters.CfgScale,
                parameters.Width,
                parameters.Height,
                parameters.BatchSize,
                parameters.Sampler,
                parameters.Seed,
                parameters.Loras.Select(lora => new LoraSelection(lora.ModelId, lora.Weight)).ToList())
            : null);

    public async Task<ScanDirectories> GetScanDirectoriesAsync(CancellationToken ct)
    {
        ScanDirectoriesDto dto = await _http
            .GetFromJsonAsync<ScanDirectoriesDto>("/api/v1/settings/directories", ct)
            .ConfigureAwait(false)
            ?? throw new InvalidOperationException("Empty response from backend.");

        return new ScanDirectories(dto.ModelDir, dto.LoraDir);
    }

    public async Task<ScanDirectories> SetScanDirectoriesAsync(ScanDirectories directories, CancellationToken ct)
    {
        HttpResponseMessage response = await _http
            .PutAsJsonAsync(
                "/api/v1/settings/directories",
                new ScanDirectoriesDto(directories.ModelDir, directories.LoraDir),
                ct)
            .ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        ScanDirectoriesDto dto = await response.Content
            .ReadFromJsonAsync<ScanDirectoriesDto>(ct)
            .ConfigureAwait(false)
            ?? throw new InvalidOperationException("Empty response from backend.");

        return new ScanDirectories(dto.ModelDir, dto.LoraDir);
    }

    public void Dispose()
    {
        _http.Dispose();
        _modelSwitchHttp.Dispose();
    }

    sealed record TextToImageBody(
        string Prompt,
        [property: JsonPropertyName("negative_prompt")] string NegativePrompt,
        int Steps,
        [property: JsonPropertyName("cfg_scale")] double CfgScale,
        int Width,
        int Height,
        [property: JsonPropertyName("batch_size")] int BatchSize,
        string Sampler,
        long? Seed,
        List<LoraBody> Loras);

    sealed record LoraBody([property: JsonPropertyName("model_id")] string ModelId, double Weight);

    sealed record ImageToImageBody(
        string Prompt,
        [property: JsonPropertyName("negative_prompt")] string NegativePrompt,
        int Steps,
        [property: JsonPropertyName("cfg_scale")] double CfgScale,
        int Width,
        int Height,
        [property: JsonPropertyName("batch_size")] int BatchSize,
        string Sampler,
        long? Seed,
        List<LoraBody> Loras,
        [property: JsonPropertyName("initial_image")] string InitialImage,
        double Strength);

    sealed record GenerationJobDto(
        [property: JsonPropertyName("job_id")] string JobId,
        string Status,
        [property: JsonPropertyName("images_completed")] int ImagesCompleted,
        [property: JsonPropertyName("total_images")] int TotalImages,
        [property: JsonPropertyName("image_url")] string? ImageUrl,
        [property: JsonPropertyName("image_path")] string? ImagePath,
        string? Error);

    sealed record HealthDto(
        string Status,
        string Device,
        [property: JsonPropertyName("loaded_model")] string? LoadedModel,
        [property: JsonPropertyName("model_ready")] bool ModelReady,
        [property: JsonPropertyName("model_error")] string? ModelError,
        [property: JsonPropertyName("model_loading_stage")] string? ModelLoadingStage,
        [property: JsonPropertyName("model_download_source")] string? ModelDownloadSource,
        [property: JsonPropertyName("model_download_destination")] string? ModelDownloadDestination);

    sealed record GalleryImageDto(
        [property: JsonPropertyName("image_id")] string ImageId,
        [property: JsonPropertyName("image_url")] string ImageUrl,
        [property: JsonPropertyName("image_path")] string ImagePath,
        [property: JsonPropertyName("created_at")] double CreatedAt,
        GalleryParametersDto? Parameters);

    sealed record GalleryParametersDto(
        string Prompt,
        [property: JsonPropertyName("negative_prompt")] string NegativePrompt,
        int? Steps,
        [property: JsonPropertyName("cfg_scale")] double? CfgScale,
        int? Width,
        int? Height,
        [property: JsonPropertyName("batch_size")] int? BatchSize,
        string? Sampler,
        long? Seed,
        List<LoraBody> Loras);

    sealed record ModelDto(
        [property: JsonPropertyName("model_id")] string ModelId,
        [property: JsonPropertyName("is_active")] bool IsActive,
        [property: JsonPropertyName("size_on_disk_bytes")] long SizeOnDiskBytes);

    sealed record SetActiveModelBody([property: JsonPropertyName("model_id")] string ModelId);

    sealed record BackendErrorDto(string? Detail);

    sealed record LoraDto(
        [property: JsonPropertyName("lora_id")] string LoraId,
        [property: JsonPropertyName("size_on_disk_bytes")] long SizeOnDiskBytes);

    sealed record ScanDirectoriesDto(
        [property: JsonPropertyName("model_dir")] string? ModelDir,
        [property: JsonPropertyName("lora_dir")] string? LoraDir);
}

sealed record HealthInfo(string Status, string Device, string? LoadedModel, bool ModelReady,
    string? ModelError, string? ModelLoadingStage, string? ModelDownloadSource, string? ModelDownloadDestination);

sealed record ScanDirectories(string? ModelDir, string? LoraDir);
