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

    public async Task<GenerationResult> GenerateTextToImageAsync(GenerationRequest request, CancellationToken ct)
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
            request.Sampler,
            request.Seed,
            loras);

        HttpResponseMessage response = await _http
            .PostAsJsonAsync("/api/v1/generations/text-to-image", body, ct)
            .ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        GenerationJobDto dto = await response.Content
            .ReadFromJsonAsync<GenerationJobDto>(ct)
            .ConfigureAwait(false)
            ?? throw new InvalidOperationException("Empty response from backend.");

        return new GenerationResult(dto.JobId, dto.Status, dto.ImageUrl, dto.Error);
    }

    public async Task<byte[]> DownloadImageAsync(string imageUrl, CancellationToken ct) =>
        await _http.GetByteArrayAsync(imageUrl, ct).ConfigureAwait(false);

    public async Task<List<string>> GetSamplersAsync(CancellationToken ct) =>
        await _http.GetFromJsonAsync<List<string>>("/api/v1/samplers", ct).ConfigureAwait(false)
        ?? [];

    public async Task<HealthInfo> GetHealthAsync(CancellationToken ct)
    {
        HealthDto dto = await _http.GetFromJsonAsync<HealthDto>("/api/v1/health", ct).ConfigureAwait(false)
            ?? throw new InvalidOperationException("Empty response from backend.");

        return new HealthInfo(dto.Status, dto.Device, dto.LoadedModel);
    }

    public async Task<List<ModelInfo>> GetModelsAsync(CancellationToken ct)
    {
        List<ModelDto>? dtos = await _http.GetFromJsonAsync<List<ModelDto>>("/api/v1/models", ct).ConfigureAwait(false);
        return dtos?.Select(dto => new ModelInfo(dto.ModelId, dto.IsActive, dto.SizeOnDiskBytes)).ToList() ?? [];
    }

    public async Task<ModelInfo> SetActiveModelAsync(string modelId, CancellationToken ct)
    {
        HttpResponseMessage response = await _http
            .PostAsJsonAsync("/api/v1/models/active", new SetActiveModelBody(modelId), ct)
            .ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        ModelDto dto = await response.Content
            .ReadFromJsonAsync<ModelDto>(ct)
            .ConfigureAwait(false)
            ?? throw new InvalidOperationException("Empty response from backend.");

        return new ModelInfo(dto.ModelId, dto.IsActive, dto.SizeOnDiskBytes);
    }

    public async Task<ModelInfo> ImportModelAsync(string modelPath, CancellationToken ct)
    {
        HttpResponseMessage response = await _http
            .PostAsJsonAsync("/api/v1/models/imported", new ImportModelBody(modelPath), ct)
            .ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

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

    public async Task<LoraFileInfo> ImportLoraAsync(string loraPath, CancellationToken ct)
    {
        HttpResponseMessage response = await _http
            .PostAsJsonAsync("/api/v1/loras/imported", new ImportLoraBody(loraPath), ct)
            .ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        LoraDto dto = await response.Content
            .ReadFromJsonAsync<LoraDto>(ct)
            .ConfigureAwait(false)
            ?? throw new InvalidOperationException("Empty response from backend.");

        return new LoraFileInfo(dto.LoraId, dto.SizeOnDiskBytes);
    }

    public void Dispose() => _http.Dispose();

    sealed record TextToImageBody(
        string Prompt,
        [property: JsonPropertyName("negative_prompt")] string NegativePrompt,
        int Steps,
        [property: JsonPropertyName("cfg_scale")] double CfgScale,
        int Width,
        int Height,
        string Sampler,
        long? Seed,
        List<LoraBody> Loras);

    sealed record LoraBody([property: JsonPropertyName("model_id")] string ModelId, double Weight);

    sealed record GenerationJobDto(
        [property: JsonPropertyName("job_id")] string JobId,
        string Status,
        [property: JsonPropertyName("image_url")] string? ImageUrl,
        string? Error);

    sealed record HealthDto(
        string Status,
        string Device,
        [property: JsonPropertyName("loaded_model")] string LoadedModel);

    sealed record ModelDto(
        [property: JsonPropertyName("model_id")] string ModelId,
        [property: JsonPropertyName("is_active")] bool IsActive,
        [property: JsonPropertyName("size_on_disk_bytes")] long SizeOnDiskBytes);

    sealed record SetActiveModelBody([property: JsonPropertyName("model_id")] string ModelId);

    sealed record ImportModelBody([property: JsonPropertyName("model_path")] string ModelPath);

    sealed record LoraDto(
        [property: JsonPropertyName("lora_id")] string LoraId,
        [property: JsonPropertyName("size_on_disk_bytes")] long SizeOnDiskBytes);

    sealed record ImportLoraBody([property: JsonPropertyName("lora_path")] string LoraPath);
}

sealed record HealthInfo(string Status, string Device, string LoadedModel);
