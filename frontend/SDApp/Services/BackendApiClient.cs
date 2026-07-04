using System.Net.Http.Json;
using System.Text.Json.Serialization;
using SDApp.Models;

namespace SDApp.Services;

sealed class BackendApiClient(int port) : IDisposable
{
    readonly HttpClient _http = new() { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

    public async Task<GenerationResult> GenerateTextToImageAsync(GenerationRequest request, CancellationToken ct)
    {
        TextToImageBody body = new(
            request.Prompt,
            request.NegativePrompt,
            request.Steps,
            request.CfgScale,
            request.Width,
            request.Height,
            request.Sampler,
            request.Seed);

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

    public void Dispose() => _http.Dispose();

    sealed record TextToImageBody(
        string Prompt,
        [property: JsonPropertyName("negative_prompt")] string NegativePrompt,
        int Steps,
        [property: JsonPropertyName("cfg_scale")] double CfgScale,
        int Width,
        int Height,
        string Sampler,
        long? Seed);

    sealed record GenerationJobDto(
        [property: JsonPropertyName("job_id")] string JobId,
        string Status,
        [property: JsonPropertyName("image_url")] string? ImageUrl,
        string? Error);

    sealed record HealthDto(
        string Status,
        string Device,
        [property: JsonPropertyName("loaded_model")] string LoadedModel);
}

sealed record HealthInfo(string Status, string Device, string LoadedModel);
