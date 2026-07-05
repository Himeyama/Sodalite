namespace Sodalite.Models;

sealed record LoraSelection(string ModelId, double Weight);

sealed record GenerationRequest(
    string Prompt,
    string NegativePrompt = "",
    int Steps = 20,
    double CfgScale = 7.0,
    int Width = 512,
    int Height = 512,
    string Sampler = "euler_a",
    long? Seed = null,
    IReadOnlyList<LoraSelection>? Loras = null);
