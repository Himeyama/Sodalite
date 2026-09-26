namespace Sodalite.Models;

sealed record GenerationResult(
    string JobId,
    string Status,
    int CurrentStep,
    int TotalSteps,
    int ImagesCompleted,
    int TotalImages,
    string? ImageUrl,
    string? ImagePath,
    string? Error);
