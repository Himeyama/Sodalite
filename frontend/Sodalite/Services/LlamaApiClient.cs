using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Sodalite.Services;

/// <summary>llama.cpp server の OpenAI 互換エンドポイントだけを扱う軽量クライアント。</summary>
sealed class LlamaApiClient : IDisposable
{
    // llama.cpp server はメッセージ中の null の tool_calls / tool_call_id を受け付けず
    // HTTP 500 にする版があるため、OpenAI 互換APIへは null プロパティを送らない。
    static readonly JsonSerializerOptions RequestJsonOptions = new(JsonSerializerDefaults.Web)
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    readonly HttpClient _http = new()
    {
        BaseAddress = new Uri("http://127.0.0.1:8080/v1/"),
        Timeout = TimeSpan.FromMinutes(5),
    };

    public async Task<List<string>> GetModelsAsync(CancellationToken ct)
    {
        ModelsResponse response = await _http.GetFromJsonAsync<ModelsResponse>("models", ct).ConfigureAwait(false)
            ?? throw new InvalidOperationException("Empty response from llama.cpp.");
        return response.Data.Select(model => model.Id).Where(id => !string.IsNullOrWhiteSpace(id)).ToList();
    }

    public async Task<ChatResponseMessage> CompleteAsync(
        string model, List<ChatRequestMessage> messages, CancellationToken ct)
    {
        ChatCompletionRequest request = new(model, messages, ChatTools, ToolChoice: "auto");
        HttpResponseMessage response = await _http
            .PostAsJsonAsync("chat/completions", request, RequestJsonOptions, ct)
            .ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            string detail = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException($"llama.cpp returned {(int)response.StatusCode}: {detail}");
        }

        ChatCompletionResponse completion = await response.Content.ReadFromJsonAsync<ChatCompletionResponse>(ct)
            .ConfigureAwait(false)
            ?? throw new InvalidOperationException("Empty response from llama.cpp.");
        return completion.Choices.FirstOrDefault()?.Message
            ?? throw new InvalidOperationException("llama.cpp returned no response choices.");
    }

    public void Dispose() => _http.Dispose();

    static readonly List<ChatTool> ChatTools =
    [
        new("function", new ToolFunction(
            "update_image_prompts",
            "Update the Stable Diffusion image prompt and/or negative prompt. Use this whenever the user asks to create, revise, add to, remove from, or otherwise change either prompt.",
            JsonDocument.Parse("""
                {"type":"object","properties":{"prompt":{"type":"string","description":"The complete updated positive image prompt."},"negative_prompt":{"type":"string","description":"The complete updated negative prompt."}},"required":["prompt","negative_prompt"],"additionalProperties":false}
                """).RootElement.Clone()))
    ];

    sealed record ModelsResponse(List<ModelDto> Data);
    sealed record ModelDto(string Id);
    sealed record ChatCompletionRequest(
        string Model,
        List<ChatRequestMessage> Messages,
        List<ChatTool> Tools,
        [property: JsonPropertyName("tool_choice")] string ToolChoice);
    sealed record ChatCompletionResponse(List<Choice> Choices);
    sealed record Choice(ChatResponseMessage Message);
    sealed record ChatTool(string Type, ToolFunction Function);
    sealed record ToolFunction(string Name, string Description, JsonElement Parameters);
}

sealed record ChatRequestMessage(
    string Role,
    string? Content = null,
    [property: JsonPropertyName("tool_calls")] List<ChatToolCall>? ToolCalls = null,
    [property: JsonPropertyName("tool_call_id")] string? ToolCallId = null);

sealed record ChatResponseMessage(
    string Role,
    string? Content,
    [property: JsonPropertyName("tool_calls")] List<ChatToolCall>? ToolCalls,
    [property: JsonPropertyName("reasoning_content")] string? ReasoningContent = null);

sealed record ChatToolCall(string Id, string Type, ChatToolFunction Function);
sealed record ChatToolFunction(string Name, string Arguments);
