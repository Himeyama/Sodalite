using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Encodings.Web;
using System.Globalization;
using Microsoft.UI.Input;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Documents;
using Microsoft.UI.Xaml.Input;
using Sodalite.Services;
using Windows.System;

namespace Sodalite.Views;

/// <summary>llama.cpp と会話し、ツール呼び出しだけで生成用プロンプトを更新するパネル。</summary>
public sealed partial class ChatPanel : UserControl, IDisposable
{
    const string AssistantInstructions = """
        You are an assistant for Stable Diffusion image generation. Reply in the operating system's display language specified below; Markdown is allowed. Keep any explanation useful and concise rather than exposing private step-by-step reasoning.
        Before updating prompts, reason carefully and thoroughly in private. Consider the entire conversation, the current positive and negative prompts, every concrete visual requirement in the newest user message, requested additions, removals, replacements, and any conflicts between them. Resolve ambiguities conservatively without dropping requested details.
        Treat a user message that describes an image or a desired change as a direct prompt instruction and call update_image_prompts. Build a lean, high-signal prompt: include only terms necessary for the user's request and the relevant existing image. Use each tag only once. Remove duplicate terms, near-synonyms, conflicting terms, and irrelevant leftover terms. Do not retain an existing term merely because it is already present; retain it only when it remains directly relevant. When the user asks for a wholly new image, replace the prior positive prompt rather than combining unrelated content.
        Prioritize the main subject and only its essential attributes. Avoid fine-grained depiction because excessive detail can reduce generation quality: omit decorative, technical, and secondary details unless they are indispensable to the user's request. Do not invent any details. Use the current image prompt and negative prompt as context when determining what remains relevant, then pass complete replacement values for both prompt and negative_prompt. Unless the user explicitly asks to omit or change them, include this comprehensive quality baseline in every update. Positive tags: "masterpiece", "best quality", "ultra-detailed", "intricate details", "highres". Negative tags: "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art)", "(watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2)", "(blur, blurry, grainy)", "morbid", "ugly", "asymmetrical", "mutated malformed", "mutilated", "poorly lit", "bad shadow", "draft", "cropped", "out of frame", "cut off", "censored", "jpeg artifacts", "out of focus", "glitch", "duplicate", "(airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1)", "(3D, 3D Game, 3D Game Scene, 3D Character:1.1)", "(watermark, bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)". Do not add other quality tags, styles, composition, lighting, camera terms, negative keywords, or other details unless the user explicitly requested them.
        Prompts must contain only comma-separated standalone keywords, never sentences, prose, noun phrases, or grammar words. Prefer common, well-established Stable Diffusion and image-dataset tags over rare wording, elaborate synonyms, or newly coined descriptions. Do not use prepositions, conjunctions, articles, or other connector words such as "on", "in", "at", "with", "and", "the", or "a". Translate the user's request almost literally into the minimum necessary keywords. Do not claim that prompts changed unless you called the tool. After calling the tool, never repeat or display the complete updated prompt or negative prompt in chat; only give a brief Japanese explanation of what you changed. Keep image prompts concise and suitable for Stable Diffusion; English prompt keywords are preferred when useful.

        Prompt-format examples (use these only as references for tag style and vocabulary; never copy unrelated terms into a user's prompt):
        Positive: masterpiece, best quality, ultra-detailed, intricate details, highres, basketball, 1girl
        Negative: (worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1), (3D, 3D Game, 3D Game Scene, 3D Character:1.1), (watermark, bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)
        """;

    readonly LlamaApiClient _client = new();
    readonly List<ChatRequestMessage> _history = [];
    readonly List<ChatLogEntry> _chatLog = [];
    readonly Microsoft.UI.Dispatching.DispatcherQueueTimer _availabilityTimer;
    Func<(string Prompt, string NegativePrompt)>? _getPrompts;
    bool _isCheckingAvailability;
    bool _isAvailable;
    bool _isChatLogVisible;
    bool _isSending;

    public event EventHandler<bool>? AvailabilityChanged;
    public event EventHandler<string?>? SelectedModelChanged;
    public event EventHandler<ChatPromptUpdate>? PromptUpdateRequested;

    public ChatPanel()
    {
        InitializeComponent();

        _availabilityTimer = DispatcherQueue.CreateTimer();
        _availabilityTimer.Interval = TimeSpan.FromSeconds(10);
        _availabilityTimer.Tick += async (_, _) => await RefreshAvailabilityAsync();
    }

    internal async Task InitializeAsync(Func<(string Prompt, string NegativePrompt)> getPrompts)
    {
        _getPrompts = getPrompts;
        await RefreshAvailabilityAsync();
        _availabilityTimer.Start();
    }

    /// <summary>llama.cpp の /v1/models を定期確認し、状態が変わったときだけ親へ通知する。</summary>
    async Task RefreshAvailabilityAsync()
    {
        if (_isCheckingAvailability)
        {
            return;
        }

        _isCheckingAvailability = true;
        using CancellationTokenSource timeout = new(TimeSpan.FromSeconds(3));
        try
        {
            List<string> models = await _client.GetModelsAsync(timeout.Token);
            if (models.Count == 0)
            {
                SetAvailability(false);
                return;
            }

            ModelComboBox.ItemsSource = models;
            if (ModelComboBox.SelectedItem is not string selectedModel || !models.Contains(selectedModel))
            {
                string? savedModel = AppSettings.LastChatModelId;
                ModelComboBox.SelectedItem = models.Contains(savedModel ?? "") ? savedModel : models[0];
            }

            SetAvailability(true);
        }
        catch (Exception)
        {
            // llama.cpp は任意の外部プロセス。未起動・タイムアウト時は UI を出さない。
            SetAvailability(false);
        }
        finally
        {
            _isCheckingAvailability = false;
        }
    }

    void SetAvailability(bool available)
    {
        if (_isAvailable == available)
        {
            return;
        }

        _isAvailable = available;
        AvailabilityChanged?.Invoke(this, available);
    }

    void ModelComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        string? model = ModelComboBox.SelectedItem as string;
        AppSettings.LastChatModelId = model;
        SelectedModelChanged?.Invoke(this, model);
    }

    async void SendButton_Click(object sender, RoutedEventArgs e) => await SendAsync();

    void MessageTextBox_TextChanged(object sender, TextChangedEventArgs e) => UpdateSendButtonState();

    async void MessageTextBox_KeyDown(object sender, KeyRoutedEventArgs e)
    {
        if (e.Key == VirtualKey.Enter && !e.KeyStatus.IsMenuKeyDown && !IsShiftKeyDown())
        {
            e.Handled = true;
            await SendAsync();
        }
    }

    static bool IsShiftKeyDown() =>
        (InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Shift) & Windows.UI.Core.CoreVirtualKeyStates.Down) != 0;

    async Task SendAsync()
    {
        string content = MessageTextBox.Text.Trim();
        if (_isSending || string.IsNullOrEmpty(content) || ModelComboBox.SelectedItem is not string model)
        {
            return;
        }

        _isSending = true;
        SendButton.IsEnabled = false;
        MessageTextBox.IsEnabled = false;
        SendingProgressRing.Visibility = Visibility.Visible;
        SendingProgressRing.IsActive = true;
        try
        {
            MessageTextBox.Text = "";
            AddUserMessage(content);
            _history.Add(new ChatRequestMessage("user", content));
            _chatLog.Add(new ChatLogEntry("user", content));
            RefreshChatLog();
            await GetAssistantResponseAsync(model);
        }
        catch (Exception ex)
        {
            AddAssistantMessage($"**通信エラー:** {ex.Message}");
        }
        finally
        {
            _isSending = false;
            MessageTextBox.IsEnabled = true;
            UpdateSendButtonState();
            SendingProgressRing.IsActive = false;
            SendingProgressRing.Visibility = Visibility.Collapsed;
            MessageTextBox.Focus(FocusState.Programmatic);
        }
    }

    void UpdateSendButtonState()
    {
        if (SendButton is not null)
        {
            SendButton.IsEnabled = !_isSending && !string.IsNullOrWhiteSpace(MessageTextBox.Text);
        }
    }

    async Task GetAssistantResponseAsync(string model)
    {
        // ツール結果を受けて最終応答を返すモデルにも対応する。無限ツール呼び出しは防ぐ。
        for (int round = 0; round < 4; round++)
        {
            ChatResponseMessage response = await _client.CompleteAsync(model, BuildRequestMessages(), CancellationToken.None);
            _history.Add(new ChatRequestMessage("assistant", response.Content, response.ToolCalls));
            // ツール応答には更新済みプロンプトが混じり得るため、本文・推論ともログには残さない。
            if (response.ToolCalls is not { Count: > 0 })
            {
                _chatLog.Add(new ChatLogEntry("assistant", response.Content, response.ReasoningContent));
            }
            RefreshChatLog();

            // ツール呼び出しを伴う途中応答には、モデルがプロンプト全文を含める場合がある。
            // 入力欄が正本なので、ここでは表示せずツール結果後の短い説明だけを見せる。
            if (response.ToolCalls is not { Count: > 0 } && !string.IsNullOrWhiteSpace(response.Content))
            {
                AddAssistantMessage(response.Content);
            }

            if (response.ToolCalls is not { Count: > 0 })
            {
                return;
            }

            foreach (ChatToolCall call in response.ToolCalls)
            {
                string result = ApplyToolCall(call);
                _history.Add(new ChatRequestMessage("tool", result, ToolCallId: call.Id));
            }
        }

        AddAssistantMessage("ツール呼び出しの上限に達しました。");
    }

    List<ChatRequestMessage> BuildRequestMessages()
    {
        (string prompt, string negativePrompt) = _getPrompts?.Invoke() ?? ("", "");
        string systemLanguage = CultureInfo.CurrentUICulture.Name;
        string system = $"{AssistantInstructions}\n\nOperating system display language: {systemLanguage}\n\nCurrent image prompt:\n{prompt}\n\nCurrent negative prompt:\n{negativePrompt}";
        List<ChatRequestMessage> messages = [new("system", system)];
        messages.AddRange(_history);
        return messages;
    }

    string ApplyToolCall(ChatToolCall call)
    {
        if (call.Function.Name != "update_image_prompts")
        {
            return "Unsupported tool.";
        }

        try
        {
            using JsonDocument arguments = JsonDocument.Parse(call.Function.Arguments);
            JsonElement root = arguments.RootElement;
            string prompt = RemoveDuplicateTags(root.GetProperty("prompt").GetString() ?? "");
            string negativePrompt = RemoveDuplicateTags(root.GetProperty("negative_prompt").GetString() ?? "");
            PromptUpdateRequested?.Invoke(this, new ChatPromptUpdate(prompt, negativePrompt));
            return "The image prompts were updated in the application.";
        }
        catch (Exception)
        {
            return "The tool arguments were invalid; no prompt was changed.";
        }
    }

    static string RemoveDuplicateTags(string prompt)
    {
        HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);
        List<string> uniqueTags = [];
        foreach (string tag in prompt.Split([',', '，'], StringSplitOptions.RemoveEmptyEntries))
        {
            string trimmedTag = tag.Trim();
            if (!string.IsNullOrEmpty(trimmedTag) && seen.Add(trimmedTag))
            {
                uniqueTags.Add(trimmedTag);
            }
        }

        return string.Join(", ", uniqueTags);
    }

    void ResetButton_Click(object sender, RoutedEventArgs e)
    {
        _history.Clear();
        _chatLog.Clear();
        MessagesPanel.Children.Clear();
        RefreshChatLog();
    }

    /// <summary>右クリック時だけ入力欄の下に会話ログを展開する隠し操作。</summary>
    void ChatPanel_RightTapped(object sender, RightTappedRoutedEventArgs e)
    {
        _isChatLogVisible = !_isChatLogVisible;
        ChatLogScrollViewer.Visibility = _isChatLogVisible ? Visibility.Visible : Visibility.Collapsed;
        if (_isChatLogVisible)
        {
            RefreshChatLog();
        }
    }

    void RefreshChatLog()
    {
        if (!_isChatLogVisible)
        {
            return;
        }

        ChatLogTextBlock.Text = JsonSerializer.Serialize(_chatLog, new JsonSerializerOptions
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        });
    }

    void AddUserMessage(string content)
    {
        TextBlock text = new()
        {
            Text = content,
            TextWrapping = TextWrapping.Wrap,
            IsTextSelectionEnabled = true,
            Foreground = new Microsoft.UI.Xaml.Media.SolidColorBrush(
                Windows.UI.Color.FromArgb(255, 255, 255, 255)),
        };
        Border bubble = new()
        {
            Child = text,
            Padding = new Thickness(10, 6, 10, 6),
            CornerRadius = new CornerRadius(0),
            Background = new Microsoft.UI.Xaml.Media.SolidColorBrush(
                Windows.UI.Color.FromArgb(255, 0, 120, 212)),
            HorizontalAlignment = HorizontalAlignment.Right,
            MaxWidth = 300,
        };
        // WinUI の CornerRadius は絶対値指定。高さに対する丸みを 50% に抑える。
        bubble.SizeChanged += (_, _) => bubble.CornerRadius = new CornerRadius(bubble.ActualHeight / 4);
        MessagesPanel.Children.Add(bubble);
        ScrollToBottom();
    }

    void AddAssistantMessage(string markdown)
    {
        RichTextBlock text = new() { TextWrapping = TextWrapping.Wrap, IsTextSelectionEnabled = true };
        AddMarkdownBlocks(text, markdown);
        MessagesPanel.Children.Add(text);
        ScrollToBottom();
    }

    static void AddMarkdownBlocks(RichTextBlock target, string markdown)
    {
        foreach (string line in markdown.Replace("\r\n", "\n").Split('\n'))
        {
            Paragraph paragraph = new();
            string value = line;
            if (value.StartsWith("# "))
            {
                value = value[2..];
                paragraph.FontSize = 18;
                paragraph.FontWeight = Microsoft.UI.Text.FontWeights.SemiBold;
            }
            else if (value.StartsWith("## "))
            {
                value = value[3..];
                paragraph.FontSize = 16;
                paragraph.FontWeight = Microsoft.UI.Text.FontWeights.SemiBold;
            }
            else if (value.StartsWith("- ") || value.StartsWith("* "))
            {
                paragraph.Inlines.Add(new Run { Text = "• " });
                value = value[2..];
            }

            AddInlineMarkdown(paragraph, value);
            target.Blocks.Add(paragraph);
        }
    }

    static void AddInlineMarkdown(Paragraph paragraph, string text)
    {
        bool bold = false;
        string[] parts = text.Split("**");
        for (int i = 0; i < parts.Length; i++)
        {
            if (i > 0)
            {
                bold = !bold;
            }

            if (!string.IsNullOrEmpty(parts[i]))
            {
                paragraph.Inlines.Add(new Run { Text = parts[i], FontWeight = bold ? Microsoft.UI.Text.FontWeights.SemiBold : Microsoft.UI.Text.FontWeights.Normal });
            }
        }
    }

    void ScrollToBottom() => _ = DispatcherQueue.TryEnqueue(() => MessagesScrollViewer.ChangeView(null, MessagesScrollViewer.ScrollableHeight, null));

    public void Dispose()
    {
        _availabilityTimer.Stop();
        _client.Dispose();
    }
}

public sealed record ChatPromptUpdate(string Prompt, string NegativePrompt);

/// <summary>右クリックで開くデバッグ用ログの1レコード。reasoning は llama.cpp が返した場合のみ含める。</summary>
sealed record ChatLogEntry(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] string? Content,
    [property: JsonPropertyName("reasoning")] string? Reasoning = null);
