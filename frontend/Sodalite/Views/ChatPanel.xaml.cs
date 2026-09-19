using System.Text.Json;
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
        You are an assistant for Stable Diffusion image generation. Before replying or updating prompts, carefully reason about the user's intent, the current positive and negative prompts, visual composition, likely generation results, and any trade-offs. Do not rush to a superficial answer. Reply in the operating system's display language specified below; Markdown is allowed. Keep any explanation useful and concise rather than exposing private step-by-step reasoning.
        When the user asks to create, revise, add, remove, or otherwise change an image prompt or negative prompt, you MUST call update_image_prompts. Pass the complete replacement values for both prompt and negative_prompt. Do not claim that prompts changed unless you called the tool. Keep image prompts concise and suitable for Stable Diffusion; English prompt keywords are preferred when useful.
        """;

    readonly LlamaApiClient _client = new();
    readonly List<ChatRequestMessage> _history = [];
    readonly Microsoft.UI.Dispatching.DispatcherQueueTimer _availabilityTimer;
    Func<(string Prompt, string NegativePrompt)>? _getPrompts;
    bool _isCheckingAvailability;
    bool _isAvailable;
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
            await GetAssistantResponseAsync(model);
        }
        catch (Exception ex)
        {
            AddAssistantMessage($"**通信エラー:** {ex.Message}");
        }
        finally
        {
            _isSending = false;
            SendButton.IsEnabled = true;
            MessageTextBox.IsEnabled = true;
            SendingProgressRing.IsActive = false;
            SendingProgressRing.Visibility = Visibility.Collapsed;
            MessageTextBox.Focus(FocusState.Programmatic);
        }
    }

    async Task GetAssistantResponseAsync(string model)
    {
        // ツール結果を受けて最終応答を返すモデルにも対応する。無限ツール呼び出しは防ぐ。
        for (int round = 0; round < 4; round++)
        {
            ChatResponseMessage response = await _client.CompleteAsync(model, BuildRequestMessages(), CancellationToken.None);
            _history.Add(new ChatRequestMessage("assistant", response.Content, response.ToolCalls));

            if (!string.IsNullOrWhiteSpace(response.Content))
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
            string prompt = root.GetProperty("prompt").GetString() ?? "";
            string negativePrompt = root.GetProperty("negative_prompt").GetString() ?? "";
            PromptUpdateRequested?.Invoke(this, new ChatPromptUpdate(prompt, negativePrompt));
            return "The image prompts were updated in the application.";
        }
        catch (Exception)
        {
            return "The tool arguments were invalid; no prompt was changed.";
        }
    }

    void ResetButton_Click(object sender, RoutedEventArgs e)
    {
        _history.Clear();
        MessagesPanel.Children.Clear();
    }

    void AddUserMessage(string content)
    {
        TextBlock text = new()
        {
            Text = content,
            TextWrapping = TextWrapping.Wrap,
            Foreground = new Microsoft.UI.Xaml.Media.SolidColorBrush(
                Windows.UI.Color.FromArgb(255, 255, 255, 255)),
        };
        Border bubble = new()
        {
            Child = text,
            Padding = new Thickness(10, 6, 10, 6),
            CornerRadius = new CornerRadius(10),
            Background = new Microsoft.UI.Xaml.Media.SolidColorBrush(
                Windows.UI.Color.FromArgb(255, 0, 120, 212)),
            HorizontalAlignment = HorizontalAlignment.Right,
            MaxWidth = 300,
        };
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
