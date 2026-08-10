using System.Text.Json;
using System.Text.Json.Serialization;

namespace Sodalite.Services;

/// <summary>
/// Unpackaged実行のため Windows.Storage.ApplicationData は使えず(パッケージIDが必要)、
/// %LOCALAPPDATA%配下に自前でJSON設定ファイルを読み書きする。
/// </summary>
static class AppSettings
{
    static readonly string SettingsFilePath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Sodalite",
        "settings.json");

    public static string? LastModelId
    {
        get => Load().LastModelId;
        set => Save(Load() with { LastModelId = value });
    }

    /// <summary>次回バックエンド起動時に webui (LANアクセス可能なブラウザUI) を有効にするか。</summary>
    public static bool WebUiEnabled
    {
        get => Load().WebUiEnabled;
        set => Save(Load() with { WebUiEnabled = value });
    }

    static Settings Load()
    {
        try
        {
            string json = File.ReadAllText(SettingsFilePath);
            return JsonSerializer.Deserialize<Settings>(json) ?? new Settings(null, false);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException)
        {
            return new Settings(null, false);
        }
    }

    static void Save(Settings settings)
    {
        string? directory = Path.GetDirectoryName(SettingsFilePath);
        if (directory is not null)
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(SettingsFilePath, JsonSerializer.Serialize(settings));
    }

    sealed record Settings(
        [property: JsonPropertyName("last_model_id")] string? LastModelId,
        [property: JsonPropertyName("webui_enabled")] bool WebUiEnabled);
}
