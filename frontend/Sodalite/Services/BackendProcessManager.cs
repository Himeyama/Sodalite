using System.Diagnostics;
using System.Net;
using System.Net.Sockets;

namespace Sodalite.Services;

sealed class BackendProcessManager : IAsyncDisposable
{
    // webui (LANアクセス対応) 有効時に使う固定ポート。LAN内の他端末からアクセスする際に
    // 毎回変わるのは不便なため、動的ポートではなく固定値にする。
    const int WebUiPort = 8188;

    static readonly TimeSpan HealthCheckInterval = TimeSpan.FromMilliseconds(500);
    static readonly TimeSpan HealthCheckTimeout = TimeSpan.FromSeconds(120);

    readonly string _backendProjectPath;
    readonly BackendEnvironmentSetup _environmentSetup;
    readonly JobObject _jobObject = new();
    Process? _process;

    public int Port { get; private set; }

    /// <summary>
    /// webui (LANアクセス対応) が有効な場合の接続先 URL。LAN内の他端末からアクセスする際に案内する用途。
    /// IPv4 の LAN アドレスが見つからない場合はループバックにフォールバックする。
    /// </summary>
    public string? WebUiUrl { get; private set; }

    public BackendProcessManager(string backendProjectPath)
    {
        _backendProjectPath = backendProjectPath;
        _environmentSetup = new BackendEnvironmentSetup(backendProjectPath);
    }

    /// <param name="onSetupProgress">
    /// 初回セットアップ(uv sync)開始直前に空文字で一度、その後 uv sync のログ行が出るたびに呼ばれる。
    /// UI に進捗を実況表示するために使う。セットアップ済みでスキップする場合は一度も呼ばれない。
    /// </param>
    public async Task<int> StartAsync(
        string? modelId = null,
        bool enableWebUi = false,
        IProgress<string>? onSetupProgress = null,
        CancellationToken ct = default)
    {
        // uv run の前に、Python 仮想環境が用意済みであることを保証する。未セットアップ or 前回失敗なら
        // ここで uv sync を実行する(成功時のみマーカーが書かれ、失敗時は次回起動で再試行される)。
        await _environmentSetup.EnsureAsync(onSetupProgress, ct).ConfigureAwait(false);

        Port = enableWebUi ? WebUiPort : FindFreePort();
        WebUiUrl = enableWebUi ? $"http://{FindLanIPAddress()}:{Port}" : null;

        ProcessStartInfo startInfo = new()
        {
            FileName = "uv",
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        // hf_xet 経由のダウンロードはこの環境でハングすることがあり、モデル切り替え時の
        // from_pretrained がそのまま返らなくなる。通常の HTTP ダウンロードに固定する。
        startInfo.Environment["HF_HUB_DISABLE_XET"] = "1";

        if (_environmentSetup.Accelerator == "rocm")
        {
            // MIOpen's default cache-miss path benchmarks/compiles many convolution
            // kernels. On SDXL this can block the first generation for several
            // minutes and produces a sawtooth GPU-utilization graph. FAST uses an
            // existing FindDb entry or the immediate fallback instead.
            startInfo.Environment["MIOPEN_FIND_MODE"] = "FAST";
            startInfo.Environment["MIOPEN_FIND_ENFORCE"] = "NONE";
        }

        // uv sync (BackendEnvironmentSetup) が作った仮想環境を使う。同じ場所を指さないと
        // uv run が別の .venv を作り直そうとして起動が壊れるため、必ず一致させること
        // (開発構成では null なので設定せず backend\.venv を使う)。
        if (BackendLocator.VenvPath is string venvPath)
        {
            startInfo.Environment["UV_PROJECT_ENVIRONMENT"] = venvPath;
        }

        startInfo.ArgumentList.Add("run");
        startInfo.ArgumentList.Add("--project");
        startInfo.ArgumentList.Add(_backendProjectPath);
        // uv run also resolves project dependencies. Keep it on the same accelerator
        // extra selected by uv sync, otherwise it can replace the accelerator-specific
        // PyTorch build with the default PyPI build immediately before backend startup.
        startInfo.ArgumentList.Add("--extra");
        startInfo.ArgumentList.Add(_environmentSetup.Accelerator);
        // EnsureAsync already performed an exact sync. DirectML additionally
        // removes torchvision after that sync, so do not let uv reinstall it.
        startInfo.ArgumentList.Add("--no-sync");
        startInfo.ArgumentList.Add("Sodalite-backend");
        startInfo.ArgumentList.Add("--port");
        startInfo.ArgumentList.Add(Port.ToString());

        if (!string.IsNullOrEmpty(modelId))
        {
            startInfo.ArgumentList.Add("--model-id");
            startInfo.ArgumentList.Add(modelId);
        }

        // --webui を付けると config.py 側で host の既定が 0.0.0.0 になり、LAN 内の他端末から
        // 無認証でアクセス可能になる(信頼できる LAN 前提)。ユーザーが明示的に有効化した場合のみ付与する。
        if (enableWebUi)
        {
            startInfo.ArgumentList.Add("--webui");
        }

        _process = Process.Start(startInfo) ?? throw new InvalidOperationException("Failed to start backend process.");

        // stdout/stderr をリダイレクトした以上、必ず読み出し続けてパイプを空にする。
        // diffusers/tqdm は推論のたびに数 KB を stderr へ出力するため、読み出さないと
        // OS のパイプバッファ(数 KB)が数回の生成で満杯になり、python が書き込みで
        // ブロック → 推論が停止して以降のリクエストが永久に返らなくなる
        // (「2回目以降の生成が終わらない」の真因)。ログ内容自体は使わないので破棄する。
        _process.OutputDataReceived += DiscardProcessOutput;
        _process.ErrorDataReceived += DiscardProcessOutput;
        _process.BeginOutputReadLine();
        _process.BeginErrorReadLine();

        // このアプリが WinRT の FailFast 等で異常終了しても、OS が確実に
        // uv/python の子プロセスツリーを終了させるよう Job Object に紐付ける。
        _jobObject.Assign(_process.SafeHandle);

        await WaitForHealthyAsync(ct).ConfigureAwait(false);

        return Port;
    }

    async Task WaitForHealthyAsync(CancellationToken ct)
    {
        using HttpClient http = new() { Timeout = TimeSpan.FromSeconds(2) };
        DateTime deadline = DateTime.UtcNow + HealthCheckTimeout;

        while (DateTime.UtcNow < deadline)
        {
            if (_process is { HasExited: true })
            {
                throw new InvalidOperationException($"Backend process exited early with code {_process.ExitCode}.");
            }

            try
            {
                HttpResponseMessage response = await http
                    .GetAsync($"http://127.0.0.1:{Port}/api/v1/health", ct)
                    .ConfigureAwait(false);

                if (response.IsSuccessStatusCode)
                {
                    return;
                }
            }
            catch (HttpRequestException)
            {
                // Backend not ready yet; retry until the deadline.
            }
            catch (TaskCanceledException) when (!ct.IsCancellationRequested)
            {
                // Per-request timeout while the backend is still starting up; retry until the deadline.
            }

            await Task.Delay(HealthCheckInterval, ct).ConfigureAwait(false);
        }

        throw new TimeoutException("Backend did not become healthy in time.");
    }

    // リダイレクトしたパイプを空にし続けるためだけのハンドラ。ログ内容は破棄する。
    static void DiscardProcessOutput(object sender, DataReceivedEventArgs e)
    {
    }

    static int FindFreePort()
    {
        using TcpListener listener = new(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    // LAN内の他端末からアクセスするための案内用に、このマシンのLAN側 IPv4 アドレスを推定する。
    // 複数 NIC がある場合は最初に見つかったものを使う簡易実装。見つからなければループバックを返す。
    static string FindLanIPAddress()
    {
        foreach (IPAddress address in Dns.GetHostAddresses(Dns.GetHostName()))
        {
            if (address.AddressFamily == AddressFamily.InterNetwork && !IPAddress.IsLoopback(address))
            {
                return address.ToString();
            }
        }

        return IPAddress.Loopback.ToString();
    }

    public ValueTask DisposeAsync()
    {
        if (_process is { HasExited: false })
        {
            _process.Kill(entireProcessTree: true);
        }

        _process?.Dispose();
        _jobObject.Dispose();
        return ValueTask.CompletedTask;
    }
}
