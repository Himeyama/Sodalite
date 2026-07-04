using System.Diagnostics;
using System.Net;
using System.Net.Sockets;

namespace SDApp.Services;

sealed class BackendProcessManager : IAsyncDisposable
{
    static readonly TimeSpan HealthCheckInterval = TimeSpan.FromMilliseconds(500);
    static readonly TimeSpan HealthCheckTimeout = TimeSpan.FromSeconds(120);

    readonly string _backendProjectPath;
    readonly JobObject _jobObject = new();
    Process? _process;

    public int Port { get; private set; }

    public BackendProcessManager(string backendProjectPath) => _backendProjectPath = backendProjectPath;

    public async Task<int> StartAsync(CancellationToken ct = default)
    {
        Port = FindFreePort();

        ProcessStartInfo startInfo = new()
        {
            FileName = "uv",
            Arguments = $"run --project \"{_backendProjectPath}\" sdapp-backend --port {Port}",
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        _process = Process.Start(startInfo) ?? throw new InvalidOperationException("Failed to start backend process.");

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

    static int FindFreePort()
    {
        using TcpListener listener = new(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
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
