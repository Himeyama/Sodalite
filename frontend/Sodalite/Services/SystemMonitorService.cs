using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using Microsoft.Win32;

namespace Sodalite.Services;

/// <summary>CPU・メモリ・GPU・VRAM の使用状況スナップショット。GPU 系は取得できない環境では null。</summary>
readonly record struct SystemStats(
    double CpuUsagePercent,
    double MemoryUsedGiB,
    double MemoryTotalGiB,
    double? GpuUsagePercent,
    double? VramUsedGiB,
    double? VramTotalGiB);

/// <summary>
/// CPU/メモリは Win32 API から直接取得する。NVIDIA の GPU/VRAM は nvidia-smi、Radeon は
/// Windows GPU パフォーマンスカウンターとアダプターレジストリから取得する。
/// </summary>
sealed class SystemMonitorService
{
    const double BytesPerGiB = 1024.0 * 1024.0 * 1024.0;

    (ulong Idle, ulong Kernel, ulong User)? _lastCpuTimes;
    bool _nvidiaSmiUnavailable;
    bool _radeonDetectionAttempted;
    double? _radeonVramTotalGiB;
    List<PerformanceCounter>? _gpuEngineCounters;
    List<PerformanceCounter>? _gpuMemoryCounters;
    DateTime _lastGpuCounterRefreshUtc;

    public async Task<SystemStats> GetStatsAsync(CancellationToken ct = default)
    {
        double cpuUsagePercent = GetCpuUsagePercent();
        (double usedGiB, double totalGiB) = GetMemoryUsage();
        (double? gpuPercent, double? vramUsedGiB, double? vramTotalGiB) =
            await GetGpuStatsAsync(ct).ConfigureAwait(false);

        return new SystemStats(cpuUsagePercent, usedGiB, totalGiB, gpuPercent, vramUsedGiB, vramTotalGiB);
    }

    double GetCpuUsagePercent()
    {
        if (!GetSystemTimes(out FILETIME idleTime, out FILETIME kernelTime, out FILETIME userTime))
        {
            return 0;
        }

        ulong idle = ToUInt64(idleTime);
        ulong kernel = ToUInt64(kernelTime);
        ulong user = ToUInt64(userTime);

        if (_lastCpuTimes is not { } last)
        {
            _lastCpuTimes = (idle, kernel, user);
            return 0;
        }

        _lastCpuTimes = (idle, kernel, user);

        ulong idleDiff = idle - last.Idle;
        ulong totalDiff = (kernel - last.Kernel) + (user - last.User);

        return totalDiff == 0 ? 0 : (1.0 - (double)idleDiff / totalDiff) * 100.0;
    }

    static (double UsedGiB, double TotalGiB) GetMemoryUsage()
    {
        MEMORYSTATUSEX status = new() { dwLength = (uint)Marshal.SizeOf<MEMORYSTATUSEX>() };
        if (!GlobalMemoryStatusEx(ref status))
        {
            return (0, 0);
        }

        double totalGiB = status.ullTotalPhys / BytesPerGiB;
        double usedGiB = (status.ullTotalPhys - status.ullAvailPhys) / BytesPerGiB;
        return (usedGiB, totalGiB);
    }

    async Task<(double? GpuPercent, double? VramUsedGiB, double? VramTotalGiB)> GetGpuStatsAsync(CancellationToken ct)
    {
        if (_nvidiaSmiUnavailable)
        {
            return GetRadeonGpuStats();
        }

        ProcessStartInfo startInfo = new()
        {
            FileName = "nvidia-smi",
            Arguments = "--query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits",
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        try
        {
            using Process process = Process.Start(startInfo)
                ?? throw new InvalidOperationException("Failed to start nvidia-smi.");

            string output = await process.StandardOutput.ReadToEndAsync(ct).ConfigureAwait(false);
            await process.WaitForExitAsync(ct).ConfigureAwait(false);

            if (process.ExitCode != 0)
            {
                _nvidiaSmiUnavailable = true;
                return GetRadeonGpuStats();
            }

            // 複数 GPU 構成では先頭行(1台目)のみを表示対象とする。
            string firstLine = output.AsSpan().TrimStart().ToString().Split('\n')[0];
            string[] parts = firstLine.Split(',');
            if (parts.Length != 3)
            {
                _nvidiaSmiUnavailable = true;
                return GetRadeonGpuStats();
            }

            double gpuPercent = double.Parse(parts[0], CultureInfo.InvariantCulture);
            double vramUsedMiB = double.Parse(parts[1], CultureInfo.InvariantCulture);
            double vramTotalMiB = double.Parse(parts[2], CultureInfo.InvariantCulture);

            const double MiBPerGiB = 1024.0;
            return (gpuPercent, vramUsedMiB / MiBPerGiB, vramTotalMiB / MiBPerGiB);
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or InvalidOperationException)
        {
            // nvidia-smi が存在しない(非 NVIDIA 環境)場合はここに来る。以後は再試行しない。
            _nvidiaSmiUnavailable = true;
            return GetRadeonGpuStats();
        }
    }

    (double? GpuPercent, double? VramUsedGiB, double? VramTotalGiB) GetRadeonGpuStats()
    {
        if (!_radeonDetectionAttempted)
        {
            _radeonDetectionAttempted = true;
            _radeonVramTotalGiB = DetectRadeonVramTotalGiB();
        }

        if (_radeonVramTotalGiB is null)
        {
            return (null, null, null);
        }

        RefreshGpuCountersIfNeeded();

        // Windows exposes one utilization counter per GPU engine. Task Manager's
        // overall GPU figure likewise follows the busiest engine rather than adding
        // parallel engines (which could otherwise exceed 100%).
        double? gpuPercent = ReadMaximumCounter(_gpuEngineCounters);
        double? dedicatedBytes = ReadMaximumRawValue(_gpuMemoryCounters);
        double? usedGiB = dedicatedBytes / BytesPerGiB;
        return (gpuPercent, usedGiB, _radeonVramTotalGiB);
    }

    void RefreshGpuCountersIfNeeded()
    {
        // GPU Engine instances are created per process. The monitor starts before
        // the Python backend, so periodically enumerate again to include it once
        // ROCm/DirectML begins submitting work.
        if (DateTime.UtcNow - _lastGpuCounterRefreshUtc < TimeSpan.FromSeconds(5))
        {
            return;
        }

        _lastGpuCounterRefreshUtc = DateTime.UtcNow;
        DisposeCounters(_gpuEngineCounters);
        DisposeCounters(_gpuMemoryCounters);
        _gpuEngineCounters = CreateCounters("GPU Engine", "Utilization Percentage");
        _gpuMemoryCounters = CreateCounters("GPU Adapter Memory", "Dedicated Usage");
    }

    static void DisposeCounters(List<PerformanceCounter>? counters)
    {
        if (counters is null)
        {
            return;
        }

        foreach (PerformanceCounter counter in counters)
        {
            counter.Dispose();
        }
    }

    static List<PerformanceCounter> CreateCounters(string categoryName, string counterName)
    {
        try
        {
            PerformanceCounterCategory category = new(categoryName);
            return category.GetInstanceNames()
                .Select(instance => new PerformanceCounter(categoryName, counterName, instance, true))
                .ToList();
        }
        catch (Exception ex) when (ex is InvalidOperationException or UnauthorizedAccessException)
        {
            return [];
        }
    }

    static double? ReadMaximumCounter(List<PerformanceCounter>? counters)
    {
        if (counters is not { Count: > 0 })
        {
            return null;
        }

        double maximum = 0;
        bool readAny = false;
        foreach (PerformanceCounter counter in counters)
        {
            try
            {
                maximum = Math.Max(maximum, counter.NextValue());
                readAny = true;
            }
            catch (InvalidOperationException)
            {
                // GPU engine instances disappear when their owning process exits.
            }
        }

        return readAny ? Math.Clamp(maximum, 0, 100) : null;
    }

    static double? ReadMaximumRawValue(List<PerformanceCounter>? counters)
    {
        if (counters is not { Count: > 0 })
        {
            return null;
        }

        long maximum = 0;
        bool readAny = false;
        foreach (PerformanceCounter counter in counters)
        {
            try
            {
                maximum = Math.Max(maximum, counter.RawValue);
                readAny = true;
            }
            catch (InvalidOperationException)
            {
                // Adapter instances can disappear after a display-driver reset.
            }
        }

        return readAny ? maximum : null;
    }

    static double? DetectRadeonVramTotalGiB()
    {
        try
        {
            using RegistryKey? videoKey = Registry.LocalMachine.OpenSubKey(
                @"SYSTEM\CurrentControlSet\Control\Video");
            if (videoKey is null)
            {
                return null;
            }

            ulong largestBytes = 0;
            foreach (string adapterKeyName in videoKey.GetSubKeyNames())
            {
                using RegistryKey? adapterKey = videoKey.OpenSubKey($@"{adapterKeyName}\0000");
                string description = string.Join(
                    " ",
                    adapterKey?.GetValue("DriverDesc") as string ?? string.Empty,
                    adapterKey?.GetValue("ProviderName") as string ?? string.Empty);
                if (!description.Contains("AMD", StringComparison.OrdinalIgnoreCase)
                    && !description.Contains("Radeon", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                object? memoryValue = adapterKey?.GetValue("HardwareInformation.qwMemorySize");
                ulong bytes = memoryValue switch
                {
                    long value when value > 0 => (ulong)value,
                    int value when value > 0 => (uint)value,
                    byte[] value when value.Length >= sizeof(ulong) => BitConverter.ToUInt64(value),
                    _ => 0,
                };
                largestBytes = Math.Max(largestBytes, bytes);
            }

            return largestBytes == 0 ? null : largestBytes / BytesPerGiB;
        }
        catch (Exception ex) when (ex is System.Security.SecurityException
                                   or UnauthorizedAccessException
                                   or IOException)
        {
            return null;
        }
    }

    static ulong ToUInt64(FILETIME fileTime) =>
        ((ulong)(uint)fileTime.dwHighDateTime << 32) | (uint)fileTime.dwLowDateTime;

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool GetSystemTimes(out FILETIME lpIdleTime, out FILETIME lpKernelTime, out FILETIME lpUserTime);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    static extern bool GlobalMemoryStatusEx(ref MEMORYSTATUSEX lpBuffer);

    [StructLayout(LayoutKind.Sequential)]
    struct FILETIME
    {
        public int dwLowDateTime;
        public int dwHighDateTime;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct MEMORYSTATUSEX
    {
        public uint dwLength;
        public uint dwMemoryLoad;
        public ulong ullTotalPhys;
        public ulong ullAvailPhys;
        public ulong ullTotalPageFile;
        public ulong ullAvailPageFile;
        public ulong ullTotalVirtual;
        public ulong ullAvailVirtual;
        public ulong ullAvailExtendedVirtual;
    }
}
