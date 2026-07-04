using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace SDApp.Services;

/// <summary>
/// Win32 Job Object でプロセスを紐付け、親(このアプリ)が異常終了しても
/// OS が確実に子プロセスツリー(uv/python)を終了させるためのラッパー。
/// フロントエンドが WinRT の FailFast でクラッシュした場合、.NET の例外ハンドラは
/// 経由しないため、通常の Dispose/Kill 呼び出しに頼らない OS レベルの保証が必要になる。
/// </summary>
sealed class JobObject : IDisposable
{
    const uint JobObjectExtendedLimitInformation = 9;
    const uint JobObjectLimitKillOnJobClose = 0x2000;

    readonly SafeFileHandle _handle;

    public JobObject()
    {
        _handle = new SafeFileHandle(CreateJobObjectW(IntPtr.Zero, null), ownsHandle: true);
        if (_handle.IsInvalid)
        {
            throw new InvalidOperationException("Failed to create job object.");
        }

        JobObjectBasicLimitInformation basicInfo = new() { LimitFlags = JobObjectLimitKillOnJobClose };
        JobObjectExtendedLimitInformationStruct info = new() { BasicLimitInformation = basicInfo };

        int length = Marshal.SizeOf<JobObjectExtendedLimitInformationStruct>();
        nint infoPtr = Marshal.AllocHGlobal(length);
        try
        {
            Marshal.StructureToPtr(info, infoPtr, false);

            if (!SetInformationJobObject(_handle, JobObjectExtendedLimitInformation, infoPtr, (uint)length))
            {
                throw new InvalidOperationException("Failed to configure job object.");
            }
        }
        finally
        {
            Marshal.FreeHGlobal(infoPtr);
        }
    }

    public void Assign(SafeHandle processHandle)
    {
        if (!AssignProcessToJobObject(_handle, processHandle))
        {
            throw new InvalidOperationException("Failed to assign process to job object.");
        }
    }

    public void Dispose() => _handle.Dispose();

    [StructLayout(LayoutKind.Sequential)]
    struct JobObjectBasicLimitInformation
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public nint MinimumWorkingSetSize;
        public nint MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public nint Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct IoCounters
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct JobObjectExtendedLimitInformationStruct
    {
        public JobObjectBasicLimitInformation BasicLimitInformation;
        public IoCounters IoInfo;
        public nint ProcessMemoryLimit;
        public nint JobMemoryLimit;
        public nint PeakProcessMemoryUsed;
        public nint PeakJobMemoryUsed;
    }

    // SafeHandle を引数に取る P/Invoke は LibraryImport (ソースジェネレータ) がアンセーフコードを
    // 生成するため、AllowUnsafeBlocks を避けるべくここのみ従来の DllImport を使用する。
    [DllImport("kernel32.dll", EntryPoint = "CreateJobObjectW", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern nint CreateJobObjectW(nint lpJobAttributes, string? lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    static extern bool SetInformationJobObject(
        SafeHandle hJob,
        uint jobObjectInfoClass,
        nint lpJobObjectInfo,
        uint cbJobObjectInfoLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    static extern bool AssignProcessToJobObject(SafeHandle hJob, SafeHandle hProcess);
}
