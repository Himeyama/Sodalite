# Sodalite 配布用インストーラーのビルドスクリプト。
#
# 1. フロントエンドを dotnet publish (Release, win-x64, 自己完結) して staging\app へ
# 2. Python バックエンドのソースを staging\backend へコピー (.venv 等の配布不要物は除外)
# 3. makensis で NSIS インストーラーを dist\ へコンパイル
#
# 前提: .NET 9 SDK, NSIS (makensis) がインストール済みであること。
# 使い方: installer\build-installer.ps1 [-Version 1.0.0]

param(
    [string]$Version = "0.2.0"
)

$ErrorActionPreference = "Stop"

$installerDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $installerDir
$frontendProj = Join-Path $repoRoot "frontend\Sodalite\Sodalite.csproj"
$backendSrc = Join-Path $repoRoot "backend"

$stagingDir = Join-Path $installerDir "staging"
$stagingApp = Join-Path $stagingDir "app"
$stagingBackend = Join-Path $stagingDir "backend"
$distDir = Join-Path $installerDir "dist"

# --- クリーンな staging を用意 ---
if (Test-Path $stagingDir) { Remove-Item -Recurse -Force $stagingDir }
New-Item -ItemType Directory -Force -Path $stagingApp | Out-Null
New-Item -ItemType Directory -Force -Path $distDir | Out-Null

# --- 1. フロントエンドを publish ---
Write-Host "==> dotnet publish (frontend)" -ForegroundColor Cyan
dotnet publish $frontendProj -c Release -r win-x64 --self-contained true -o $stagingApp
if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed with exit code $LASTEXITCODE" }

# --- 2. バックエンドソースをコピー (配布不要物を除外) ---
Write-Host "==> copy backend sources" -ForegroundColor Cyan
# .venv/outputs/キャッシュ/テスト/コンパイル済みバイトコードは配布に含めない。
$excludeDirs = @(".venv", "outputs", "__pycache__", ".ruff_cache", ".pytest_cache", "tests")
$excludeFilePatterns = @("*.pyc", "*.pyo")

New-Item -ItemType Directory -Force -Path $stagingBackend | Out-Null
$srcFull = (Resolve-Path $backendSrc).Path
Get-ChildItem -Path $srcFull -Recurse -Force | ForEach-Object {
    $item = $_
    $relative = $item.FullName.Substring($srcFull.Length).TrimStart('\')
    $segments = $relative -split '\\'

    # 除外ディレクトリを経路に含むものはスキップ
    if ($segments | Where-Object { $excludeDirs -contains $_ }) { return }
    # 除外ファイルパターンはスキップ
    if (-not $item.PSIsContainer -and ($excludeFilePatterns | Where-Object { $item.Name -like $_ })) { return }

    $dest = Join-Path $stagingBackend $relative
    if ($item.PSIsContainer) {
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
    }
    else {
        $destParent = Split-Path -Parent $dest
        if (-not (Test-Path $destParent)) { New-Item -ItemType Directory -Force -Path $destParent | Out-Null }
        Copy-Item -Path $item.FullName -Destination $dest -Force
    }
}

# --- 3. makensis を探して実行 ---
Write-Host "==> compile installer (makensis)" -ForegroundColor Cyan
$makensisCmd = Get-Command makensis -ErrorAction SilentlyContinue
$makensis = if ($makensisCmd) { $makensisCmd.Source } else { $null }
if (-not $makensis) {
    $candidates = @(
        "$env:ProgramFiles\NSIS\makensis.exe",
        "${env:ProgramFiles(x86)}\NSIS\makensis.exe"
    )
    $makensis = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $makensis) {
    throw "makensis が見つかりません。NSIS (https://nsis.sourceforge.io/) をインストールしてから再実行してください。"
}

# .nsi 内の相対パス (staging\, ..\LICENSE, dist\) は makensis のカレントディレクトリ基準で
# 解決されるため、installer\ を作業ディレクトリにして実行する。
Push-Location $installerDir
try {
    & $makensis "/DPRODUCT_VERSION=$Version" "Sodalite.nsi"
    if ($LASTEXITCODE -ne 0) { throw "makensis failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

$output = Join-Path $distDir "Sodalite-Setup-$Version.exe"
Write-Host "==> Done: $output" -ForegroundColor Green
