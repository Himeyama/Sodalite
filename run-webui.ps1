<#
.SYNOPSIS
    Pythonバックエンドをブラウザ用 webui モードで起動し、LAN内の他デバイスからアクセスできるようにする。
    0.0.0.0 にバインドするため、同一LAN内のスマホ・タブレット・別PCのブラウザから利用できる。

.DESCRIPTION
    WinUI3アプリは不要。バックエンドが API と webui (静的HTML/JS) を同一プロセスで配信する。
    無認証のため、信頼できるネットワークでのみ使用すること。
    Windows ファイアウォールで受信を許可する必要がある場合がある。

.PARAMETER Port
    リッスンするポート。既定は 8188。

.PARAMETER ModelId
    起動時に読み込むモデル。省略時はバックエンド既定。

.EXAMPLE
    ./run-webui.ps1
    ./run-webui.ps1 -Port 9000
#>
param(
    [int]$Port = 8188,
    [string]$ModelId
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$backend = Join-Path $root "backend"

# LAN からアクセスするための主要な IPv4 アドレスを提示する。
$lanIp = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and $_.PrefixOrigin -ne "WellKnown" } |
    Sort-Object -Property InterfaceMetric |
    Select-Object -First 1 -ExpandProperty IPAddress

Write-Host ""
Write-Host "Sodalite webui を起動します..." -ForegroundColor Cyan
Write-Host "  このPC:      http://127.0.0.1:$Port/" -ForegroundColor Green
if ($lanIp) {
    Write-Host "  LAN内から:   http://${lanIp}:$Port/" -ForegroundColor Green
}
Write-Host "  (無認証: 信頼できるLANでのみ使用してください)" -ForegroundColor Yellow
Write-Host ""

Push-Location $backend
try {
    uv sync
    $backendArgs = @("run", "sodalite-backend", "--webui", "--port", $Port)
    if ($ModelId) {
        $backendArgs += @("--model-id", $ModelId)
    }
    & uv @backendArgs
}
finally {
    Pop-Location
}
