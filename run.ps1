<#
.SYNOPSIS
    SDApp (WinUI3フロントエンド) をビルドして起動する。
    Pythonバックエンドはフロントエンドが自動的にサブプロセスとして起動する。
    ビルドは Makefile に委譲し、変更のないターゲット(uv sync / dotnet build)はスキップされる。

.EXAMPLE
    ./run.ps1
#>

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Push-Location $root
try {
    make run
}
finally {
    Pop-Location
}
