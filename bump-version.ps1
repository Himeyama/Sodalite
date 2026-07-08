<#
.SYNOPSIS
    Sodalite のアプリバージョンを全箇所一括で更新する。

.DESCRIPTION
    以下のファイルのバージョン文字列を書き換え、backend/uv.lock を再生成する。
      - backend/pyproject.toml
      - backend/uv.lock (uv lock --project backend で再生成)
      - frontend/Sodalite/Sodalite.csproj
      - frontend/Sodalite/app.manifest (4桁表記、末尾に .0 を付与)
      - installer/Sodalite.nsi
      - installer/build-installer.ps1

.PARAMETER Version
    新しいバージョン番号 (例: 0.7.0)。x.y.z 形式で指定する。

.EXAMPLE
    ./bump-version.ps1 -Version 0.7.0
#>

param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$Version
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

function Update-VersionInFile {
    param(
        [string]$Path,
        [string]$Pattern,
        [string]$Replacement
    )

    $content = Get-Content -Path $Path -Raw -Encoding UTF8
    $updated = $content -replace $Pattern, $Replacement

    if ($updated -eq $content) {
        throw "バージョン文字列が見つかりませんでした: $Path"
    }

    # BOM 付き UTF-8 を要求するファイル (.nsi) は元のバイト列に BOM が付いているかで判定する。
    $firstBytes = Get-Content -Path $Path -AsByteStream -TotalCount 3
    $hasBom = ($firstBytes[0] -eq 0xEF) -and ($firstBytes[1] -eq 0xBB) -and ($firstBytes[2] -eq 0xBF)
    $encoding = if ($hasBom) { New-Object System.Text.UTF8Encoding($true) } else { New-Object System.Text.UTF8Encoding($false) }
    [System.IO.File]::WriteAllText($Path, $updated, $encoding)

    Write-Output "更新: $Path"
}

Push-Location $root
try {
    Update-VersionInFile `
        -Path "backend/pyproject.toml" `
        -Pattern 'version = "\d+\.\d+\.\d+"' `
        -Replacement "version = ""$Version"""

    Update-VersionInFile `
        -Path "frontend/Sodalite/Sodalite.csproj" `
        -Pattern '<Version>\d+\.\d+\.\d+</Version>' `
        -Replacement "<Version>$Version</Version>"

    Update-VersionInFile `
        -Path "frontend/Sodalite/app.manifest" `
        -Pattern 'version="\d+\.\d+\.\d+\.0"' `
        -Replacement "version=""$Version.0"""

    Update-VersionInFile `
        -Path "installer/Sodalite.nsi" `
        -Pattern '!define PRODUCT_VERSION "\d+\.\d+\.\d+"' `
        -Replacement "!define PRODUCT_VERSION ""$Version"""

    Update-VersionInFile `
        -Path "installer/build-installer.ps1" `
        -Pattern '\[string\]\$Version = "\d+\.\d+\.\d+"' `
        -Replacement "[string]`$Version = ""$Version"""

    Write-Output "uv.lock を再生成しています..."
    uv lock --project backend
    if ($LASTEXITCODE -ne 0) {
        throw "uv lock に失敗しました (exit code $LASTEXITCODE)"
    }

    Write-Output ""
    Write-Output "バージョンを $Version に更新しました。差分を確認してからコミットしてください。"
}
finally {
    Pop-Location
}
