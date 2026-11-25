# Build nativo de Cabrita para Windows.
# Uso: powershell -ExecutionPolicy Bypass -File scripts\package_windows.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location "$Root\rust"
try {
    cargo build --release
    New-Item -ItemType Directory -Force -Path "$Root\dist" | Out-Null
    Copy-Item "target\release\cabrita.exe" "$Root\dist\cabrita.exe" -Force
    Write-Host "Windows: $Root\dist\cabrita.exe"
} finally {
    Pop-Location
}
