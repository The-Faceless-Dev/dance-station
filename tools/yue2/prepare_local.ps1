[CmdletBinding()]
param(
    [string]$ModelDir = "D:\models\yue2-3b-gguf",
    [string]$AudioCppDir = "D:\audio.cpp",
    [string]$AudioCppRef = "6fbbee4efd1c2251d3774edd89a002a0039da5f7",
    [string]$VsInstall = "",
    [switch]$SkipModelDownload,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

function Invoke-Checked([string]$File, [string[]]$Arguments) {
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$File exited with code $LASTEXITCODE"
    }
}

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

if (-not $SkipModelDownload) {
    if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
        throw "The Hugging Face CLI 'hf' is required to download the public YuE2 package."
    }
    $files = @(
        "yue2-3b-q8_0.gguf",
        "yue2-vae-f16.gguf",
        "sidecars/yue2-model-config.json",
        "sidecars/yue2-generation-config.json",
        "sidecars/yue2-qwen.tiktoken",
        "sidecars/yue2-vae-config.json"
    )
    Invoke-Checked "hf" (@("download", "audio-cpp/Yue2-3B-GGUF") + $files + @("--local-dir", $ModelDir))
}

$required = @(
    (Join-Path $ModelDir "yue2-3b-q8_0.gguf"),
    (Join-Path $ModelDir "yue2-vae-f16.gguf"),
    (Join-Path $ModelDir "sidecars/yue2-model-config.json"),
    (Join-Path $ModelDir "sidecars/yue2-generation-config.json"),
    (Join-Path $ModelDir "sidecars/yue2-qwen.tiktoken"),
    (Join-Path $ModelDir "sidecars/yue2-vae-config.json")
)
$missing = $required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }
if ($missing) { throw "YuE2 package is incomplete: $($missing -join ', ')" }

if (-not $SkipBuild) {
    if (-not (Test-Path -LiteralPath $AudioCppDir)) {
        Invoke-Checked "git" @("clone", "--branch", "dev", "https://github.com/0xShug0/audio.cpp.git", $AudioCppDir)
    }
    Invoke-Checked "git" @("-C", $AudioCppDir, "fetch", "--depth", "1", "origin", $AudioCppRef)
    Invoke-Checked "git" @("-C", $AudioCppDir, "checkout", "--detach", $AudioCppRef)
    $buildScript = Join-Path $AudioCppDir "scripts/build_windows.ps1"
    if (-not (Test-Path -LiteralPath $buildScript -PathType Leaf)) { throw "audio.cpp Windows build script not found: $buildScript" }
    $pythonScripts = (& python -c "import sysconfig; print(sysconfig.get_path('scripts'))").Trim()
    if ($pythonScripts -and (Test-Path (Join-Path $pythonScripts "cmake.exe"))) {
        $env:PATH = "$pythonScripts;$env:PATH"
    }
    $bundledNinja = Get-ChildItem "$VsInstall\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($bundledNinja) {
        $env:PATH = "$(Split-Path $bundledNinja.FullName -Parent);$env:PATH"
    }
    $buildArgs = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $buildScript,
        "-Preset", "windows-cuda-release", "-ModelSet", "custom", "-Models", "yue2",
        "-Target", "audiocpp_cli", "-Jobs", "8"
    )
    if ($VsInstall) { $buildArgs += @("-VsInstall", $VsInstall) }
    Invoke-Checked "powershell.exe" $buildArgs
}

$candidates = @(
    (Join-Path $AudioCppDir "build/windows-cuda-release/bin/audiocpp_cli.exe"),
    (Join-Path $AudioCppDir "build/windows-cuda-release/bin/audiocpp_cli")
)
$cli = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $cli) { throw "YuE2 audio.cpp CLI was not found after setup." }

Write-Output "YUe2 model directory: $ModelDir"
Write-Output "audio.cpp CLI: $cli"
Write-Output "Set these local variables before starting the worker:"
Write-Output "`$env:YUE2_MODEL_ROOT='$ModelDir'"
Write-Output "`$env:YUE2_AUDIOCPP_CLI='$cli'"
