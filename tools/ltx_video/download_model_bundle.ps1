param(
    [Parameter(Mandatory = $false)]
    [string]$Root = "models/ltx-2.5",
    [Parameter(Mandatory = $false)]
    [string]$MirrorRepository = "comfyicu/LTX-2.5",
    [Parameter(Mandatory = $false)]
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$rootPath = (Resolve-Path -LiteralPath $Root).Path
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\")).Path
$logRoot = Join-Path $repoRoot "tmp\ltx-model-download"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$aria = (Get-Command aria2c -ErrorAction Stop).Source

$files = @(
    @{ Relative = "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"; Bytes = 18721732720 },
    @{ Relative = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"; Bytes = 26263860594 },
    @{ Relative = "vae/ltx-2.5-video-vae-conv-bf16.safetensors"; Bytes = 1452269922 },
    @{ Relative = "vae/ltx-2.5-audio-vae-bf16.safetensors"; Bytes = 364866540 },
    @{ Relative = "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"; Bytes = 995778752 }
)

function Write-Log([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath (Join-Path $logRoot "bundle-supervisor.log") -Value $line
    Write-Output $line
}

function Get-DownloadProcess([string]$OutputName) {
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -eq "aria2c.exe" -and $_.CommandLine -match [regex]::Escape("--out $OutputName") } |
        Select-Object -First 1
}

function Wait-ForFile([hashtable]$Spec) {
    $relative = $Spec.Relative
    $outputName = Split-Path -Leaf $relative
    $target = Join-Path $rootPath ($relative -replace "/", "\")
    $state = "$target.aria2"
    $stdout = Join-Path $logRoot ((Split-Path -LeafBase $relative) + "-aria2.stdout.log")
    $stderr = Join-Path $logRoot ((Split-Path -LeafBase $relative) + "-aria2.stderr.log")

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
    $existing = Get-Item -LiteralPath $target -ErrorAction SilentlyContinue
    if ($existing -and $existing.Length -eq $Spec.Bytes -and -not (Test-Path -LiteralPath $state)) {
        Write-Log "already complete relative=$relative bytes=$($existing.Length)"
        return
    }

    $url = "https://huggingface.co/$MirrorRepository/resolve/main/$relative`?download=true"
    $arguments = @(
        "--dir", (Split-Path -Parent $target),
        "--out", $outputName,
        "--continue=true",
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=4M",
        "--file-allocation=none",
        "--summary-interval=10",
        "--console-log-level=notice",
        "--allow-overwrite=true",
        $url
    )

    $process = Get-DownloadProcess $outputName
    if (-not $process) {
        Write-Log "starting relative=$relative url=$url"
        $started = Start-Process -FilePath $aria -ArgumentList $arguments -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
        Write-Log "started relative=$relative pid=$($started.Id)"
    } else {
        Write-Log "monitoring existing relative=$relative pid=$($process.ProcessId)"
    }

    $lastSize = -1L
    $stalePolls = 0
    while ($true) {
        Start-Sleep -Seconds $PollSeconds
        $file = Get-Item -LiteralPath $target -ErrorAction SilentlyContinue
        $stateExists = Test-Path -LiteralPath $state
        $process = Get-DownloadProcess $outputName
        $size = if ($file) { [int64]$file.Length } else { 0L }
        if ($size -eq $lastSize) { $stalePolls++ } else { $stalePolls = 0 }
        $lastSize = $size
        Write-Log "status relative=$relative bytes=$size expected=$($Spec.Bytes) state=$stateExists active=$([bool]$process) stalePolls=$stalePolls"

        if (-not $process -and -not $stateExists) {
            if ($size -ne $Spec.Bytes) {
                throw "download exited with an incomplete file relative=$relative bytes=$size expected=$($Spec.Bytes)"
            }
            Write-Log "complete relative=$relative bytes=$size"
            return
        }
    }
}

Write-Log "bundle supervisor started root=$rootPath mirror=$MirrorRepository"
foreach ($file in $files) {
    Wait-ForFile $file
}

$verifier = Join-Path $repoRoot "tools\ltx_video\verify_model_bundle.py"
$reportPath = Join-Path $logRoot "bundle-status.json"
& python $verifier $rootPath | Tee-Object -FilePath $reportPath
if ($LASTEXITCODE -ne 0) {
    throw "model bundle verification failed; see $reportPath"
}
Write-Log "bundle verification succeeded report=$reportPath"
