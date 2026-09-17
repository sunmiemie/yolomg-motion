param(
    [Parameter(Mandatory = $true)][string]$InputVideo,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [int]$MaxFrames = 0,
    [int]$GpuId = 0,
    [int]$VerifyEvery = 0,
    [switch]$NoVideo
)
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv-gpu/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'GPU environment missing; see GPU_ACCELERATION.md' }
$taskInputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($InputVideo)
$taskOutputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
$taskOptions = @()
if ($NoVideo) { $taskOptions += '--no-video' }
Push-Location -LiteralPath $PSScriptRoot
try {
    & $taskPython -m yolomg_motion.roi_cli --input $taskInputPath --output $taskOutputPath `
        --device directml --gpu-id $GpuId --motion-backend opencl --image-size 1280 `
        --confidence 0.10 --lock-confidence 0.10 --detect-every 3 --verify-every $VerifyEvery `
        --threads 4 --max-frames $MaxFrames --target-fps 60 @taskOptions
    $taskExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $taskExitCode
