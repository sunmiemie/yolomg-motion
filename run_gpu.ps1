param(
    [Parameter(Mandatory = $true)][string]$InputVideo,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [int]$MaxFrames = 0,
    [int]$GpuId = 0,
    [ValidateSet('cpu', 'opencl')][string]$TrackingBackend = 'cpu'
)

$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv-gpu/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'GPU environment missing. See GPU_ACCELERATION.md for setup.'
}
$taskInputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($InputVideo)
$taskOutputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
Push-Location -LiteralPath $PSScriptRoot
try {
    & $taskPython -m yolomg_motion.track_cli --input $taskInputPath --output $taskOutputPath `
        --device directml --gpu-id $GpuId --motion-backend opencl --tracking-backend $TrackingBackend `
        --image-size 1280 --confidence 0.10 --lock-confidence 0.10 --detect-every 3 `
        --max-detector-gap 90 --threads 4 --max-frames $MaxFrames
    $taskExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $taskExitCode
