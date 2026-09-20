param(
    [Parameter(Mandatory = $true)][string]$InputVideo,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [int]$MaxFrames = 0,
    [int]$GpuId = 0,
    [int]$RoiSize = 320,
    [int]$MotionContext = 640,
    [int]$DetectEvery = 10,
    [int]$MaxRois = 2,
    [int]$ScanEvery = 3,
    [double]$MotionThreshold = 8.0,
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
        --pipeline zoom --device directml --gpu-id $GpuId --motion-backend opencl --image-size 640 `
        --roi-size $RoiSize --motion-context $MotionContext --zoom-detect-every $DetectEvery `
        --max-rois $MaxRois --scan-every $ScanEvery --motion-threshold $MotionThreshold `
        --confidence 0.10 --lock-confidence 0.10 --threads 4 --max-frames $MaxFrames --target-fps 60 @taskOptions
    $taskExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $taskExitCode
