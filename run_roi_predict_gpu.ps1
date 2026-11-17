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
    [int]$PredictionMaxGap = 20,
    [double]$PredictionGate = 16.0,
    [double]$PredictionProcessNoise = 0.25,
    [int]$PredictionMaxRoi = 480,
    [switch]$NoVideo,
    [switch]$StrictVerification,
    [switch]$LegacyDetection,
    [switch]$NoScaleRetry,
    [switch]$FixedRoi,
    [switch]$NoWeakAssociation,
    [double]$LowConfidence = 0.04
)
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv-gpu/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'GPU environment missing; see GPU_ACCELERATION.md' }
$taskInputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($InputVideo)
$taskOutputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
$taskOptions = @()
if ($NoVideo) { $taskOptions += '--no-video' }
if ($StrictVerification) { $taskOptions += '--strict-verification' }
if (-not ($LegacyDetection -or $StrictVerification)) {
    $taskOptions += '--contrast-search'
    if (-not $NoScaleRetry) { $taskOptions += '--scale-retry' }
    if (-not $FixedRoi) { $taskOptions += '--adaptive-roi' }
    if (-not $NoWeakAssociation) { $taskOptions += @('--low-confidence', $LowConfidence.ToString([Globalization.CultureInfo]::InvariantCulture)) }
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $taskPython -m yolomg_motion.roi_cli --input $taskInputPath --output $taskOutputPath `
        --pipeline zoom --motion-prediction --device directml --gpu-id $GpuId --motion-backend opencl --image-size 640 `
        --roi-size $RoiSize --motion-context $MotionContext --zoom-detect-every $DetectEvery `
        --max-rois $MaxRois --scan-every $ScanEvery --motion-threshold $MotionThreshold `
        --prediction-max-gap $PredictionMaxGap --prediction-gate $PredictionGate `
        --prediction-process-noise $PredictionProcessNoise --prediction-max-roi $PredictionMaxRoi `
        --confidence 0.10 --lock-confidence 0.10 --threads 4 --max-frames $MaxFrames --target-fps 60 @taskOptions
    $taskExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $taskExitCode
