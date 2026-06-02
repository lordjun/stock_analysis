$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
$ReportScript = Join-Path $ProjectRoot "daily_sector_report.py"
$ReportDate = Get-Date -Format "yyyyMMdd"
$OutputDir = Join-Path $ProjectRoot "reports\styled"
$ReasonFile = Join-Path $ProjectRoot "inputs\reason_notes_$ReportDate.json"

Set-Location $ProjectRoot
$env:MPLBACKEND = "Agg"
$ReportArgs = @(
    $ReportScript,
    "--date", $ReportDate,
    "--data-source", "ths",
    "--board-source", "both",
    "--output-dir", $OutputDir
)

if (Test-Path -LiteralPath $ReasonFile) {
    $ReportArgs += @("--reason-file", $ReasonFile)
}

& $Python @ReportArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
