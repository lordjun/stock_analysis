$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TaskName = "AShareSectorLeaderDailyReport"
$ScriptPath = Join-Path $ProjectRoot "scripts\run_daily_report.ps1"

$Argument = '-NoProfile -ExecutionPolicy Bypass -File "' + $ScriptPath + '"'
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $Argument
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 16:30
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Generate daily A-share top sector and leader stock PPT report" -Force
Write-Host "Registered scheduled task: $TaskName"
