$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvRoot = Join-Path $projectRoot ".venv"
$venvScripts = Join-Path $venvRoot "Scripts"

if (-not (Test-Path -LiteralPath $venvScripts)) {
    throw "Virtual environment scripts directory not found: $venvScripts"
}

$env:VIRTUAL_ENV = $venvRoot
$env:PATH = "$venvScripts;$env:PATH"

Write-Output "Activated virtual environment: $env:VIRTUAL_ENV"
