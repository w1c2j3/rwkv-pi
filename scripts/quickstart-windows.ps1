<#
.SYNOPSIS
  RWKV-Pi One-Command Local Deployment & Launcher Setup for Native Windows (PowerShell 5.1 / 7+)

.DESCRIPTION
  Installs missing Node.js / Git / Python via winget, downloads prebuilt rwkv_lightning_cuda
  Windows x64 binaries, downloads & verifies RWKV-7 7.2B v0.2 weights (auto-selected by free GPU memory)
  and the built-in demo projects from Hugging Face (default: hf-mirror.com), builds rwkv-pi,
  and registers `pi`, `rwkv-engine`, and `rwkv-demos` into your Windows User PATH.

.EXAMPLE
  curl.exe -fsSL https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/quickstart-windows.ps1 | Out-String | iex
  $env:RWKV_AGENT_DIR="D:\rwkv-agent"; curl.exe -fsSL https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/quickstart-windows.ps1 | Out-String | iex
#>
[CmdletBinding()]
param(
  [string]$InstallDir = $(if ($env:RWKV_AGENT_DIR) { $env:RWKV_AGENT_DIR } else { "$HOME\rwkv-agent" }),
  [ValidateSet("auto", "w8a16", "bf16")]
  [string]$Precision = $(if ($env:RWKV_PRECISION) { $env:RWKV_PRECISION } else { "auto" }),
  [ValidateSet("12.9", "13.2")]
  [string]$Cuda = $(if ($env:RWKV_CUDA_VER) { $env:RWKV_CUDA_VER } else { "12.9" }),
  [string]$DeepSeekKey = $(if ($env:DEEPSEEK_API_KEY) { $env:DEEPSEEK_API_KEY } else { "" }),
  [string]$HfEndpoint = $(if ($env:HF_ENDPOINT) { $env:HF_ENDPOINT } else { "https://hf-mirror.com" }),
  [int]$Port = $(if ($env:RWKV_PORT) { [int]$env:RWKV_PORT } else { 8000 }),
  [string]$ApiKey = $(if ($env:RWKV_API_KEY) { $env:RWKV_API_KEY } else { "rwkv-local" }),
  [string]$PiRepo = $(if ($env:RWKV_PI_REPO) { $env:RWKV_PI_REPO } else { "https://github.com/w1c2j3/rwkv-pi.git" }),
  [switch]$InstallOnly,
  [switch]$SkipUserPath
)

$ErrorActionPreference = "Stop"
$EngineVersion = "v1.8.15"
$HfEndpoint = $HfEndpoint.TrimEnd("/")
$NpmRegistry = if ($HfEndpoint -like "*huggingface.co*") { "https://registry.npmjs.org" } else { "https://registry.npmmirror.com" }
$env:GIT_TERMINAL_PROMPT = "0"

# Run external native commands without PowerShell 5.1 turning stderr progress/warnings into terminating errors.
function Invoke-Native([scriptblock]$Script) {
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    & $Script | Out-Host
    return $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $prev
  }
}

function Assert-Command([string]$Name, [string]$Hint) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "[ERROR] Missing required command '$Name'. $Hint"
  }
}

function Download-File([string]$Url, [string]$Dest) {
  $curlExe = "$env:SystemRoot\System32\curl.exe"
  if (Test-Path $curlExe) {
    $rc = Invoke-Native { & $curlExe -fsSL --retry 5 --retry-delay 2 -H "User-Agent: curl/8.5" $Url -o $Dest }
    if ($rc -ne 0) { throw "curl.exe failed downloading $Url (exit $rc)" }
    return
  }
  Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Dest -Headers @{ "User-Agent" = "curl/8.5" }
}

function Download-Resumable([string]$Url, [string]$Dest) {
  $aria2 = Get-Command "aria2c.exe" -ErrorAction SilentlyContinue
  if ($aria2) {
    $dir = Split-Path -Parent $Dest
    $name = Split-Path -Leaf $Dest
    $rc = Invoke-Native { & $aria2.Source -x 16 -s 16 -c --auto-file-renaming=false --allow-overwrite=true -d $dir -o $name $Url }
    if ($rc -ne 0) { throw "aria2c failed downloading $Url (exit $rc)" }
    return
  }
  $curlExe = "$env:SystemRoot\System32\curl.exe"
  if (Test-Path $curlExe) {
    $rc = Invoke-Native { & $curlExe -fL --retry 5 --retry-delay 2 -C - -H "User-Agent: curl/8.5" $Url -o $Dest }
    if ($rc -ne 0) {
      if (Test-Path $Dest) { Remove-Item -Force $Dest -ErrorAction SilentlyContinue }
      $rc = Invoke-Native { & $curlExe -fL --retry 5 --retry-delay 2 -H "User-Agent: curl/8.5" $Url -o $Dest }
      if ($rc -ne 0) { throw "curl.exe failed downloading $Url (exit $rc)" }
    }
    return
  }
  Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Dest -Headers @{ "User-Agent" = "curl/8.5" }
}

function Expand-ZipFast([string]$ZipPath, [string]$DestDir) {
  $tarExe = "$env:SystemRoot\System32\tar.exe"
  if (Test-Path $tarExe) {
    $rc = Invoke-Native { & $tarExe -xf $ZipPath -C $DestDir }
    if ($rc -eq 0) { return }
  }
  $oldProg = $ProgressPreference
  $ProgressPreference = "SilentlyContinue"
  try {
    Expand-Archive -Path $ZipPath -DestinationPath $DestDir -Force
  } finally {
    $ProgressPreference = $oldProg
  }
}

function Update-SessionPath {
  $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
  $user = [Environment]::GetEnvironmentVariable("Path", "User")
  $env:Path = "$machine;$user;$env:Path"
}

function Invoke-Winget([string]$Verb, [string]$Id, [string]$Label) {
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw "[ERROR] $Label is missing and winget is unavailable. Install $Label manually, then re-run."
  }
  Write-Host "    $Verb $Label via winget ($Id)..."
  $null = Invoke-Native { & winget $Verb --id $Id -e --silent --accept-source-agreements --accept-package-agreements }
  Update-SessionPath
}

function Test-NodeOk {
  if (-not (Get-Command node -ErrorAction SilentlyContinue)) { return $false }
  $rc = Invoke-Native { & node -e "const [a,b]=process.versions.node.split('.').map(Number); process.exit(a>22||(a===22&&b>=19)?0:1)" }
  return ($rc -eq 0)
}

function Test-PythonOk {
  $py = Get-Command python -ErrorAction SilentlyContinue
  return ($py -and $py.Source -notlike "*\WindowsApps\*")
}

Write-Host "==> [1/6] Checking prerequisites (Node.js, Git, Python)..." -ForegroundColor Cyan
if (-not (Test-NodeOk)) {
  $verb = if (Get-Command node -ErrorAction SilentlyContinue) { "upgrade" } else { "install" }
  Invoke-Winget $verb "OpenJS.NodeJS.LTS" "Node.js LTS"
  if (-not (Test-NodeOk)) { throw "[ERROR] Node.js >= 22.19 is still unavailable. Install it from https://nodejs.org and re-run." }
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  Invoke-Winget "install" "Git.Git" "Git (includes Git Bash used by pi)"
  Assert-Command "git" "Install Git from https://git-scm.com and re-run."
}
if (-not (Test-PythonOk)) {
  try { Invoke-Winget "install" "Python.Python.3.12" "Python 3.12 (for the Python demos)" } catch {}
  if (-not (Test-PythonOk)) { Write-Host "    [WARN] Python not found; Python-based demos will not run." -ForegroundColor Yellow }
}
Assert-Command "npm.cmd" "npm comes bundled with Node.js"
$null = Invoke-Native { & node -v }
Write-Host "    Node.js & Git verified."

# Pick weights by FREE memory on GPU 0 (the engine's device): BF16 needs ~14.2 GB, W8A16 ~8.5 GB.
$VramFree = $null
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
  $prevEap = $ErrorActionPreference
  $ErrorActionPreference = "SilentlyContinue"
  $gpuInfo = (& $smi.Source --id=0 --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
  $ErrorActionPreference = $prevEap
  if ($gpuInfo -and "$gpuInfo" -match "^\s*(\d+)\s*,\s*(\d+)") {
    $VramTotal = [int]$Matches[1]
    $VramFree = [int]$Matches[2]
    Write-Host "    GPU 0 memory: $VramFree MiB free / $VramTotal MiB total"
  }
}
if ($null -eq $VramFree) {
  Write-Host "    [WARN] nvidia-smi not found; make sure an NVIDIA driver with CUDA $Cuda support is installed." -ForegroundColor Yellow
}
if ($Precision -eq "auto") {
  $Precision = if ($null -ne $VramFree -and $VramFree -ge 16000) { "bf16" } else { "w8a16" }
  Write-Host "    Auto-selected precision: $Precision"
}
$NeedMib = if ($Precision -eq "bf16") { 16000 } else { 9000 }
if ($null -ne $VramFree -and $VramFree -lt $NeedMib) {
  Write-Host "    [WARN] $Precision needs ~$($NeedMib / 1000) GB free VRAM; close other GPU apps before running pi." -ForegroundColor Yellow
}

$InstallDir = [System.IO.Path]::GetFullPath($InstallDir)
foreach ($sub in @("models", "bin", "logs", "data")) {
  New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir $sub) | Out-Null
}

$HfRepo = "RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource"
$HfBase = "$HfEndpoint/$HfRepo/resolve/main"
$SumsPath = Join-Path $InstallDir "models\SHA256SUMS.txt"
Download-File "$HfBase/SHA256SUMS.txt" $SumsPath
$Sums = @{}
foreach ($line in (Get-Content $SumsPath)) {
  $parts = $line.Trim() -split "\s+"
  if ($parts.Count -ge 2) { $Sums[$parts[1].TrimStart("*")] = $parts[0].ToLowerInvariant() }
}
function Test-Sha([string]$Path, [string]$Expected) {
  (Test-Path $Path) -and ((Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLowerInvariant() -eq $Expected)
}

Write-Host "==> [2/6] Preparing rwkv_lightning_cuda ($EngineVersion, Windows x64 CUDA $Cuda)..." -ForegroundColor Cyan
$EngineZip = "rwkv-lightning-$EngineVersion-windows-x64-cuda$Cuda.zip"
$EngineExe = Join-Path $InstallDir "bundle\rwkv_lighting_cuda.exe"
$VocabPath = Join-Path $InstallDir "bundle\rwkv_vocab_v20230424.txt"

if ((Test-Path $EngineExe) -and (Test-Path $VocabPath)) {
  Write-Host "    Found existing engine bundle at $InstallDir\bundle, skipping download."
} else {
  $ExpectedEngineSha = $Sums["engine/$EngineZip"]
  if (-not $ExpectedEngineSha) { throw "[ERROR] Checksum for engine/$EngineZip not found in SHA256SUMS.txt" }
  $ZipPath = Join-Path $InstallDir $EngineZip
  if (-not (Test-Sha $ZipPath $ExpectedEngineSha)) {
    Download-Resumable "$HfBase/engine/$EngineZip" $ZipPath
    if (-not (Test-Sha $ZipPath $ExpectedEngineSha)) {
      throw "[ERROR] Engine archive checksum mismatch; delete $ZipPath and re-run."
    }
  }
  if (Test-Path (Join-Path $InstallDir "bundle")) {
    Remove-Item -Recurse -Force (Join-Path $InstallDir "bundle")
  }
  Write-Host "    Extracting engine bundle..."
  Expand-ZipFast $ZipPath $InstallDir
  Remove-Item -Force $ZipPath
}
$rc = Invoke-Native { & $EngineExe --version }
if ($rc -ne 0) { throw "[ERROR] rwkv_lighting_cuda.exe --version failed (exit $rc)." }

Write-Host "==> [3/6] Preparing RWKV-7 7.2B model weights ($Precision)..." -ForegroundColor Cyan
if ($Precision -eq "w8a16") {
  $ModelFile = "RWKV7-Pi-7.2B-v0.2.w8a16.rwkvq"
  $DecodeBatch = 4
  $ChunkSize = 512
} else {
  $ModelFile = "RWKV7-Pi-7.2B-v0.2.pth"
  $DecodeBatch = 16
  $ChunkSize = 1024
}
$ModelPath = Join-Path $InstallDir "models\$ModelFile"
$ExpectedModelSha = $Sums[$ModelFile]
if (-not $ExpectedModelSha) {
  throw "[ERROR] Checksum for $ModelFile not found in SHA256SUMS.txt"
}

$ModelVerified = $false
if (Test-Path $ModelPath) {
  Write-Host "    Checking existing $ModelFile SHA-256..."
  if (Test-Sha $ModelPath $ExpectedModelSha) {
    $ModelVerified = $true
    Write-Host "    Verified existing model ($ExpectedModelSha)."
  }
}

if (-not $ModelVerified) {
  Write-Host "    Downloading $ModelFile from $HfEndpoint/$HfRepo ..."
  Download-Resumable "$HfBase/$ModelFile" $ModelPath
  if ($Precision -eq "w8a16") {
    try {
      Download-File "$HfBase/$ModelFile.source.json" "$ModelPath.source.json"
    } catch {}
  }
  Write-Host "    Verifying SHA-256 checksum (may take 15-30s)..."
  $curSha = (Get-FileHash -Algorithm SHA256 -Path $ModelPath).Hash.ToLowerInvariant()
  if ($curSha -ne $ExpectedModelSha) {
    throw "[ERROR] Checksum mismatch for $ModelPath (expected $ExpectedModelSha, got $curSha). Delete it and re-run."
  }
  Write-Host "    Model checksum verified: $ExpectedModelSha"
}

Write-Host "==> [4/6] Downloading built-in real-world demo benchmark suite (demos\)..." -ForegroundColor Cyan
$DemosZip = Join-Path $InstallDir "rwkv-pi-demos.zip"
Download-File "$HfBase/rwkv-pi-demos.zip" $DemosZip
if (-not (Test-Path (Join-Path $InstallDir "demos\.templates"))) {
  Expand-ZipFast $DemosZip $InstallDir
  Write-Host "    Extracted 7 real-world demo sandboxes to $InstallDir\demos\"
} else {
  Write-Host "    Existing $InstallDir\demos\ preserved (run 'rwkv-demos reset' anytime to restore clean copies)."
}

Write-Host "==> [5/6] Installing and building rwkv-pi client..." -ForegroundColor Cyan
$PiDir = Join-Path $InstallDir "rwkv-pi"
if (-not (Test-Path (Join-Path $PiDir ".git"))) {
  $rc = Invoke-Native { & git clone --depth 1 $PiRepo $PiDir }
  if ($rc -ne 0) { throw "[ERROR] git clone failed for $PiRepo (exit $rc)" }
} else {
  $rc = Invoke-Native { & git -C $PiDir pull --ff-only }
  if ($rc -ne 0) { Write-Host "    [WARN] Could not update rwkv-pi; keeping the existing copy." -ForegroundColor Yellow }
}
Push-Location $PiDir
try {
  $rc = Invoke-Native { & npm.cmd ci --no-audit --no-fund --registry $NpmRegistry }
  if ($rc -ne 0) { throw "[ERROR] npm ci failed inside $PiDir (exit $rc)" }
} finally {
  Pop-Location
}

$EnvPath = Join-Path $PiDir ".env"
if (-not $DeepSeekKey -and (Test-Path $EnvPath)) {
  foreach ($ln in (Get-Content $EnvPath)) {
    if ($ln -match "^DEEPSEEK_API_KEY=(.+)$") {
      $DeepSeekKey = $Matches[1].Trim()
      break
    }
  }
}

$DefaultModelId = $ModelFile -replace '\.(rwkvq|pth)$', ''
$DataDirForward = ($InstallDir -replace '\\', '/') + "/data"

$EngineConfJson = @{
  installDir = $InstallDir
  modelPath = $ModelPath
  vocabPath = $VocabPath
  host = "127.0.0.1"
  port = $Port
  apiKey = $ApiKey
  decodeBatch = $DecodeBatch
  chunkSize = $ChunkSize
} | ConvertTo-Json -Depth 4
Set-Content -Path (Join-Path $InstallDir "engine.json") -Value $EngineConfJson -Encoding Ascii

$EnvLines = @(
  "RWKV_BACKEND=lightning"
  "RWKV_BASE_URL=http://127.0.0.1:$Port/v1"
  "RWKV_MODEL=$DefaultModelId"
  "RWKV_API_KEY=$ApiKey"
  "RWKV_TOOL_FORMAT=xml"
  "RWKV_PROMPT_TEMPLATE=xml"
  "RWKV_CONTEXT_WINDOW=32768"
  "RWKV_MAX_TOKENS=8192"
  "RWKV_HISTORY_MODE=epoch"
  "RWKV_EPOCH_BUDGET_TOKENS=9600"
  "RWKV_EPOCH_KEEP_RECENT_PAIRS=3"
  "RWKV_EPOCH_MAX_RESULT_TOKENS=2000"
  "RWKV_EDIT_MODE=line"
  "RWKV_TEMPERATURE=0.5"
  "RWKV_TOP_P=0.7"
  "RWKV_TOP_K=65536"
  "RWKV_ALPHA_PRESENCE=0.8"
  "RWKV_ALPHA_FREQUENCY=0.2"
  "RWKV_ALPHA_DECAY=0.996"
  "RWKV_PLUGINS=none"
  "RWKV_SUBAGENT_SANDBOX=none"
  "RWKV_SUBAGENT_MOUNT="
  "RWKV_PI_DATA_ROOT=$DataDirForward"
  ""
  "# ============================================================"
  "# DeepSeek Official API Key (Optional)"
  "# Used by BOTH native web search (web_search / web_fetch) AND /rwkv-subagent collaboration."
  "# Set via: rwkv-engine set-deepseek <DEEPSEEK_API_KEY>   (or paste directly below)"
  "# ============================================================"
  "DEEPSEEK_API_KEY=$DeepSeekKey"
  "RWKV_WEB_API_KEY=$DeepSeekKey"
) -join "`n"
Set-Content -Path $EnvPath -Value ($EnvLines + "`n") -Encoding Ascii

# Provide python3 shims (for CMD/PowerShell and Git Bash) so `python3` never hits Windows Store's stub.
Set-Content -Path (Join-Path $InstallDir "bin\python3.cmd") -Value "@echo off`r`npython %*`r`n" -Encoding Ascii
[System.IO.File]::WriteAllBytes((Join-Path $InstallDir "bin\python3"), [System.Text.Encoding]::ASCII.GetBytes("#!/usr/bin/env sh`nexec python `"`$@`"`n"))

$EnginePs1 = @'
param(
  [string]$Action = "status",
  [string]$Arg2 = "",
  [switch]$Quiet
)
$ErrorActionPreference = "Stop"
$ConfPath = Join-Path $PSScriptRoot "..\engine.json"
$Conf = Get-Content $ConfPath -Raw | ConvertFrom-Json
$env:NO_PROXY = "127.0.0.1,localhost"
$env:no_proxy = "127.0.0.1,localhost"

$PidFile = Join-Path $Conf.installDir "logs\engine.pid"
$OutLog = Join-Path $Conf.installDir "logs\engine.out.log"
$ErrLog = Join-Path $Conf.installDir "logs\engine.err.log"
$EnvFile = Join-Path $Conf.installDir "rwkv-pi\.env"
$BaseUrl = "http://$($Conf.host):$($Conf.port)/v1"

function Get-LocalJson([string]$Path) {
  $curlExe = "$env:SystemRoot\System32\curl.exe"
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "SilentlyContinue"
  try {
    $raw = & $curlExe --noproxy "*" -fs --max-time 3 -H "Authorization: Bearer $($Conf.apiKey)" "$BaseUrl$Path" 2>$null
    $rc = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $prev
  }
  if ($rc -ne 0 -or -not $raw) { return $null }
  try { return (($raw -join "`n") | ConvertFrom-Json) } catch { return $null }
}

switch ($Action.ToLowerInvariant()) {
  "start" {
    $models = Get-LocalJson "/models"
    if ($models) {
      $mid = if ($models.data -and $models.data.Count -gt 0) { $models.data[0].id } else { "unknown" }
      if (-not $Quiet) { Write-Host "Engine already running at $BaseUrl (model: $mid)" -ForegroundColor Green }
      exit 0
    }
    if (-not $Quiet) { Write-Host "Starting rwkv_lighting_cuda.exe in background (log: $OutLog)..." -ForegroundColor Cyan }
    $BundleDir = Join-Path $Conf.installDir "bundle"
    $DataDir = Join-Path $Conf.installDir "data"
    New-Item -ItemType Directory -Force -Path $DataDir, (Join-Path $Conf.installDir "logs") | Out-Null
    $env:PATH = "$BundleDir;$env:PATH"
    $Exe = Join-Path $BundleDir "rwkv_lighting_cuda.exe"
    $SessionDb = Join-Path $DataDir "rwkv_sessions.db"
    $ArgsList = @(
      "--model-path", "`"$($Conf.modelPath)`"",
      "--vocab-path", "`"$($Conf.vocabPath)`"",
      "--host", $Conf.host,
      "--port", "$($Conf.port)",
      "--decode-batch-size", "$($Conf.decodeBatch)",
      "--chunk-size", "$($Conf.chunkSize)",
      "--wkv32",
      "--chunk-load",
      "--cmix-sparse", "off",
      "--state-db-path", "`"$SessionDb`"",
      "--password", "`"$($Conf.apiKey)`""
    )
    $Proc = Start-Process -FilePath $Exe -ArgumentList $ArgsList `
      -WorkingDirectory $DataDir `
      -WindowStyle Hidden -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -PassThru
    Set-Content -Path $PidFile -Value "$($Proc.Id)" -Encoding Ascii
    for ($i = 0; $i -lt 90; $i++) {
      $models = Get-LocalJson "/models"
      if ($models) {
        $mid = if ($models.data -and $models.data.Count -gt 0) { $models.data[0].id } else { "unknown" }
        if (-not $Quiet) { Write-Host "Engine ready at $BaseUrl (PID $($Proc.Id), model: $mid)" -ForegroundColor Green }
        exit 0
      }
      if ($Proc.HasExited) {
        Write-Host "[ERROR] Engine exited unexpectedly (code $($Proc.ExitCode)). Recent error log:" -ForegroundColor Red
        if (Test-Path $ErrLog) { Get-Content $ErrLog -Tail 30 }
        if (Test-Path $OutLog) { Get-Content $OutLog -Tail 30 }
        exit 1
      }
      Start-Sleep -Seconds 1
    }
    throw "[ERROR] Timed out waiting for engine after 90s. Check $OutLog and $ErrLog"
  }
  "stop" {
    if (Test-Path $PidFile) {
      $procId = [int](Get-Content $PidFile -Raw).Trim()
      $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
      if ($p) {
        Stop-Process -Id $procId -Force
        Write-Host "Stopped rwkv_lighting_cuda.exe (PID $procId); GPU VRAM released." -ForegroundColor Yellow
      } else {
        Write-Host "Process $procId is not running."
      }
      Remove-Item -Force $PidFile -ErrorAction SilentlyContinue
    } else {
      Write-Host "No PID file found at $PidFile."
    }
  }
  "restart" {
    & $PSCommandPath -Action stop
    & $PSCommandPath -Action start
  }
  "status" {
    $models = Get-LocalJson "/models"
    if ($models) {
      $mid = if ($models.data -and $models.data.Count -gt 0) { $models.data[0].id } else { "unknown" }
      $st = Get-LocalJson "/server/status"
      Write-Host "ONLINE  $BaseUrl  model=$mid" -ForegroundColor Green
      if ($st) { Write-Host ($st | ConvertTo-Json -Compress) }
    } else {
      Write-Host "OFFLINE (run: rwkv-engine start)" -ForegroundColor Yellow
      exit 1
    }
  }
  "logs" {
    if (Test-Path $OutLog) { Get-Content $OutLog -Tail 60 -Wait }
    else { Write-Host "No log file yet at $OutLog" }
  }
  "set-deepseek" {
    if (-not $Arg2) {
      throw "Usage: rwkv-engine set-deepseek <DEEPSEEK_API_KEY>"
    }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
      & node -e "const fs=require('node:fs'); const [f,k]=process.argv.slice(1); let t=fs.existsSync(f)?fs.readFileSync(f,'utf8'):''; const up=(s,n,v)=>{const r=new RegExp('^'+n+'=.*$','m'); return r.test(s)?s.replace(r,n+'='+v):s.trimEnd()+'\n'+n+'='+v+'\n';}; t=up(t,'DEEPSEEK_API_KEY',k); t=up(t,'RWKV_WEB_API_KEY',k); fs.writeFileSync(f,t,'utf8');" $EnvFile $Arg2
    } finally {
      $ErrorActionPreference = $prev
    }
    Write-Host "Saved DEEPSEEK_API_KEY to $EnvFile (web_search, web_fetch, and /rwkv-subagent are now enabled)." -ForegroundColor Green
  }
  default {
    throw "Usage: rwkv-engine {start|stop|restart|status|logs|set-deepseek <KEY>}"
  }
}
'@
Set-Content -Path (Join-Path $InstallDir "bin\rwkv-engine.ps1") -Value $EnginePs1 -Encoding Ascii
Set-Content -Path (Join-Path $InstallDir "bin\rwkv-engine.cmd") -Value "@echo off`r`npowershell -NoProfile -ExecutionPolicy Bypass -File `"%~dp0rwkv-engine.ps1`" %*`r`n" -Encoding Ascii

$DemosPs1 = @'
param(
  [string]$Action = "list",
  [string]$Target = ""
)
$ErrorActionPreference = "Stop"
$ConfPath = Join-Path $PSScriptRoot "..\engine.json"
$Conf = Get-Content $ConfPath -Raw | ConvertFrom-Json
$DemosDir = Join-Path $Conf.installDir "demos"
$TplDir = Join-Path $DemosDir ".templates"

switch ($Action.ToLowerInvariant()) {
  "list" {
    Write-Host "Built-in demo sandboxes in ${DemosDir}:" -ForegroundColor Cyan
    if (Test-Path $TplDir) {
      Get-ChildItem -Directory $TplDir | ForEach-Object {
        Write-Host "  - $(Join-Path $DemosDir $_.Name)"
      }
    }
    Write-Host ""
    Write-Host "Prompts guide: $(Join-Path $DemosDir 'PROMPTS.zh-CN.md')"
    Write-Host "Reset anytime: rwkv-demos reset"
  }
  "reset" {
    if (-not (Test-Path $TplDir)) {
      throw "[ERROR] Template directory missing at $TplDir"
    }
    Get-ChildItem -Directory $TplDir | ForEach-Object {
      if (-not $Target -or $_.Name -eq $Target) {
        $dst = Join-Path $DemosDir $_.Name
        if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
        Copy-Item -Recurse -Force $_.FullName $dst
        Write-Host "  [OK] Reset $dst" -ForegroundColor Green
      }
    }
  }
  default {
    throw "Usage: rwkv-demos {list|reset [demo-name]}"
  }
}
'@
Set-Content -Path (Join-Path $InstallDir "bin\rwkv-demos.ps1") -Value $DemosPs1 -Encoding Ascii
Set-Content -Path (Join-Path $InstallDir "bin\rwkv-demos.cmd") -Value "@echo off`r`npowershell -NoProfile -ExecutionPolicy Bypass -File `"%~dp0rwkv-demos.ps1`" %*`r`n" -Encoding Ascii

$PiPs1 = @'
$ErrorActionPreference = "Stop"
$ConfPath = Join-Path $PSScriptRoot "..\engine.json"
$Conf = Get-Content $ConfPath -Raw | ConvertFrom-Json
$env:NO_PROXY = "127.0.0.1,localhost"
$env:no_proxy = "127.0.0.1,localhost"
$EnvFile = Join-Path $Conf.installDir "rwkv-pi\.env"

# Ensure our bin\ (python3 shim) and Git for Windows's bin\bash.exe (even on non-C: drives) precede System32\bash.exe (WSL) and WindowsApps.
$ExtraPaths = @($PSScriptRoot)
$gitCmd = Get-Command git.exe -ErrorAction SilentlyContinue
if ($gitCmd) {
  $gitRoot = Split-Path -Parent (Split-Path -Parent $gitCmd.Source)
  foreach ($sub in @("bin", "usr\bin")) {
    $cand = Join-Path $gitRoot $sub
    if (Test-Path $cand) { $ExtraPaths += $cand }
  }
}
$env:PATH = ($ExtraPaths -join ";") + ";" + $env:PATH

# Load optional per-directory .rwkv-pi.env (e.g. demos\04-python-web-research)
if (Test-Path ".\.rwkv-pi.env") {
  foreach ($ln in (Get-Content ".\.rwkv-pi.env")) {
    $trimmed = $ln.Trim()
    if (-not $trimmed -or $trimmed.StartsWith("#")) { continue }
    $idx = $trimmed.IndexOf("=")
    if ($idx -gt 0) {
      $k = $trimmed.Substring(0, $idx).Trim()
      $v = $trimmed.Substring($idx + 1).Trim()
      if (-not [Environment]::GetEnvironmentVariable($k, "Process")) {
        [Environment]::SetEnvironmentVariable($k, $v, "Process")
      }
    }
  }
}

if (-not $env:DEEPSEEK_API_KEY -and (Test-Path $EnvFile)) {
  foreach ($ln in (Get-Content $EnvFile)) {
    if ($ln -match "^DEEPSEEK_API_KEY=(.+)$") {
      $env:DEEPSEEK_API_KEY = $Matches[1].Trim()
      break
    }
  }
}

if ($env:DEEPSEEK_API_KEY) {
  if (-not $env:RWKV_WEB_API_KEY) { $env:RWKV_WEB_API_KEY = $env:DEEPSEEK_API_KEY }
  if (-not $env:RWKV_TOOLS) { $env:RWKV_TOOLS = "read,write,edit,bash,grep,find,ls,web_search,web_fetch" }
} else {
  if (-not $env:RWKV_TOOLS) { $env:RWKV_TOOLS = "read,write,edit,bash,grep,find,ls" }
}

if (-not $env:RWKV_SUBAGENT_SANDBOX) { $env:RWKV_SUBAGENT_SANDBOX = "none" }

$SkipEngine = $false
foreach ($a in $args) {
  if ($a -in @("-h", "--help", "-v", "--version", "login", "logout", "status", "token")) {
    $SkipEngine = $true
  }
}

if (-not $SkipEngine) {
  & (Join-Path $PSScriptRoot "rwkv-engine.ps1") -Action start -Quiet
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  $curlExe = "$env:SystemRoot\System32\curl.exe"
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "SilentlyContinue"
  try {
    $raw = & $curlExe --noproxy "*" -fs --max-time 3 -H "Authorization: Bearer $($Conf.apiKey)" "http://$($Conf.host):$($Conf.port)/v1/models" 2>$null
  } finally {
    $ErrorActionPreference = $prev
  }
  if ($raw) {
    try {
      $j = ($raw -join "`n") | ConvertFrom-Json
      if ($j.data -and $j.data.Count -gt 0 -and $j.data[0].id) {
        $env:RWKV_MODEL = $j.data[0].id
      }
    } catch {}
  }
}

$CliJs = Join-Path $Conf.installDir "rwkv-pi\dist\cli.js"
$prev = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
  & node "--env-file=$EnvFile" $CliJs @args
  $rc = $LASTEXITCODE
} finally {
  $ErrorActionPreference = $prev
}
exit $rc
'@
Set-Content -Path (Join-Path $InstallDir "bin\pi.ps1") -Value $PiPs1 -Encoding Ascii
Set-Content -Path (Join-Path $InstallDir "bin\pi.cmd") -Value "@echo off`r`npowershell -NoProfile -ExecutionPolicy Bypass -File `"%~dp0pi.ps1`" %*`r`n" -Encoding Ascii

$BinDir = Join-Path $InstallDir "bin"
if (-not $SkipUserPath) {
  $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
  if (-not $UserPath) { $UserPath = "" }
  $PathItems = $UserPath -split ";" | Where-Object { $_ -ne "" }
  if ($PathItems -notcontains $BinDir) {
    $NewUserPath = (@($BinDir) + $PathItems) -join ";"
    [Environment]::SetEnvironmentVariable("Path", $NewUserPath, "User")
    Write-Host "    Added $BinDir to Windows User PATH."
  }
}
if (($env:PATH -split ";") -notcontains $BinDir) {
  $env:PATH = "$BinDir;$env:PATH"
}

Write-Host "==> [6/6] Finalizing setup..." -ForegroundColor Cyan
if (-not $InstallOnly) {
  & (Join-Path $BinDir "rwkv-engine.ps1") -Action start
} else {
  Write-Host "    Skipped engine startup (-InstallOnly)."
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  RWKV-Pi is ready (Windows x64)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Install root : $InstallDir"
Write-Host "  Model        : $ModelFile"
Write-Host "  Config (.env): $EnvPath"
Write-Host "  Engine API   : http://127.0.0.1:$Port/v1"
Write-Host ""
Write-Host "  Next (works in this window and any new one):"
Write-Host "       cd `"$InstallDir\demos\06-textindex`"; pi"
Write-Host ""
Write-Host "  Optional DeepSeek key (web_search + /rwkv-subagent):"
Write-Host "       rwkv-engine set-deepseek <DEEPSEEK_API_KEY>"
Write-Host "  Manage: rwkv-engine status|logs|stop   rwkv-demos list|reset"
Write-Host "============================================================" -ForegroundColor Green
