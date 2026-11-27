<#
.SYNOPSIS
  RWKV-Pi One-Command Local Deployment & Launcher Setup for Native Windows (PowerShell 5.1 / 7+)

.DESCRIPTION
  Downloads prebuilt rwkv_lightning_cuda Windows x64 binaries, downloads & verifies RWKV-7 7.2B
  model weights and built-in real-world demo projects from Hugging Face (defaulting to hf-mirror.com),
  builds rwkv-pi, and registers `pi`, `rwkv-engine`, and `rwkv-demos` into your Windows User PATH.

.EXAMPLE
  .\quickstart-windows.ps1
  .\quickstart-windows.ps1 -InstallDir "D:\rwkv-agent" -DeepSeekKey "sk-xxxx"
  .\quickstart-windows.ps1 -Precision bf16 -Cuda 13.2
#>
[CmdletBinding()]
param(
  [string]$InstallDir = "$HOME\rwkv-agent",
  [ValidateSet("w8a16", "bf16")]
  [string]$Precision = "w8a16",
  [ValidateSet("12.9", "13.2")]
  [string]$Cuda = "12.9",
  [string]$DeepSeekKey = "",
  [string]$HfEndpoint = "https://hf-mirror.com",
  [int]$Port = 8000,
  [string]$ApiKey = "rwkv-local",
  [string]$PiRepo = "https://github.com/w1c2j3/rwkv-pi.git",
  [switch]$InstallOnly
)

$ErrorActionPreference = "Stop"
$EngineVersion = "v1.8.15"
$HfEndpoint = $HfEndpoint.TrimEnd("/")

function Assert-Command([string]$Name, [string]$Hint) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "[ERROR] Missing required command '$Name'. $Hint"
  }
}

function Download-Resumable([string]$Url, [string]$Dest) {
  $aria2 = Get-Command "aria2c.exe" -ErrorAction SilentlyContinue
  if ($aria2) {
    $dir = Split-Path -Parent $Dest
    $name = Split-Path -Leaf $Dest
    & $aria2.Source -x 16 -s 16 -c --auto-file-renaming=false --allow-overwrite=true -d $dir -o $name $Url
    if ($LASTEXITCODE -ne 0) { throw "aria2c failed downloading $Url" }
    return
  }
  $curlExe = "$env:SystemRoot\System32\curl.exe"
  if (Test-Path $curlExe) {
    & $curlExe -fL --retry 5 --retry-delay 2 -C - -H "User-Agent: curl/8.5" $Url -o $Dest
    if ($LASTEXITCODE -ne 0) { throw "curl.exe failed downloading $Url" }
    return
  }
  Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Dest -Headers @{ "User-Agent" = "curl/8.5" }
}

Write-Host "==> [1/6] Checking Windows prerequisites..." -ForegroundColor Cyan
Assert-Command "git" "Install via PowerShell: winget install --id Git.Git -e"
Assert-Command "node" "Install Node.js >= 22.19 via PowerShell: winget install --id OpenJS.NodeJS.LTS -e"
Assert-Command "npm.cmd" "npm comes bundled with Node.js"

& node -e "const [a,b]=process.versions.node.split('.').map(Number); process.exit(a>22||(a===22&&b>=19)?0:1)"
if ($LASTEXITCODE -ne 0) {
  $nv = (& node -v)
  throw "[ERROR] Node.js >= 22.19.0 is required, found $nv. Upgrade via: winget install --id OpenJS.NodeJS.LTS -e"
}

$InstallDir = [System.IO.Path]::GetFullPath($InstallDir)
foreach ($sub in @("models", "bin", "logs", "data")) {
  New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir $sub) | Out-Null
}

Write-Host "==> [2/6] Preparing rwkv_lightning_cuda ($EngineVersion, Windows x64 CUDA $Cuda)..." -ForegroundColor Cyan
$EngineZip = "rwkv-lightning-$EngineVersion-windows-x64-cuda$Cuda.zip"
$EngineUrl = "https://github.com/Alic-Li/rwkv_lightning_cuda/releases/download/$EngineVersion/$EngineZip"
$EngineExe = Join-Path $InstallDir "bundle\rwkv_lighting_cuda.exe"
$VocabPath = Join-Path $InstallDir "bundle\rwkv_vocab_v20230424.txt"

if ((Test-Path $EngineExe) -and (Test-Path $VocabPath)) {
  Write-Host "    Found existing engine bundle at $InstallDir\bundle, skipping download."
} else {
  $ShaFile = Join-Path $InstallDir "$EngineZip.sha256"
  $ZipPath = Join-Path $InstallDir $EngineZip
  Download-Resumable "$EngineUrl.sha256" $ShaFile
  $ExpectedEngineSha = ((Get-Content $ShaFile -Raw).Trim() -split "\s+")[0].ToLowerInvariant()

  $NeedDownload = $true
  if (Test-Path $ZipPath) {
    $ActualSha = (Get-FileHash -Algorithm SHA256 -Path $ZipPath).Hash.ToLowerInvariant()
    if ($ActualSha -eq $ExpectedEngineSha) { $NeedDownload = $false }
    else { Remove-Item -Force $ZipPath }
  }
  if ($NeedDownload) {
    Download-Resumable $EngineUrl $ZipPath
    $ActualSha = (Get-FileHash -Algorithm SHA256 -Path $ZipPath).Hash.ToLowerInvariant()
    if ($ActualSha -ne $ExpectedEngineSha) {
      throw "[ERROR] Engine archive SHA-256 mismatch (expected $ExpectedEngineSha, got $ActualSha)"
    }
  }
  if (Test-Path (Join-Path $InstallDir "bundle")) {
    Remove-Item -Recurse -Force (Join-Path $InstallDir "bundle")
  }
  Expand-Archive -Path $ZipPath -DestinationPath $InstallDir -Force
}
& $EngineExe --version
if ($LASTEXITCODE -ne 0) { throw "[ERROR] rwkv_lighting_cuda.exe --version failed." }

Write-Host "==> [3/6] Preparing RWKV-7 7.2B model weights ($Precision)..." -ForegroundColor Cyan
$HfRepo = "RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource"
if ($Precision -eq "w8a16") {
  $ModelFile = "RWKV7-Pi-7.2B-v0.1-final.w8a16.rwkvq"
  $DecodeBatch = 4
  $ChunkSize = 512
} else {
  $ModelFile = "RWKV7-Pi-7.2B-v0.1-final.pth"
  $DecodeBatch = 16
  $ChunkSize = 1024
}
$ModelPath = Join-Path $InstallDir "models\$ModelFile"
$SumsPath = Join-Path $InstallDir "models\SHA256SUMS.txt"

Download-Resumable "$HfEndpoint/$HfRepo/resolve/main/SHA256SUMS.txt" $SumsPath
$ExpectedModelSha = $null
foreach ($line in (Get-Content $SumsPath)) {
  $parts = $line.Trim() -split "\s+"
  if ($parts.Count -ge 2) {
    $fn = $parts[1].TrimStart("*")
    if ($fn -eq $ModelFile) { $ExpectedModelSha = $parts[0].ToLowerInvariant(); break }
  }
}
if (-not $ExpectedModelSha) {
  throw "[ERROR] Checksum for $ModelFile not found in SHA256SUMS.txt"
}

$ModelVerified = $false
if (Test-Path $ModelPath) {
  Write-Host "    Checking existing $ModelFile SHA-256..."
  $curSha = (Get-FileHash -Algorithm SHA256 -Path $ModelPath).Hash.ToLowerInvariant()
  if ($curSha -eq $ExpectedModelSha) {
    $ModelVerified = $true
    Write-Host "    Verified existing model ($ExpectedModelSha)."
  }
}

if (-not $ModelVerified) {
  Write-Host "    Downloading $ModelFile from $HfEndpoint/$HfRepo ..."
  Download-Resumable "$HfEndpoint/$HfRepo/resolve/main/$ModelFile" $ModelPath
  if ($Precision -eq "w8a16") {
    try {
      Download-Resumable "$HfEndpoint/$HfRepo/resolve/main/$ModelFile.source.json" "$ModelPath.source.json"
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
Download-Resumable "$HfEndpoint/$HfRepo/resolve/main/rwkv-pi-demos.zip" $DemosZip
if (-not (Test-Path (Join-Path $InstallDir "demos\.templates"))) {
  Expand-Archive -Path $DemosZip -DestinationPath $InstallDir -Force
  Write-Host "    Extracted 7 real-world demo sandboxes to $InstallDir\demos\"
} else {
  Write-Host "    Existing $InstallDir\demos\ preserved (run 'rwkv-demos reset' anytime to restore clean copies)."
}

Write-Host "==> [5/6] Installing and building rwkv-pi client..." -ForegroundColor Cyan
$PiDir = Join-Path $InstallDir "rwkv-pi"
if (-not (Test-Path (Join-Path $PiDir ".git"))) {
  & git clone $PiRepo $PiDir
  if ($LASTEXITCODE -ne 0) { throw "[ERROR] git clone failed for $PiRepo" }
}
Push-Location $PiDir
try {
  & npm.cmd ci --no-audit --no-fund
  if ($LASTEXITCODE -ne 0) { throw "[ERROR] npm ci failed inside $PiDir" }
} finally {
  Pop-Location
}

$EnvPath = Join-Path $PiDir ".env"
if (-not $DeepSeekKey) {
  if ($env:DEEPSEEK_API_KEY) {
    $DeepSeekKey = $env:DEEPSEEK_API_KEY.Trim()
  } elseif (Test-Path $EnvPath) {
    foreach ($ln in (Get-Content $EnvPath)) {
      if ($ln -match "^DEEPSEEK_API_KEY=(.+)$") {
        $DeepSeekKey = $Matches[1].Trim()
        break
      }
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
  "RWKV_EDIT_MODE=text"
  "RWKV_PLUGINS=none"
  "RWKV_SUBAGENT_SANDBOX=none"
  "RWKV_SUBAGENT_MOUNT="
  "RWKV_PI_DATA_ROOT=$DataDirForward"
  ""
  "# ============================================================"
  "# DeepSeek Official API Key (Optional)"
  "# Used by BOTH native web search (web_search / web_fetch) AND /rwkv-subagent collaboration."
  "# Set via: rwkv-engine set-deepseek sk-xxxx   (or paste directly below)"
  "# ============================================================"
  "DEEPSEEK_API_KEY=$DeepSeekKey"
  "RWKV_WEB_API_KEY=$DeepSeekKey"
) -join "`n"
Set-Content -Path $EnvPath -Value ($EnvLines + "`n") -Encoding Ascii

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
  $raw = & $curlExe --noproxy "*" -fsS --max-time 3 -H "Authorization: Bearer $($Conf.apiKey)" "$BaseUrl$Path" 2>$null
  if ($LASTEXITCODE -ne 0 -or -not $raw) { return $null }
  try { return ($raw | ConvertFrom-Json) } catch { return $null }
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
    $Exe = Join-Path $Conf.installDir "bundle\rwkv_lighting_cuda.exe"
    $SessionDb = Join-Path $Conf.installDir "data\rwkv_sessions.db"
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
      -WorkingDirectory (Join-Path $Conf.installDir "bundle") `
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
    & node -e "const fs=require('node:fs'); const [f,k]=process.argv.slice(1); let t=fs.existsSync(f)?fs.readFileSync(f,'utf8'):''; const up=(s,n,v)=>{const r=new RegExp('^'+n+'=.*$','m'); return r.test(s)?s.replace(r,n+'='+v):s.trimEnd()+'\n'+n+'='+v+'\n';}; t=up(t,'DEEPSEEK_API_KEY',k); t=up(t,'RWKV_WEB_API_KEY',k); fs.writeFileSync(f,t,'utf8');" $EnvFile $Arg2
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
  $raw = & $curlExe --noproxy "*" -fsS --max-time 3 -H "Authorization: Bearer $($Conf.apiKey)" "http://$($Conf.host):$($Conf.port)/v1/models" 2>$null
  if ($raw) {
    try {
      $j = $raw | ConvertFrom-Json
      if ($j.data -and $j.data.Count -gt 0 -and $j.data[0].id) {
        $env:RWKV_MODEL = $j.data[0].id
      }
    } catch {}
  }
}

$CliJs = Join-Path $Conf.installDir "rwkv-pi\dist\cli.js"
& node "--env-file=$EnvFile" $CliJs @args
exit $LASTEXITCODE
'@
Set-Content -Path (Join-Path $InstallDir "bin\pi.ps1") -Value $PiPs1 -Encoding Ascii
Set-Content -Path (Join-Path $InstallDir "bin\pi.cmd") -Value "@echo off`r`npowershell -NoProfile -ExecutionPolicy Bypass -File `"%~dp0pi.ps1`" %*`r`n" -Encoding Ascii

$BinDir = Join-Path $InstallDir "bin"
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $UserPath) { $UserPath = "" }
$PathItems = $UserPath -split ";" | Where-Object { $_ -ne "" }
if ($PathItems -notcontains $BinDir) {
  $NewUserPath = (@($BinDir) + $PathItems) -join ";"
  [Environment]::SetEnvironmentVariable("Path", $NewUserPath, "User")
  Write-Host "    Added $BinDir to Windows User PATH."
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
Write-Host "  RWKV-Pi Local Deployment Ready! (Native Windows x64)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Install root : $InstallDir"
Write-Host "  Demos suite  : $InstallDir\demos  (reset via: rwkv-demos reset)"
Write-Host "  Config (.env): $EnvPath"
Write-Host "  Model        : $ModelFile"
Write-Host "  Engine API   : http://127.0.0.1:$Port/v1"
Write-Host ""
Write-Host "  1) Optional: Configure DeepSeek API Key (for web_search & /rwkv-subagent):"
Write-Host "       rwkv-engine set-deepseek sk-your-deepseek-key"
Write-Host ""
Write-Host "  2) Run Pi in built-in real-world demos or ANY project folder:"
Write-Host "       cd `"$InstallDir\demos\01-portal-security`""
Write-Host "       pi"
Write-Host ""
Write-Host "  3) Manage background GPU engine & demos:"
Write-Host "       rwkv-engine status | logs | stop"
Write-Host "       rwkv-demos list | reset"
Write-Host "============================================================" -ForegroundColor Green
