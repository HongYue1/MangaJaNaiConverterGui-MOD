<#
.SYNOPSIS
    Builds the portable JaNai Upscaler folder: a private Python environment,
    the upscaling backend, the models and the JPEG XL tools.

.DESCRIPTION
    Setup is driven by uv. If uv is not on PATH it is downloaded into .\tools
    (about 20 MB), and from then on everything goes through it:

      uv venv   backend\python      the interpreter, a managed CPython
      uv pip    requirements.txt    torch, spandrel, libvips and friends

    Nothing is installed machine-wide and nothing is written outside this
    folder: uv's download cache and its managed interpreters are redirected
    into backend\, so the whole thing stays portable and deletable.

    What a finished folder looks like:

      backend\python        the virtual environment (uv)
      backend\pythons       the managed CPython the venv is built on
      backend\models        model weights
      backend\src           upscaling backend, copied from the repository
      backend\ImageMagick   ICC profiles, copied from the repository
      backend\_cache        uv cache and downloads - safe to delete
      tools\                uv.exe, cjxl.exe, djxl.exe
      janai.config.json     what was resolved, and how
      janai.runtime.txt     interpreter path, for the launcher

.PARAMETER Python
    Python version for the environment. Default 3.13, the newest version with
    wheels for every pinned dependency.

.PARAMETER Torch
    Which torch build to install: auto (ask uv to match the installed driver),
    cpu, cu126, cu128 or cu129.

.PARAMETER Models
    Which model packs to download: all, manga, illustration, none.

.PARAMETER Reuse
    One-off shortcut for a machine that already has MangaJaNaiConverterGui
    installed: junction its python and models folders into backend\ instead of
    creating an environment and downloading several GB. Nothing is installed
    into the shared runtime, so that app keeps working. Everything else about
    the app behaves exactly as it does on a clean install.

.PARAMETER From
    The MangaJaNaiConverterGui install to borrow with -Reuse. Autodetected in
    %APPDATA% and %LOCALAPPDATA% when omitted.

.PARAMETER BackendFrom
    The MangaJaNaiConverterGui\backend folder holding src and ImageMagick.
    Autodetected from the repository this app sits in.

.PARAMETER JxlTools
    Folder holding cjxl.exe and djxl.exe. Autodetected from PATH when omitted.

.PARAMETER NoJxlPlugin
    Skip pillow-jxl-plugin. JXL still works whenever cjxl.exe was found.

.PARAMETER Offline
    Fail instead of reaching the network: for repairing a folder whose cache
    and managed interpreter are already in place.

.PARAMETER Force
    Redo work that looks done: recreate the environment, recopy, re-download.

.EXAMPLE
    .\setup.cmd
    Clean install: uv, Python 3.13, the CUDA torch matching this machine, all
    model packs.

.EXAMPLE
    .\setup.cmd -Torch cpu -Models manga
    No CUDA, grayscale manga models only.

.EXAMPLE
    .\setup.cmd -Reuse
    Borrow the runtime and models of an installed MangaJaNaiConverterGui and
    download nothing.
#>
[CmdletBinding()]
param(
    [string] $Python = '3.13',
    [ValidateSet('auto', 'cpu', 'cu126', 'cu128', 'cu129')]
    [string] $Torch = 'auto',
    [ValidateSet('all', 'manga', 'illustration', 'none')]
    [string] $Models = 'all',
    [switch] $Reuse,
    [string] $From,
    [Alias('ReuseFrom')]
    [string] $BackendFrom,
    [string] $JxlTools,
    [switch] $NoJxlPlugin,
    [switch] $Offline,
    [switch] $Force
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Here = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Backend = Join-Path $Here 'backend'
$PyRoot = Join-Path $Backend 'python'
$ModelsDir = Join-Path $Backend 'models'
$ExtrasDir = Join-Path $Backend 'extras'
$Cache = Join-Path $Backend '_cache'
$ToolsDir = Join-Path $Here 'tools'
$Worker = Join-Path $Here 'worker\worker.py'
$Resolver = Join-Path $Here 'common\paths.py'
$Requirements = Join-Path $Here 'requirements.txt'
$ConfigFile = Join-Path $Here 'janai.config.json'
$RuntimeFile = Join-Path $Here 'janai.runtime.txt'

# Keep uv's state inside the folder: managed interpreters and the wheel cache
# both go under backend\, so nothing is left behind in %LOCALAPPDATA%.
$env:UV_CACHE_DIR = Join-Path $Cache 'uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Backend 'pythons'
$env:UV_NO_CONFIG = '1'
$env:PYTHONUTF8 = '1'

$UvUrl = 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'
$ModelExt = '^\.(pth|safetensors|onnx|pt|ckpt)$'

$ModelPacks = @(
    @{ Name = 'MangaJaNai V1 (grayscale manga)'; Key = 'manga'; Url = 'https://github.com/the-database/mangajanai/releases/download/1.0.0/MangaJaNai_V1_ModelsOnly.zip' },
    @{ Name = 'IllustrationJaNai V3 denoise'; Key = 'illustration'; Url = 'https://github.com/the-database/MangaJaNai/releases/download/3.0.0/IllustrationJaNai_V3denoise.zip' },
    @{ Name = 'IllustrationJaNai V3 detail'; Key = 'illustration'; Url = 'https://github.com/the-database/MangaJaNai/releases/download/3.0.0/IllustrationJaNai_V3detail.zip' }
)

$PyExe = $null
$Uv = $null
$Config = [ordered]@{}

function Step([string] $m) { Write-Host ''; Write-Host "==> $m" -ForegroundColor Cyan }
function Info([string] $m) { Write-Host "    $m" -ForegroundColor Gray }
function Warn([string] $m) { Write-Host "    ! $m" -ForegroundColor Yellow }
function Die([string] $m) { Write-Host ''; Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }

function Full([string] $p) {
    if (-not $p) { return $null }
    return [IO.Path]::GetFullPath((Join-Path (Get-Location).Path $p))
}

function Same([string] $a, [string] $b) {
    if (-not $a -or -not $b) { return $false }
    return ([IO.Path]::GetFullPath($a).TrimEnd('\') -ieq [IO.Path]::GetFullPath($b).TrimEnd('\'))
}

function Find-Interpreter([string] $dir) {
    # A uv venv, a standalone CPython, or the nested layout an installed
    # MangaJaNaiConverterGui exposes through a junction.
    if (-not $dir) { return $null }
    foreach ($rel in @('Scripts\python.exe', 'python.exe', 'python\python.exe')) {
        $cand = Join-Path $dir $rel
        if (Test-Path -LiteralPath $cand -PathType Leaf) { return (Resolve-Path -LiteralPath $cand).Path }
    }
    return $null
}

function Count-Models([string] $dir) {
    if (-not $dir -or -not (Test-Path -LiteralPath $dir)) { return 0 }
    return @(Get-ChildItem -LiteralPath $dir -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -match $ModelExt }).Count
}

function Get-FolderSize([string] $dir) {
    if (-not (Test-Path -LiteralPath $dir)) { return 0 }
    $sum = (Get-ChildItem -LiteralPath $dir -Recurse -File -ErrorAction SilentlyContinue |
        Measure-Object -Property Length -Sum).Sum
    if (-not $sum) { return 0 }
    return [math]::Round($sum / 1GB, 2)
}

# --------------------------------------------------------------------------- #
# links, copies, downloads
# --------------------------------------------------------------------------- #
function Test-Reparse([string] $path) {
    if (-not (Test-Path -LiteralPath $path)) { return $false }
    $item = Get-Item -LiteralPath $path -Force
    return [bool]($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
}

function Get-LinkTarget([string] $path) {
    $item = Get-Item -LiteralPath $path -Force
    $target = @($item.Target)[0]
    if (-not $target) { $target = @($item.LinkTarget)[0] }
    return $target
}

function Remove-Link([string] $path) {
    # rmdir unlinks a junction without following it into the real folder.
    & cmd.exe /c "rmdir `"$path`"" 2>&1 | Out-Null
    return -not (Test-Path -LiteralPath $path)
}

function New-Junction([string] $link, [string] $target) {
    if (-not (Test-Path -LiteralPath $target)) { Warn "link target is missing: $target"; return $false }
    if (Test-Reparse $link) {
        if ((Same (Get-LinkTarget $link) $target) -and -not $Force) {
            Info "$(Split-Path -Leaf $link) -> already linked"
            return $true
        }
        if (-not (Remove-Link $link)) { Warn "could not remove the existing link: $link"; return $false }
    }
    elseif (Test-Path -LiteralPath $link) {
        if (-not $Force) {
            Warn "$(Split-Path -Leaf $link) is a real folder here, leaving it alone"
            return $true
        }
        Remove-Item -LiteralPath $link -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $link) | Out-Null
    try {
        New-Item -ItemType Junction -Path $link -Target $target -ErrorAction Stop | Out-Null
    }
    catch {
        & cmd.exe /c "mklink /J `"$link`" `"$target`"" 2>&1 | Out-Null
    }
    if (Test-Reparse $link) {
        Info "$(Split-Path -Leaf $link) -> $target"
        return $true
    }
    Warn "could not create a junction at $link"
    return $false
}

function Get-Download([string] $url, [string] $dest) {
    $name = ($url -split '/')[-1]
    if ((Test-Path $dest) -and -not $Force) {
        Info "cached: $name"
        return
    }
    if ($Offline) { Die "-Offline was given but $name is not in backend\_cache" }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
    $tmp = "$dest.part"
    if (Test-Path $tmp) { Remove-Item $tmp -Force }
    Info "downloading $name"
    if (Get-Command aria2c.exe -ErrorAction SilentlyContinue) {
        & aria2c.exe -x4 -s4 -k1M --console-log-level=warn --summary-interval=0 --allow-overwrite=true `
            -d (Split-Path -Parent $tmp) -o (Split-Path -Leaf $tmp) $url
    }
    elseif (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        & curl.exe -L --fail --retry 3 --progress-bar -o $tmp $url
    }
    else {
        Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing
    }
    if (-not (Test-Path $tmp) -or (Get-Item $tmp).Length -eq 0) { Die "download failed: $url" }
    Move-Item -Force $tmp $dest
}

function Expand-Zip([string] $zip, [string] $dest) {
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    if (Get-Command 7z.exe -ErrorAction SilentlyContinue) {
        & 7z.exe x -bso0 -bsp0 -y "-o$dest" $zip | Out-Null
        if ($LASTEXITCODE -eq 0) { return }
        Warn '7z failed, falling back to Expand-Archive'
    }
    Expand-Archive -LiteralPath $zip -DestinationPath $dest -Force
}

# --------------------------------------------------------------------------- #
# uv
# --------------------------------------------------------------------------- #
function Get-Uv {
    # Ours first, then whatever the user already has, then the network.
    $mine = Join-Path $ToolsDir 'uv.exe'
    if (Test-Path -LiteralPath $mine) { return $mine }

    $found = Get-Command uv.exe -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
    foreach ($guess in @(
            (Join-Path $env:USERPROFILE '.local\bin\uv.exe'),
            (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links\uv.exe'),
            (Join-Path $env:LOCALAPPDATA 'uv\bin\uv.exe'))) {
        if ($guess -and (Test-Path -LiteralPath $guess)) { return $guess }
    }

    Info 'uv was not found on this machine - fetching it into tools\'
    $zip = Join-Path $Cache 'uv-windows.zip'
    Get-Download $UvUrl $zip
    $tmp = Join-Path $Cache 'uv-unpacked'
    Expand-Zip $zip $tmp
    New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
    foreach ($exe in @('uv.exe', 'uvx.exe')) {
        $src = Get-ChildItem -LiteralPath $tmp -Recurse -File -Filter $exe -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($src) { Copy-Item -LiteralPath $src.FullName -Destination (Join-Path $ToolsDir $exe) -Force }
    }
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    if (-not (Test-Path -LiteralPath $mine)) { Die 'the uv download did not contain uv.exe' }
    return $mine
}

function Invoke-Uv {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]] $UvArgs)
    if ($Offline) { $UvArgs += '--offline' }
    Info "uv $($UvArgs -join ' ')"
    & $Uv @UvArgs
    return ($LASTEXITCODE -eq 0)
}

# --------------------------------------------------------------------------- #
Write-Host ''
Write-Host '  JaNai Upscaler - setup' -ForegroundColor White
Write-Host "  app folder: $Here" -ForegroundColor DarkGray

Step 'Checking the host'
if (-not [Environment]::Is64BitOperatingSystem) { Die 'a 64-bit version of Windows is required' }
foreach ($needed in @($Worker, $Resolver, $Requirements)) {
    if (-not (Test-Path $needed)) { Die "$(Split-Path -Leaf $needed) is missing - the app folder is incomplete" }
}
Info "PowerShell $($PSVersionTable.PSVersion)"
New-Item -ItemType Directory -Force -Path $Backend | Out-Null

# --------------------------------------------------------------------------- #
Step 'Backend source and ICC profiles'
$parent = Split-Path -Parent $Here
$grand = Split-Path -Parent $parent
$srcCandidates = @()
if ($BackendFrom) { $srcCandidates += (Full $BackendFrom) }
# This repository ships src, ImageMagick and resources under .\backend, so look
# there first: a fresh clone then needs nothing copied at all.
$srcCandidates += $Backend
$srcCandidates += (Join-Path $parent 'MangaJaNaiConverterGui\backend')
$srcCandidates += (Join-Path $parent 'backend')
if ($grand) { $srcCandidates += (Join-Path $grand 'MangaJaNaiConverterGui\backend') }
$source = $null
foreach ($c in $srcCandidates) {
    if ($c -and (Test-Path (Join-Path $c 'src\progress_controller.py'))) { $source = (Resolve-Path -LiteralPath $c).Path; break }
}
$ownBackend = if (Test-Path $Backend) { (Resolve-Path -LiteralPath $Backend).Path } else { $Backend }
if ($source -and $source -eq $ownBackend) {
    Info 'already in this repository: src, ImageMagick, resources'
}
elseif ($source) {
    Info "source: $source"
    foreach ($name in @('src', 'ImageMagick', 'resources')) {
        $fromDir = Join-Path $source $name
        $toDir = Join-Path $Backend $name
        if (-not (Test-Path $fromDir)) { Warn "not in the source tree: $name"; continue }
        if ((Test-Path $toDir) -and -not $Force) { Info "$name already here"; continue }
        if (Test-Path $toDir) { Remove-Item $toDir -Recurse -Force }
        Info "copying $name"
        Copy-Item -LiteralPath $fromDir -Destination $toDir -Recurse -Force
    }
}
else {
    Die @"
the upscaling backend was not found.

  Looked for src\progress_controller.py in:
$($srcCandidates | ForEach-Object { "    $_`n" })
  A clone of this repository already contains backend\src. If it is missing,
  restore it, keep this folder inside a MangaJaNaiConverterGui checkout, or
  pass the path:
    .\setup.cmd -BackendFrom "C:\path\to\MangaJaNaiConverterGui\backend"
"@
}

# --------------------------------------------------------------------------- #
$Mode = if ($Reuse -or $From) { 'reuse' } else { 'uv' }
$Borrowed = ''
$OwnEnvironment = ($Mode -eq 'uv')

if ($Mode -eq 'reuse') {
    Step 'Borrowing an installed MangaJaNaiConverterGui'
    $candidates = @()
    if ($From) { $candidates += (Full $From) }
    foreach ($base in @($env:APPDATA, $env:LOCALAPPDATA, (Join-Path $env:USERPROFILE 'AppData\Roaming'))) {
        if ($base) { $candidates += (Join-Path $base 'MangaJaNaiConverterGui') }
    }
    $install = $null
    foreach ($c in $candidates) {
        if ($c -and (Find-Interpreter (Join-Path $c 'python'))) { $install = (Resolve-Path -LiteralPath $c).Path; break }
    }
    if (-not $install) {
        Die @"
no MangaJaNaiConverterGui runtime was found to borrow.

  Looked in: $($candidates -join '; ')

  Pass the install folder:  .\setup.cmd -From "D:\path\to\MangaJaNaiConverterGui"
  or drop -Reuse to build the environment with uv instead.
"@
    }
    $Borrowed = $install
    $srcPython = Join-Path $install 'python'
    $srcModels = Join-Path $install 'models'
    Info "install: $install"
    Info "runtime $(Get-FolderSize $srcPython) GB, models $(Get-FolderSize $srcModels) GB, $(Count-Models $srcModels) weight file(s)"

    $linkedPython = New-Junction $PyRoot $srcPython
    $linkedModels = $false
    if (Test-Path -LiteralPath $srcModels) { $linkedModels = New-Junction $ModelsDir $srcModels }
    else { Warn 'the install has no models folder' }

    if ($linkedPython) {
        $PyExe = Find-Interpreter $PyRoot
    }
    else {
        Warn 'no junction could be made; recording absolute paths in janai.config.json instead'
        $Config['python_dir'] = $srcPython
        $PyExe = Find-Interpreter $srcPython
    }
    if (-not $linkedModels -and (Test-Path -LiteralPath $srcModels)) { $Config['models_dir'] = $srcModels }
    if (-not $PyExe) { Die "no interpreter under $srcPython" }
    Warn 'nothing is installed into the borrowed runtime, so MangaJaNaiConverterGui keeps working'
}
else {
    Step 'uv'
    $Uv = Get-Uv
    $uvVersion = (& $Uv --version 2>&1 | Select-Object -First 1)
    Info "$uvVersion"
    Info $Uv
    Info "cache      $env:UV_CACHE_DIR"
    Info "pythons    $env:UV_PYTHON_INSTALL_DIR"

    Step "Python $Python environment"
    $existing = Find-Interpreter $PyRoot
    if ($existing -and -not $Force) {
        Info 'backend\python already exists (use -Force to rebuild it)'
    }
    else {
        if (Test-Reparse $PyRoot) { Remove-Link $PyRoot | Out-Null }
        # --managed-python keeps the interpreter inside this folder rather than
        # binding the environment to whatever Python happens to be installed.
        if (-not (Invoke-Uv 'venv' $PyRoot '--python' $Python '--managed-python')) {
            Warn 'retrying without --managed-python'
            if (-not (Invoke-Uv 'venv' $PyRoot '--python' $Python)) { Die 'uv venv failed' }
        }
    }
    $PyExe = Find-Interpreter $PyRoot
    if (-not $PyExe) { Die 'uv venv did not produce backend\python\Scripts\python.exe' }

    Step 'Dependencies'
    Info "torch build: $Torch"
    if (-not (Invoke-Uv 'pip' 'install' '--python' $PyExe '--torch-backend' $Torch '-r' $Requirements)) {
        Warn '--torch-backend was rejected; falling back to an explicit PyTorch index'
        $index = if ($Torch -eq 'cpu') { 'https://download.pytorch.org/whl/cpu' }
        elseif ($Torch -eq 'auto') { 'https://download.pytorch.org/whl/cu128' }
        else { "https://download.pytorch.org/whl/$Torch" }
        if (-not (Invoke-Uv 'pip' 'install' '--python' $PyExe '--extra-index-url' $index `
                    '--index-strategy' 'unsafe-best-match' '-r' $Requirements)) {
            Die 'installing the dependencies failed'
        }
    }
}

# --------------------------------------------------------------------------- #
Step 'Interpreter'
& $PyExe -c "import sys, tkinter; print('python', sys.version.split()[0], '- tkinter', tkinter.TkVersion)"
if ($LASTEXITCODE -ne 0) { Die 'the interpreter does not run, or has no Tkinter' }
Info $PyExe

# --------------------------------------------------------------------------- #
Step 'Models'
if ($Mode -eq 'reuse') {
    Info "$(Count-Models $ModelsDir) weight file(s) through the link - nothing to download"
    Info 'the link points into the other install, so no pack is written there'
}
elseif ($Models -eq 'none') {
    Info 'skipped (-Models none)'
    New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
}
else {
    New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
    foreach ($pack in $ModelPacks) {
        if ($Models -ne 'all' -and $Models -ne $pack.Key) { continue }
        Info $pack.Name
        $zip = Join-Path $Cache (($pack.Url -split '/')[-1])
        Get-Download $pack.Url $zip
        $tmp = Join-Path $Cache ('x_' + [IO.Path]::GetFileNameWithoutExtension($zip))
        Expand-Zip $zip $tmp
        $files = @(Get-ChildItem -LiteralPath $tmp -Recurse -File |
            Where-Object { $_.Extension -match $ModelExt })
        $added = 0
        foreach ($f in $files) {
            $dst = Join-Path $ModelsDir $f.Name
            if ((Test-Path $dst) -and -not $Force) { continue }
            Copy-Item -LiteralPath $f.FullName -Destination $dst -Force
            $added = $added + 1
        }
        Info "      $added new, $($files.Count) in pack"
        Remove-Item $tmp -Recurse -Force
    }
    Info "$(Count-Models $ModelsDir) weight file(s) in backend\models"
}

# --------------------------------------------------------------------------- #
Step 'JPEG XL'
$jxlSource = $null
if ($JxlTools) { $jxlSource = Full $JxlTools }
else {
    $found = Get-Command cjxl.exe -ErrorAction SilentlyContinue
    if ($found) { $jxlSource = Split-Path -Parent $found.Source }
}
$haveCjxl = Test-Path -LiteralPath (Join-Path $ToolsDir 'cjxl.exe')
if ($jxlSource -and (Test-Path $jxlSource)) {
    New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
    foreach ($name in @('cjxl.exe', 'djxl.exe')) {
        $fromExe = Join-Path $jxlSource $name
        $toExe = Join-Path $ToolsDir $name
        if (-not (Test-Path $fromExe)) { Warn "not found: $name"; continue }
        if ((Test-Path $toExe) -and -not $Force) { Info "$name already here"; continue }
        Copy-Item -LiteralPath $fromExe -Destination $toExe -Force
        Info "copied $name ($([math]::Round((Get-Item $toExe).Length / 1MB, 1)) MB)"
    }
    $haveCjxl = Test-Path -LiteralPath (Join-Path $ToolsDir 'cjxl.exe')
}
elseif (-not $haveCjxl) {
    Info 'cjxl.exe is not on PATH; relying on the Python side for JXL'
    Info "pass -JxlTools 'C:\path\to\libjxl\bin' to bundle the official encoder"
}

if ($NoJxlPlugin) {
    Info 'pillow-jxl-plugin skipped (-NoJxlPlugin)'
}
elseif ($haveCjxl -and -not $Force) {
    Info 'pillow-jxl-plugin not needed: cjxl.exe can write JXL'
}
else {
    if (-not $Uv) { $Uv = Get-Uv }
    if ($OwnEnvironment) {
        if (Invoke-Uv 'pip' 'install' '--python' $PyExe 'pillow-jxl-plugin') { Info 'installed into backend\python' }
        else { Warn 'pillow-jxl-plugin could not be installed; JXL may be unavailable' }
    }
    else {
        # Borrowed runtime: install beside it, never into it.
        New-Item -ItemType Directory -Force -Path $ExtrasDir | Out-Null
        if (Invoke-Uv 'pip' 'install' '--python' $PyExe '--target' $ExtrasDir '--no-deps' 'pillow-jxl-plugin') {
            Info 'installed into backend\extras'
        }
        else { Warn 'pillow-jxl-plugin could not be installed; JXL may be unavailable' }
    }
}

# --------------------------------------------------------------------------- #
Step 'Configuration'
$Config['mode'] = $Mode
$Config['python'] = (& $PyExe -c "import sys; print(sys.version.split()[0])" 2>&1 | Select-Object -First 1)
if ($Mode -eq 'uv') { $Config['torch_backend'] = $Torch }
if ($Borrowed) { $Config['borrowed_from'] = $Borrowed }
$Config['written'] = (Get-Date).ToString('s')
$json = $Config | ConvertTo-Json -Depth 4
# UTF-8 without a BOM: json.loads in the resolver will not accept one
[System.IO.File]::WriteAllText($ConfigFile, $json, (New-Object System.Text.UTF8Encoding($false)))
Info "wrote $(Split-Path -Leaf $ConfigFile)"

$launch = Join-Path (Split-Path -Parent $PyExe) 'pythonw.exe'
if (-not (Test-Path $launch)) { $launch = $PyExe }
Set-Content -LiteralPath $RuntimeFile -Value $launch -Encoding ASCII
Info "wrote $(Split-Path -Leaf $RuntimeFile) -> $launch"

Step 'What the app resolves'
& $PyExe $Resolver

# --------------------------------------------------------------------------- #
Step 'Self-test'
$raw = & $PyExe $Worker --probe 2>&1
$probeLine = $null
foreach ($line in $raw) {
    $text = [string] $line
    if ($text.TrimStart().StartsWith('{') -and $text -like '*"probe"*') { $probeLine = $text }
}
if (-not $probeLine) {
    Warn 'the worker did not report a probe result:'
    $raw | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    Die 'self-test failed - see the output above'
}
$p = $probeLine | ConvertFrom-Json
foreach ($e in @($p.errors)) { if ($e) { Warn $e } }
if (-not $p.ok) { Die 'the backend could not be imported - see the warnings above' }
Info ("python $($p.python)  torch $($p.torch)" + $(if ($p.cuda) { "  CUDA $($p.cuda)" } else { '  (CPU build)' }))
Info "libvips $($p.libvips) via pyvips $($p.pyvips)"
$deviceNames = @()
foreach ($d in $p.devices) {
    $label = $d.label
    if (-not $label) { $label = $d.value }
    $deviceNames += $label
}
Info ("devices: " + ($deviceNames -join ' | '))
$ok = @()
$bad = @()
foreach ($prop in $p.formats.PSObject.Properties) {
    if ($prop.Value.ok) { $ok += "$($prop.Name.ToUpper()) ($($prop.Value.via))" }
    else { $bad += "$($prop.Name.ToUpper()): $($prop.Value.reason)" }
}
Info ("writes: " + ($ok -join ', '))
foreach ($b in $bad) { Warn "cannot write $b" }
Info ("reads JXL: " + $(if ($p.read_jxl) { 'yes' } else { 'no' }) + "   reads HEIF/AVIF: " + $(if ($p.read_heif) { 'yes' } else { 'no' }))
Info "ICC profiles: $(if ($p.icc) { 'found' } else { 'missing (Lanczos will be used)' })"
Info "models detected by the worker: $(@($p.models).Count)"
if (@($p.models).Count -eq 0) { Warn 'no models were found - upscaling will fall back to plain resizing' }

Write-Host ''
if ($Borrowed) { Write-Host "  Done, with no download: borrowing $Borrowed" -ForegroundColor Green }
else { Write-Host '  Done.' -ForegroundColor Green }
Write-Host '  Start the app with JaNaiUpscaler.cmd' -ForegroundColor Green
if (Test-Path $Cache) { Write-Host '  backend\_cache can be deleted to free disk space.' -ForegroundColor DarkGray }
Write-Host ''
exit 0
