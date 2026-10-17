<#
    一次性把环境搭起来：两个 venv + 三个上游仓库 + 底模。

    用法：
        pwsh -ExecutionPolicy Bypass -File setup.ps1
        pwsh -ExecutionPolicy Bypass -File setup.ps1 -Proxy http://127.0.0.1:7890
        pwsh -ExecutionPolicy Bypass -File setup.ps1 -SkipDeps -SkipModels

    说明：
      * 全部可重入——已存在的目录/文件会跳过，中断后重跑即可。
      * 国内直连 PyPI / GitHub 常常失败，建议挂代理并把 -Proxy 传进来。
      * 下载走 Invoke-WebRequest 并显式传 -Proxy：PowerShell 7 不读 HTTP(S)_PROXY 环境变量。
#>
[CmdletBinding()]
param(
    [string]$Root   = $PSScriptRoot,
    [string]$Proxy  = "",
    [string]$Python = "",
    [switch]$SkipDeps,
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$startTime = Get-Date

# PowerShell 5.1 下 $PSScriptRoot 在 param() 默认值里可能为空，这里兜底
if (-not $Root) { $Root = $PSScriptRoot }
if (-not $Root) { $Root = (Get-Location).Path }
$Root = (Resolve-Path $Root).Path

function Say($msg)  { Write-Host "  $msg" }
function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "  ! $msg" -ForegroundColor Yellow }

# ---------- 0. 找 Python ----------
Step "检查 Python"
if (-not $Python) {
    $cand = Get-Command python -ErrorAction SilentlyContinue
    if ($cand) { $Python = $cand.Source }
    elseif (Test-Path "D:\17551\Documents\Python\py310\python.exe") { $Python = "D:\17551\Documents\Python\py310\python.exe" }
    else { throw "找不到 Python，请用 -Python 指定解释器路径" }
}
$pyVer = & $Python -c "import sys; print('%d.%d' % sys.version_info[:2])"
Say "解释器: $Python  (Python $pyVer)"
if ($pyVer -ne "3.10") { Warn "实测只验证过 3.10，$pyVer 可能遇到依赖轮子缺失" }

$pyArgs = @()
if ($Proxy) { $pyArgs += @("--proxy", $Proxy) }
$pipBase = @("-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off") + $pyArgs
$proxyArg = if ($Proxy) { @("-Proxy", $Proxy) } else { @() }

# ---------- 1. 两个 venv ----------
Step "创建虚拟环境"
foreach ($name in @(".venv", ".venv-rvc")) {
    $p = Join-Path $Root "$name\Scripts\python.exe"
    if (Test-Path $p) { Say "$name 已存在，跳过" }
    else { & $Python -m venv (Join-Path $Root $name); Say "$name 已创建" }
}
$vpy    = Join-Path $Root ".venv\Scripts\python.exe"
$vpyRvc = Join-Path $Root ".venv-rvc\Scripts\python.exe"

# ---------- 2. 上游仓库 ----------
Step "克隆上游仓库"
function Get-Repo($url, $dest, $branch) {
    # 目录已存在就跳过——不管当初是 clone 的还是解压 zip 来的
    if ((Test-Path $dest) -and (Get-ChildItem $dest -Force -ErrorAction SilentlyContinue)) {
        Say "$(Split-Path $dest -Leaf) 已存在，跳过"; return
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
    $gitArgs = @("-c", "core.quotepath=false")
    if ($Proxy) { $gitArgs += @("-c", "http.proxy=$Proxy") }
    $gitArgs += @("clone", "--depth", "1")
    if ($branch) { $gitArgs += @("-b", $branch) }
    $gitArgs += @($url, $dest)
    & git @gitArgs
    if ($LASTEXITCODE -ne 0) { Warn "克隆失败: $url（可用 -Proxy 重试）" } else { Say "$(Split-Path $dest -Leaf) OK" }
}
Get-Repo "https://github.com/yxlllc/DDSP-SVC.git" (Join-Path $Root "tools\DDSP-SVC-6.3") "6.3"
Get-Repo "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git" (Join-Path $Root "rvc") $null

# ---------- 3. 依赖 ----------
if (-not $SkipDeps) {
    Step "安装依赖（两个环境，耗时约 20-40 分钟）"
    $torch = @("torch==2.7.1+cu128", "torchaudio==2.7.1+cu128")
    $idx   = @("--index-url", "https://download.pytorch.org/whl/cu128")

    Say "torch (DDSP 环境)"
    & $vpy @pipBase @torch @idx
    Say "torch (RVC 环境)"
    & $vpyRvc @pipBase @torch @idx

    $req = Join-Path $Root "requirements.txt"
    if (Test-Path $req) { Say "requirements.txt"; & $vpy @pipBase "-r" $req }

    # RVC 的依赖清单在仓库里，名字带版本号，挑存在的那个
    $rvcReq = Get-ChildItem (Join-Path $Root "rvc") -Filter "requirments*.txt" -ErrorAction SilentlyContinue |
              Sort-Object Name | Select-Object -First 1
    if ($rvcReq) {
        Say "RVC 依赖: $($rvcReq.Name)"
        # 文件里写死了 --index-url，会覆盖命令行，先剥掉
        $tmp = Join-Path $env:TEMP "rvc_req.txt"
        Get-Content $rvcReq.FullName | Where-Object { $_ -notmatch '^\s*--index-url' } | Set-Content $tmp -Encoding utf8
        & $vpyRvc @pipBase "-r" $tmp
    } else { Warn "没找到 RVC 的依赖文件" }

    Say "额外补两个 RVC 需要的包"
    & $vpyRvc @pipBase "faiss-cpu" "local-attention" "torchfcpe" "pymss==2.0.14" "pymss-core==0.1.4"
}

# ---------- 4. 底模 ----------
if (-not $SkipModels) {
    Step "下载底模（约 3.5 GB）"
    function Get-File($url, $dest) {
        if (Test-Path $dest) { Say "$(Split-Path $dest -Leaf) 已存在"; return }
        New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
        try {
            Invoke-WebRequest -Uri $url -OutFile $dest -TimeoutSec 3600 @proxyArg
            Say "$(Split-Path $dest -Leaf)  {0:N1} MB" -f ((Get-Item $dest).Length / 1MB)
        } catch { Warn "下载失败: $url" }
    }

    $hf = "https://huggingface.co"
    # DDSP-SVC 需要
    Get-File "$hf/None1145/DDSP-SVC-Base/resolve/main/pretrain/nsf_hifigan/model" `
             (Join-Path $Root "tools\DDSP-SVC-6.3\pretrain\nsf_hifigan\model")
    Get-File "$hf/None1145/DDSP-SVC-Base/resolve/main/pretrain/contentvec/pytorch_model.bin" `
             (Join-Path $Root "tools\DDSP-SVC-6.3\pretrain\contentvec\pytorch_model.bin")
    Get-File "$hf/lengyue233/content-vec-best/resolve/main/config.json" `
             (Join-Path $Root "tools\DDSP-SVC-6.3\pretrain\contentvec\config.json")
    # 音高提取
    Get-File "https://gh-proxy.com/https://github.com/yxlllc/RMVPE/releases/download/98df11c/RMVPE-weights-20230812.zip" `
             (Join-Path $Root "models\rmvpe.zip")

    # RVC 需要
    $rvcAssets = Join-Path $Root "rvc\assets"
    foreach ($pair in @(
        @("hubert_base/config.json",              "hubert_base\config.json"),
        @("hubert_base/preprocessor_config.json", "hubert_base\preprocessor_config.json"),
        @("hubert_base/pytorch_model.bin",        "hubert_base\pytorch_model.bin"),
        @("rmvpe.pt",                             "rmvpe\rmvpe.pt"),
        @("pretrained_v2/f0G48k.pth",             "pretrained_v2\f0G48k.pth"),
        @("pretrained_v2/f0D48k.pth",             "pretrained_v2\f0D48k.pth")
    )) {
        Get-File "$hf/lj1995/VoiceConversionWebUI/resolve/main/$($pair[0])" (Join-Path $rvcAssets $pair[1])
    }

    # 解压 rmvpe
    $rmvpeZip = Join-Path $Root "models\rmvpe.zip"
    if ((Test-Path $rmvpeZip) -and -not (Test-Path (Join-Path $Root "tools\DDSP-SVC-6.3\pretrain\rmvpe\rmvpe.pt"))) {
        Expand-Archive $rmvpeZip -DestinationPath (Join-Path $Root "tools\DDSP-SVC-6.3\pretrain\rmvpe") -Force
        Say "rmvpe 已解压"
    }
}

# ---------- 5. 收尾 ----------
Step "完成"
Say ("用时 {0:N1} 分钟" -f ((Get-Date) - $startTime).TotalMinutes)
Write-Host @"

下一步：
  1. 验证环境
     & "$vpy" -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"
     应输出 2.7.1+cu128 (12, 0)

  2. 看有哪些工具
     & "$vpy" run.py list

  3. 训练前先打补丁（只需要一次）
     & "$vpy" run.py patch --repo tools\DDSP-SVC-6.3

  注意：
    * 缺的人声分离模型（MelBand-Roformer / MDX23C）需自行下载到 models\ 下
    * resemblyzer 依赖的 webrtcvad 没有 Windows 轮子，需要放桩模块，见 README
"@ -ForegroundColor Gray
