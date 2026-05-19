# 使用已有 conda 环境 pytorch2.2.2 启动服务（不安装依赖）
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "未找到 conda。请先在终端执行: conda activate pytorch2.2.2"
    exit 1
}

(& conda "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate pytorch2.2.2
python app.py
exit $LASTEXITCODE
