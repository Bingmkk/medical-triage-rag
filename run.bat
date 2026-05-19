@echo off
REM 使用已有 conda 环境 pytorch2.2.2 启动服务（不安装依赖）
cd /d "%~dp0"
call conda activate pytorch2.2.2
if errorlevel 1 (
    echo 无法激活 conda 环境 pytorch2.2.2
    exit /b 1
)
python app.py
