@echo off
chcp 65001 >nul
cd /d %~dp0
echo === 1/4 Python и ffmpeg ===
where python >nul 2>nul || winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
where ffmpeg >nul 2>nul || winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
echo Если Python или ffmpeg только что установились - закройте окно и запустите setup.bat ещё раз.
echo === 2/4 Виртуальное окружение ===
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo === 3/4 Библиотеки (PyTorch с поддержкой видеокарты NVIDIA) ===
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
echo === 4/4 Модели (~2.6 ГБ) ===
python download_models.py
python -c "import torch; print('Видеокарта:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'НЕ НАЙДЕНА - будет работать на процессоре')"
echo.
echo Готово. Положите ключ Claude в файл .anthropic_key в этой папке и запустите run.bat
pause
