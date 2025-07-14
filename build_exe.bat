 @echo off
chcp 65001 >nul
echo Whisper 앱 실행 파일 빌더
echo ================================================

echo PyInstaller를 설치합니다...
pip install pyinstaller

if %errorlevel% neq 0 (
    echo PyInstaller 설치 실패!
    pause
    exit /b 1
)

echo.
echo 실행 파일을 빌드합니다...
pyinstaller --onefile --windowed --name=WhisperApp --add-data="whisper.cpp;whisper.cpp" --hidden-import=sounddevice --hidden-import=soundfile --hidden-import=librosa --hidden-import=scipy --hidden-import=torch --hidden-import=torchaudio --hidden-import=sklearn --hidden-import=matplotlib --hidden-import=seaborn --hidden-import=requests main.py

if %errorlevel% neq 0 (
    echo 빌드 실패!
    pause
    exit /b 1
)

echo.
echo 빌드 완료!
echo ================================================
echo 생성된 파일: dist\WhisperApp.exe
echo.
echo 사용자 배포 시 주의사항:
echo 1. whisper.cpp 폴더와 모델 파일들이 실행 파일과 함께 있어야 합니다.
echo 2. 처음 실행 시 모델 다운로드가 필요할 수 있습니다.
echo 3. Windows Defender나 다른 백신이 오탐할 수 있습니다.
echo.
pause 