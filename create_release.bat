@echo off
chcp 65001 >nul
echo Whisper 앱 릴리즈 패키지 생성기
echo ================================================

set RELEASE_DIR=WhisperApp_Release
set VERSION=1.0.0

echo 버전: %VERSION%
echo.

echo 1. 기존 릴리즈 폴더 정리...
if exist %RELEASE_DIR% rmdir /s /q %RELEASE_DIR%
mkdir %RELEASE_DIR%

echo 2. PyInstaller로 실행 파일 빌드...
pyinstaller --onefile --windowed --name=WhisperApp --add-data="whisper.cpp;whisper.cpp" --hidden-import=sounddevice --hidden-import=soundfile --hidden-import=librosa --hidden-import=scipy --hidden-import=torch --hidden-import=torchaudio --hidden-import=sklearn --hidden-import=matplotlib --hidden-import=seaborn --hidden-import=requests main.py

if %errorlevel% neq 0 (
    echo 빌드 실패!
    pause
    exit /b 1
)

echo 3. 릴리즈 패키지 구성...
copy dist\WhisperApp.exe %RELEASE_DIR%\
xcopy whisper.cpp %RELEASE_DIR%\whisper.cpp\ /E /I /Y
copy README.md %RELEASE_DIR%\
copy USAGE.md %RELEASE_DIR%\
copy SPEAKER_EMOTION_GUIDE.md %RELEASE_DIR%\
copy DL_VAD_GUIDE.md %RELEASE_DIR%\

echo 4. 사용자 가이드 생성...
echo # Whisper 음성 인식 앱 v%VERSION% > %RELEASE_DIR%\사용법.txt
echo. >> %RELEASE_DIR%\사용법.txt
echo ## 설치 방법 >> %RELEASE_DIR%\사용법.txt
echo 1. 이 폴더의 모든 파일을 원하는 위치에 압축 해제하세요 >> %RELEASE_DIR%\사용법.txt
echo 2. WhisperApp.exe를 더블클릭하여 실행하세요 >> %RELEASE_DIR%\사용법.txt
echo 3. 처음 실행 시 모델 다운로드가 필요할 수 있습니다 >> %RELEASE_DIR%\사용법.txt
echo. >> %RELEASE_DIR%\사용법.txt
echo ## 시스템 요구사항 >> %RELEASE_DIR%\사용법.txt
echo - Windows 10 이상 >> %RELEASE_DIR%\사용법.txt
echo - 최소 4GB RAM >> %RELEASE_DIR%\사용법.txt
echo - 마이크 장치 >> %RELEASE_DIR%\사용법.txt
echo - 인터넷 연결 (모델 다운로드용) >> %RELEASE_DIR%\사용법.txt
echo. >> %RELEASE_DIR%\사용법.txt
echo ## 주의사항 >> %RELEASE_DIR%\사용법.txt
echo - Windows Defender가 경고할 수 있으나 안전한 파일입니다 >> %RELEASE_DIR%\사용법.txt
echo - 실행 파일과 whisper.cpp 폴더는 함께 있어야 합니다 >> %RELEASE_DIR%\사용법.txt
echo. >> %RELEASE_DIR%\사용법.txt
echo ## 문제 해결 >> %RELEASE_DIR%\사용법.txt
echo - 실행이 안 될 경우: Visual C++ Redistributable 설치 >> %RELEASE_DIR%\사용법.txt
echo - 음성 인식이 안 될 경우: 마이크 권한 확인 >> %RELEASE_DIR%\사용법.txt
echo - 모델 다운로드 실패: 인터넷 연결 확인 >> %RELEASE_DIR%\사용법.txt

echo 5. ZIP 파일 생성...
powershell -Command "Compress-Archive -Path '%RELEASE_DIR%\*' -DestinationPath 'WhisperApp_v%VERSION%.zip' -Force"

echo.
echo 릴리즈 패키지 생성 완료!
echo ================================================
echo 생성된 파일들:
echo - %RELEASE_DIR%\ (폴더)
echo - WhisperApp_v%VERSION%.zip (압축 파일)
echo.
echo 배포 준비가 완료되었습니다.
echo.
pause 