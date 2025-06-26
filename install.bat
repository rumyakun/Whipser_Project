@echo off
echo 실시간 한국어 음성 인식 프로그램 설치
echo ======================================

REM Python 패키지 설치
echo Python 패키지를 설치합니다...
pip install -r requirements.txt

REM Whisper.cpp 클론
if not exist "whisper.cpp" (
    echo Whisper.cpp를 다운로드합니다...
    git clone https://github.com/ggml-org/whisper.cpp.git
)

REM Whisper.cpp 빌드
cd whisper.cpp
echo Whisper.cpp를 빌드합니다...
make

REM 모델 다운로드
echo Whisper 모델을 다운로드합니다...
models\download-ggml-model.cmd base

cd ..

echo 설치가 완료되었습니다!
echo python main.py 명령으로 프로그램을 실행하세요.
pause 