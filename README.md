# 실시간 한국어 음성 인식 프로그램

Whisper.cpp를 기반으로 한 실시간 한국어 음성 인식 프로그램입니다.

## 설치 방법

1. Whisper.cpp 빌드:
```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
make
```

2. Python 패키지 설치:
```bash
pip install -r requirements.txt
```

3. Whisper 모델 다운로드:
```bash
# 한국어 모델 다운로드
cd whisper.cpp
./models/download-ggml-model.sh base
```

## 실행 방법

```bash
python main.py
```

## 기능

- 실시간 음성 인식
- 한국어 지원
- PySide6 기반 GUI
- 음성 활동 감지 (VAD) 