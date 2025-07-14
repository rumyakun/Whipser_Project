# Whisper 앱 실행 파일 빌드 가이드

## 개요
이 가이드는 Whisper 음성 인식 애플리케이션을 Windows 실행 파일(.exe)로 빌드하는 방법을 설명합니다.

## 사전 준비

### 1. Python 환경 확인
- Python 3.8 이상이 설치되어 있어야 합니다
- pip가 정상적으로 작동하는지 확인하세요

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. Whisper.cpp 준비
- `whisper.cpp` 폴더가 프로젝트 루트에 있어야 합니다
- 모델 파일들이 `whisper.cpp/models/` 폴더에 있어야 합니다

## 빌드 방법

### 방법 1: 배치 파일 사용 (권장)
1. `build_exe.bat` 파일을 더블클릭하여 실행
2. 자동으로 PyInstaller를 설치하고 빌드를 진행합니다
3. 완료되면 `dist/WhisperApp.exe` 파일이 생성됩니다

### 방법 2: Python 스크립트 사용
```bash
python build_exe.py
```

### 방법 3: 수동 빌드
```bash
# PyInstaller 설치
pip install pyinstaller

# 실행 파일 빌드
pyinstaller --onefile --windowed --name=WhisperApp --add-data="whisper.cpp;whisper.cpp" --hidden-import=sounddevice --hidden-import=soundfile --hidden-import=librosa --hidden-import=scipy --hidden-import=torch --hidden-import=torchaudio --hidden-import=sklearn --hidden-import=matplotlib --hidden-import=seaborn --hidden-import=requests main.py
```

## 빌드 결과

### 생성되는 파일들
- `dist/WhisperApp.exe`: 메인 실행 파일
- `build/`: 빌드 임시 파일들
- `WhisperApp.spec`: PyInstaller 설정 파일

### 파일 크기
- 실행 파일 크기는 약 200-500MB 정도입니다 (의존성 라이브러리 포함)
- 첫 실행 시 추가 파일들이 임시 폴더에 추출됩니다

## 배포 방법

### 1. 단순 배포
사용자에게 다음 파일들을 제공:
- `WhisperApp.exe`
- `whisper.cpp/` 폴더 전체
- `README.md` (사용법 안내)

### 2. 압축 배포
```bash
# 배포용 폴더 생성
mkdir WhisperApp_Release
copy dist\WhisperApp.exe WhisperApp_Release\
xcopy whisper.cpp WhisperApp_Release\whisper.cpp\ /E /I
copy README.md WhisperApp_Release\

# ZIP 파일 생성
powershell Compress-Archive -Path WhisperApp_Release\* -DestinationPath WhisperApp_Release.zip
```

### 3. 설치 프로그램 생성 (고급)
NSIS를 사용하여 설치 프로그램을 만들 수 있습니다:
1. NSIS 설치: https://nsis.sourceforge.io/
2. `build_exe.py` 실행 후 설치 프로그램 생성 옵션 선택
3. `makensis installer.nsi` 명령으로 설치 프로그램 생성

## 사용자 배포 시 주의사항

### 1. 파일 구조
사용자 환경에서 다음과 같은 구조가 필요합니다:
```
WhisperApp/
├── WhisperApp.exe
├── whisper.cpp/
│   ├── main.exe
│   ├── models/
│   │   └── ggml-base.bin
│   └── ...
└── README.md
```

### 2. 모델 파일
- 처음 실행 시 모델 다운로드가 필요할 수 있습니다
- 인터넷 연결이 필요합니다
- 모델 크기는 약 150MB입니다

### 3. 보안 경고
- Windows Defender나 다른 백신이 오탐할 수 있습니다
- 이는 PyInstaller로 패키징된 Python 앱의 일반적인 현상입니다
- 사용자에게 신뢰할 수 있는 소스임을 안내하세요

### 4. 시스템 요구사항
- Windows 10 이상
- 최소 4GB RAM
- 마이크 장치
- 인터넷 연결 (모델 다운로드용)

## 문제 해결

### 빌드 실패 시
1. Python 버전 확인 (3.8 이상)
2. 모든 의존성 패키지 설치 확인
3. `whisper.cpp` 폴더 존재 확인
4. 관리자 권한으로 실행

### 실행 파일 오류 시
1. Visual C++ Redistributable 설치
2. Windows 업데이트 확인
3. 바이러스 백신 예외 설정
4. 임시 폴더 권한 확인

### 성능 최적화
- `--onefile` 대신 `--onedir` 사용 시 더 빠른 실행
- 불필요한 라이브러리 제외로 파일 크기 감소
- UPX 압축으로 파일 크기 추가 감소

## 추가 옵션

### 아이콘 추가
```bash
pyinstaller --onefile --windowed --icon=icon.ico --name=WhisperApp main.py
```

### 콘솔 창 표시 (디버깅용)
```bash
pyinstaller --onefile --name=WhisperApp main.py
```

### 폴더 형태로 빌드
```bash
pyinstaller --onedir --windowed --name=WhisperApp main.py
``` 