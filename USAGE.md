# 실시간 한국어 음성 인식 프로그램 사용법

## 설치 방법

### 1. 자동 설치 (권장)

#### Windows
```bash
install.bat
```

#### Linux/Mac
```bash
chmod +x install.sh
./install.sh
```

### 2. 수동 설치

1. **Python 패키지 설치**
```bash
pip install -r requirements.txt
```

2. **Whisper.cpp 다운로드 및 빌드**
```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
make
```

3. **Whisper 모델 다운로드**
```bash
cd whisper.cpp
./models/download-ggml-model.sh base
cd ..
```

## 실행 방법

```bash
python main.py
```

## 프로그램 기능

### 주요 기능
- **실시간 음성 인식**: 마이크에서 실시간으로 음성을 텍스트로 변환
- **한국어 지원**: 한국어 음성 인식에 최적화
- **다양한 모델 지원**: base, small, medium, large 모델 선택 가능
- **오디오 레벨 표시**: 실시간 오디오 입력 레벨 모니터링
- **VAD (Voice Activity Detection)**: 음성 활동 감지로 정확도 향상

### 설정 옵션

#### Whisper 모델
- **base**: 빠르고 가벼운 모델 (기본값)
- **small**: base보다 정확하지만 느림
- **medium**: 더 정확하지만 더 느림
- **large**: 가장 정확하지만 가장 느림

#### 언어 설정
- **ko**: 한국어 (기본값)
- **en**: 영어
- **ja**: 일본어
- **zh**: 중국어

#### 오디오 설정
- **샘플링 레이트**: 8000Hz ~ 48000Hz (기본값: 16000Hz)
- **청크 크기**: 512 ~ 4096 샘플 (기본값: 1024)

## 사용법

### 1. 프로그램 시작
1. `python main.py` 실행
2. 프로그램이 Whisper.cpp와 모델을 확인
3. GUI 창이 열림

### 2. 녹음 시작
1. "녹음 시작" 버튼 클릭
2. 마이크에 말하기
3. 인식된 텍스트가 하단 텍스트 영역에 표시

### 3. 녹음 중지
- "녹음 중지" 버튼 클릭

### 4. 텍스트 관리
- "텍스트 지우기" 버튼으로 결과 텍스트 삭제

## 문제 해결

### 일반적인 문제

#### 1. "Whisper.cpp가 설치되지 않았습니다" 오류
```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
make
```

#### 2. "모델 파일을 찾을 수 없습니다" 오류
```bash
cd whisper.cpp
./models/download-ggml-model.sh base
```

#### 3. 오디오 입력이 감지되지 않음
- 마이크 권한 확인
- 시스템 오디오 설정 확인
- 다른 마이크 장치 시도

#### 4. 음성 인식이 느림
- 더 작은 모델 사용 (base → small)
- 샘플링 레이트 낮추기
- 청크 크기 조정

#### 5. 인식 정확도가 낮음
- 더 큰 모델 사용 (base → medium → large)
- 마이크 품질 개선
- 조용한 환경에서 사용

### 성능 최적화

#### 빠른 응답을 원하는 경우
- 모델: base
- 샘플링 레이트: 16000Hz
- 청크 크기: 512

#### 높은 정확도를 원하는 경우
- 모델: medium 또는 large
- 샘플링 레이트: 16000Hz
- 청크 크기: 2048

## 고급 기능

### VAD (Voice Activity Detection)
프로그램에는 내장된 VAD 기능이 있어 음성 구간만을 감지하여 처리합니다.

### 실시간 처리
- 멀티스레딩을 사용하여 UI 블로킹 방지
- 큐 기반 오디오 처리로 지연 최소화

### 오디오 시각화
- 실시간 오디오 레벨 표시
- 주파수 스펙트럼 분석 (고급 VAD 사용 시)

## 시스템 요구사항

### 최소 요구사항
- Python 3.8+
- 4GB RAM
- 마이크 장치

### 권장 사항
- Python 3.9+
- 8GB RAM
- 고품질 마이크
- SSD 저장장치

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여

버그 리포트나 기능 요청은 GitHub Issues를 통해 제출해주세요. 