# 🎤 화자 구분 & 감정 분석 종합 가이드

## 📖 개요

화자 구분(Speaker Diarization)과 감정 분석(Emotion Recognition)은 각각 다른 전용 모델을 사용하는 고급 음성 처리 기술입니다.

## 🎯 화자 구분 (Speaker Diarization)

### 🔧 작동 원리

1. **화자 임베딩 생성**
   - LSTM + 어텐션 기반 신경망
   - MFCC 특성에서 화자별 고유 벡터 추출
   - 256차원 임베딩 공간에서 화자 구분

2. **클러스터링**
   - K-means 또는 Agglomerative Clustering
   - 화자 수 자동 추정 (Silhouette Score 기반)
   - 실시간 화자 변경 감지

### 🏗️ 모델 구조

```python
SpeakerEncoder:
├── LSTM (3 layers, bidirectional)
├── Multi-head Attention (8 heads)
├── Embedding Layer (256 dimensions)
└── Layer Normalization
```

### 📊 주요 기능

- **실시간 화자 변경 감지**
- **화자 수 자동 추정**
- **화자별 신뢰도 계산**
- **화자 통계 분석**

### 🎛️ 주요 매개변수

```python
# 화자 구분 설정
speaker_diarization = SpeakerDiarization(
    model_path='speaker_model.pth',
    sample_rate=16000,
    frame_duration=0.025,      # 25ms 프레임
    embedding_size=256,        # 임베딩 크기
    device='cpu'
)

# 화자 변경 임계값
speaker_diarization.speaker_change_threshold = 0.7
```

## 😊 감정 분석 (Emotion Recognition)

### 🔧 작동 원리

1. **감정 특성 추출**
   - MFCC (13차원)
   - 스펙트럼 중심 주파수
   - 스펙트럼 롤오프
   - 제로 크로싱 레이트
   - 스펙트럼 대비

2. **감정 분류**
   - LSTM + 어텐션 기반 분류기
   - 7가지 감정 클래스 분류
   - 실시간 감정 변화 추적

### 🏗️ 모델 구조

```python
EmotionClassifier:
├── Feature Extractor (128 → 256)
├── LSTM (3 layers, bidirectional)
├── Multi-head Attention (8 heads)
└── Classifier (256 → 128 → 64 → 7)
```

### 📊 감정 클래스

| 감정 | 한국어 | 색상 | 설명 |
|------|--------|------|------|
| angry | 분노 | 🔴 빨강 | 화난 목소리 |
| disgust | 혐오 | 🟫 갈색 | 싫어하는 목소리 |
| fear | 두려움 | 🟣 보라 | 무서워하는 목소리 |
| happy | 기쁨 | 🟡 노랑 | 즐거운 목소리 |
| sad | 슬픔 | 🔵 파랑 | 슬픈 목소리 |
| surprise | 놀람 | 🟠 주황 | 놀란 목소리 |
| neutral | 중립 | ⚪ 회색 | 평온한 목소리 |

### 🎛️ 주요 매개변수

```python
# 감정 분석 설정
emotion_recognition = EmotionRecognition(
    model_path='emotion_model.pth',
    sample_rate=16000,
    frame_duration=0.025,      # 25ms 프레임
    device='cpu'
)

# 감정 히스토리 크기
emotion_recognition.emotion_history = deque(maxlen=50)
```

## 🔄 통합 처리 (Integrated Audio Processor)

### 🎯 통합 기능

```python
processor = IntegratedAudioProcessor(
    speaker_model_path='speaker_model.pth',
    emotion_model_path='emotion_model.pth',
    enable_speaker_diarization=True,
    enable_emotion_recognition=True,
    enable_vad=True
)
```

### 📊 실시간 처리 파이프라인

```
오디오 입력 → VAD → 화자 구분 → 감정 분석 → 통합 결과
    ↓           ↓         ↓         ↓         ↓
음성 감지   화자 변경   감정 변화   통계 분석   콜백 호출
```

### 🎛️ 콜백 시스템

```python
def on_speaker_change(speaker_info):
    print(f"화자 변경: {speaker_info}")

def on_emotion_change(emotion_info):
    print(f"감정 변경: {emotion_info['current_emotion']}")

def on_vad_change(vad_info):
    print(f"음성 감지: {vad_info}")

processor.set_callbacks(
    on_speaker_change=on_speaker_change,
    on_emotion_change=on_emotion_change,
    on_vad_change=on_vad_change
)
```

## 🚀 사용법

### 1️⃣ 기본 사용법

```python
from integrated_audio_processor import IntegratedAudioProcessor

# 프로세서 생성
processor = IntegratedAudioProcessor()

# 콜백 설정
def on_result(result):
    print(f"화자: {result.get('speaker_info')}")
    print(f"감정: {result.get('emotion_info', {}).get('current_emotion')}")

processor.set_callbacks(on_integrated_result=on_result)

# 처리 시작
processor.start_processing()

# 30초 후 분석
import time
time.sleep(30)
analysis = processor.get_integrated_analysis(duration=30.0)
print(analysis)

# 처리 중지
processor.stop_processing()
```

### 2️⃣ 고급 사용법

```python
# 커스텀 설정
processor = IntegratedAudioProcessor(
    speaker_model_path='custom_speaker_model.pth',
    emotion_model_path='custom_emotion_model.pth',
    enable_speaker_diarization=True,
    enable_emotion_recognition=True,
    enable_vad=True
)

# 실시간 통계 모니터링
while True:
    status = processor.get_processing_status()
    print(f"현재 화자: {status['current_speaker']}")
    print(f"현재 감정: {status['current_emotion']}")
    
    # 감정 추세 분석
    trend = processor.get_emotion_trend(window_size=20)
    print(f"감정 추세: {trend['trend']}")
    
    time.sleep(1)
```

### 3️⃣ 결과 저장 및 로드

```python
# 결과 저장
processor.save_results('analysis_results.json')

# 결과 로드
processor.load_results('analysis_results.json')

# 통계 정보
speaker_stats = processor.get_speaker_statistics()
emotion_stats = processor.get_emotion_statistics()
print(f"화자 통계: {speaker_stats}")
print(f"감정 통계: {emotion_stats}")
```

## 🎨 GUI 통합

### PySide6 위젯 예시

```python
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import QTimer

class AudioAnalysisWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.processor = IntegratedAudioProcessor()
        self.setup_ui()
        self.setup_processor()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        self.speaker_label = QLabel("화자: 없음")
        self.emotion_label = QLabel("감정: 없음")
        self.confidence_label = QLabel("신뢰도: 0%")
        
        layout.addWidget(self.speaker_label)
        layout.addWidget(self.emotion_label)
        layout.addWidget(self.confidence_label)
        
        self.setLayout(layout)
    
    def setup_processor(self):
        self.processor.set_callbacks(
            on_speaker_change=self.on_speaker_change,
            on_emotion_change=self.on_emotion_change
        )
        
        # 타이머로 UI 업데이트
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(100)  # 100ms마다 업데이트
    
    def on_speaker_change(self, speaker_info):
        self.speaker_label.setText(f"화자: {speaker_info.get('speaker_id', 'Unknown')}")
    
    def on_emotion_change(self, emotion_info):
        emotion = emotion_info.get('current_emotion', 'neutral')
        confidence = emotion_info.get('confidence', 0.0)
        
        self.emotion_label.setText(f"감정: {emotion}")
        self.confidence_label.setText(f"신뢰도: {confidence:.1%}")
    
    def update_ui(self):
        # 실시간 UI 업데이트
        pass
```

## 📈 성능 최적화

### 1️⃣ 모델 최적화

```python
# GPU 사용
processor = IntegratedAudioProcessor(device='cuda')

# 배치 처리
processor.batch_size = 32

# 모델 양자화 (선택사항)
import torch.quantization
quantized_model = torch.quantization.quantize_dynamic(
    processor.speaker_diarization.encoder,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### 2️⃣ 메모리 최적화

```python
# 버퍼 크기 조정
processor.audio_buffer = deque(maxlen=16000 * 5)  # 5초 버퍼

# 결과 히스토리 제한
processor.results['integrated_results'] = processor.results['integrated_results'][-1000:]
```

### 3️⃣ 실시간 성능

```python
# 처리 주기 조정
processor.processing_interval = 0.02  # 20ms

# 멀티스레딩
processor.num_threads = 4
```

## 🔧 모델 훈련

### 1️⃣ 화자 구분 모델 훈련

```python
from speaker_diarization import train_speaker_model

# 훈련 데이터 준비
speaker_labels = {
    'speaker1_audio1': 'speaker1',
    'speaker1_audio2': 'speaker1',
    'speaker2_audio1': 'speaker2',
    'speaker2_audio2': 'speaker2'
}

# 모델 훈련
train_speaker_model(
    audio_dir='speaker_training_data',
    speaker_labels=speaker_labels,
    model_save_path='speaker_model.pth'
)
```

### 2️⃣ 감정 분석 모델 훈련

```python
from emotion_recognition import train_emotion_model

# 훈련 데이터 준비
emotion_labels = {
    'happy_audio1': 3,    # happy
    'sad_audio1': 4,      # sad
    'angry_audio1': 0,    # angry
    'neutral_audio1': 6   # neutral
}

# 모델 훈련
train_emotion_model(
    audio_dir='emotion_training_data',
    emotion_labels=emotion_labels,
    model_save_path='emotion_model.pth'
)
```

## 📊 분석 결과 예시

### 통합 분석 결과

```json
{
  "duration": 30.0,
  "total_processed_chunks": 1200,
  "speaker_analysis": {
    "total_speaker_changes": 5,
    "current_speaker": "Speaker_1",
    "speaker_statistics": {
      "total_segments": 1200,
      "unique_speakers": 3,
      "speaker_distribution": {
        "Speaker_0": 400,
        "Speaker_1": 500,
        "Speaker_2": 300
      }
    }
  },
  "emotion_analysis": {
    "current_emotion": "happy",
    "emotion_trend": {
      "trend": "moderate",
      "dominant_emotion": "happy",
      "stability": 0.75
    },
    "emotion_intensity": 0.68,
    "emotion_statistics": {
      "total_analyses": 1200,
      "emotion_distribution": {
        "happy": 600,
        "neutral": 400,
        "sad": 200
      },
      "average_confidence": 0.82
    }
  },
  "vad_analysis": {
    "speech_ratio": 0.85,
    "total_speech_segments": 1020,
    "total_segments": 1200
  }
}
```

## ⚠️ 주의사항

### 1️⃣ 성능 고려사항

- **CPU 사용량**: 실시간 처리 시 높은 CPU 사용률
- **메모리 사용량**: 오디오 버퍼와 결과 저장으로 인한 메모리 증가
- **지연 시간**: 모델 추론으로 인한 약간의 지연

### 2️⃣ 정확도 제한

- **화자 구분**: 비슷한 목소리 구분 어려움
- **감정 분석**: 문화적 차이와 개인차 고려 필요
- **노이즈**: 배경 소음에 민감할 수 있음

### 3️⃣ 데이터 요구사항

- **훈련 데이터**: 충분한 화자별/감정별 오디오 샘플 필요
- **데이터 품질**: 고품질 오디오 녹음 권장
- **다양성**: 다양한 환경과 조건의 데이터 필요

## 🚀 향후 발전 방향

### 1️⃣ 모델 개선

- **Transformer 기반 모델**: 더 정확한 화자/감정 인식
- **멀티모달 융합**: 음성 + 표정 + 제스처 통합 분석
- **적응형 학습**: 실시간 모델 업데이트

### 2️⃣ 기능 확장

- **화자 인증**: 특정 화자 식별 및 인증
- **감정 강도 측정**: 감정의 세밀한 강도 분석
- **대화 분석**: 화자 간 상호작용 패턴 분석

### 3️⃣ 실용성 향상

- **에지 디바이스 최적화**: 모바일/임베디드 지원
- **클라우드 연동**: 실시간 클라우드 분석
- **API 서비스**: RESTful API 제공

## 📚 참고 자료

- [Speaker Diarization Paper](https://arxiv.org/abs/2003.02405)
- [Emotion Recognition Survey](https://arxiv.org/abs/1803.08974)
- [PyTorch Audio Tutorials](https://pytorch.org/audio/stable/tutorials/index.html)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)

---

이 가이드를 통해 화자 구분과 감정 분석 기능을 효과적으로 활용하여 고급 음성 인식 애플리케이션을 구축할 수 있습니다! 🎉 