# 딥러닝 기반 VAD (Voice Activity Detection) 가이드

## 🧠 딥러닝 VAD란?

딥러닝 기반 VAD는 기존의 규칙 기반 VAD보다 훨씬 정확한 음성 활동 감지를 제공합니다. LSTM과 어텐션 메커니즘을 사용하여 복잡한 오디오 환경에서도 높은 정확도를 달성합니다.

## 🏗️ 딥러닝 VAD 구조

### 1. **신경망 아키텍처**

```
입력 (MFCC 특성) → LSTM → 어텐션 → 분류기 → 출력 (음성 확률)
```

#### 주요 구성 요소:
- **LSTM 레이어**: 시퀀스 데이터 처리
- **어텐션 메커니즘**: 중요한 프레임에 집중
- **분류기**: 음성/무음 이진 분류

### 2. **특성 추출**

```python
# MFCC (Mel-frequency cepstral coefficients)
mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)

# 델타 특성 (1차 미분)
mfcc_delta = librosa.feature.delta(mfcc)

# 델타-델타 특성 (2차 미분)
mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

# 최종 특성: 39차원 (13 + 13 + 13)
features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
```

## 📊 성능 비교

| VAD 타입 | 정확도 | 처리 속도 | 메모리 사용량 | 환경 적응성 |
|----------|--------|-----------|---------------|-------------|
| **Simple VAD** | 70-80% | 빠름 | 낮음 | 낮음 |
| **Advanced VAD** | 80-85% | 보통 | 보통 | 보통 |
| **Deep Learning VAD** | 90-95% | 느림 | 높음 | 높음 |
| **Hybrid VAD** | 92-97% | 보통 | 높음 | 매우 높음 |

## 🚀 설치 및 설정

### 1. **필요한 패키지 설치**

```bash
# PyTorch 설치
pip install torch torchvision torchaudio

# 오디오 처리 라이브러리
pip install librosa soundfile

# 기타 의존성
pip install numpy scipy matplotlib
```

### 2. **GPU 사용 (선택사항)**

```bash
# CUDA 지원 PyTorch 설치 (GPU 사용 시)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🎯 사용 방법

### 1. **기본 사용법**

```python
from deep_learning_vad import DeepLearningVAD

# VAD 초기화
vad = DeepLearningVAD(
    model_path="vad_model.pth",  # 사전 훈련된 모델
    sample_rate=16000,
    frame_duration=0.025
)

# 음성 구간 감지 콜백
def on_speech_detected(audio_segment):
    print(f"음성 감지: {len(audio_segment)} 샘플")

# VAD 시작
vad.start_processing()

# 실시간 오디오 처리
while recording:
    audio_chunk = get_audio_chunk()
    segments = vad.process_audio_chunk(audio_chunk)
    
    for is_speech, audio_segment in segments:
        if is_speech:
            on_speech_detected(audio_segment)
```

### 2. **향상된 VAD 사용법**

```python
from enhanced_vad_processor import EnhancedVADProcessor

# 하이브리드 VAD 초기화
vad = EnhancedVADProcessor(
    sample_rate=16000,
    vad_type="hybrid",  # 여러 VAD 결과 결합
    callback=on_speech_detected
)

# VAD 타입 동적 변경
vad.set_vad_type("deep")  # 딥러닝 VAD로 변경
vad.set_vad_type("hybrid")  # 하이브리드 모드로 변경

# 통계 정보 확인
stats = vad.get_stats()
print(f"감지 정확도: {stats['detection_accuracy']:.2%}")
```

### 3. **적응형 VAD 사용법**

```python
from enhanced_vad_processor import AdaptiveVADProcessor

# 적응형 VAD 초기화 (환경에 따라 자동 조정)
vad = AdaptiveVADProcessor(
    sample_rate=16000,
    callback=on_speech_detected
)

# 현재 VAD 정보 확인
info = vad.get_current_vad_info()
print(f"현재 VAD: {info['current_vad']}")
print(f"노이즈 레벨: {info['noise_level']:.3f}")
print(f"음성 감지율: {info['speech_rate']:.3f}")
```

## 🎓 모델 훈련

### 1. **데이터 준비**

#### 오디오 파일 구조:
```
data/
├── audio/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
└── labels/
    ├── sample1.txt
    ├── sample2.txt
    └── ...
```

#### 라벨 파일 형식:
```
0.0 1.5 speech
1.5 2.0 silence
2.0 4.2 speech
4.2 5.0 silence
```

### 2. **모델 훈련**

```python
from deep_learning_vad import train_vad_model

# 모델 훈련
train_vad_model(
    audio_dir="data/audio",
    label_dir="data/labels",
    model_save_path="vad_model.pth"
)
```

### 3. **훈련 파라미터 조정**

```python
from deep_learning_vad import VADNet, VADTrainer
from torch.utils.data import DataLoader

# 모델 생성
model = VADNet(
    input_size=39,      # MFCC 특성 차원
    hidden_size=64,     # LSTM 히든 크기
    num_layers=2        # LSTM 레이어 수
)

# 훈련기 생성
trainer = VADTrainer(model)

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 훈련 실행
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    save_path="vad_model.pth"
)
```

## 🔧 고급 설정

### 1. **신뢰도 임계값 조정**

```python
# 높은 정확도 (낮은 오탐)
vad.set_confidence_threshold(0.8)

# 높은 민감도 (낮은 미탐)
vad.set_confidence_threshold(0.3)

# 기본값
vad.set_confidence_threshold(0.7)
```

### 2. **실시간 성능 최적화**

```python
# 배치 크기 조정
batch_size = 1  # 실시간 처리용

# 모델 양자화 (선택사항)
import torch.quantization

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### 3. **메모리 최적화**

```python
# 그래디언트 체크포인팅 (훈련 시)
model.use_checkpoint = True

# 혼합 정밀도 훈련
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
```

## 📈 성능 평가

### 1. **정확도 메트릭**

```python
def evaluate_vad_accuracy(predictions, ground_truth):
    """VAD 정확도 평가"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
```

### 2. **실시간 성능 측정**

```python
import time

def measure_inference_time(vad, audio_chunk):
    """추론 시간 측정"""
    start_time = time.time()
    segments = vad.process_audio_chunk(audio_chunk)
    end_time = time.time()
    
    inference_time = (end_time - start_time) * 1000  # ms
    return inference_time, segments
```

## 🎯 최적화 팁

### 1. **데이터 품질**
- 고품질 마이크 사용
- 다양한 환경에서 녹음
- 정확한 라벨링

### 2. **모델 아키텍처**
- 적절한 모델 크기 선택
- 드롭아웃으로 과적합 방지
- 배치 정규화 사용

### 3. **훈련 전략**
- 학습률 스케줄링
- 조기 종료 (Early Stopping)
- 데이터 증강

### 4. **실시간 최적화**
- 모델 양자화
- 배치 처리
- GPU 가속

## 🚨 주의사항

1. **메모리 사용량**: 딥러닝 VAD는 상당한 메모리를 사용합니다
2. **처리 지연**: 실시간 처리 시 약간의 지연이 발생할 수 있습니다
3. **GPU 의존성**: GPU 없이는 처리 속도가 느릴 수 있습니다
4. **모델 크기**: 사전 훈련된 모델 파일이 클 수 있습니다

## 🔮 향후 발전 방향

1. **경량화 모델**: 모바일 환경을 위한 경량 VAD
2. **멀티태스크 학습**: 음성 인식과 VAD 동시 학습
3. **자기 지도 학습**: 라벨 없는 데이터로 학습
4. **도메인 적응**: 특정 환경에 최적화된 VAD

딥러닝 VAD는 기존 VAD보다 훨씬 정확하지만, 적절한 하드웨어와 설정이 필요합니다. 프로젝트의 요구사항에 따라 적절한 VAD 방법을 선택하세요! 