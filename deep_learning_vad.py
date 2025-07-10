import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
from typing import List, Tuple, Optional, Callable
import threading
import queue
import time
from collections import deque

class VADNet(nn.Module):
    """딥러닝 기반 VAD 신경망"""
    
    def __init__(self, input_size: int = 128, hidden_size: int = 64, num_layers: int = 2):
        """
        VADNet 초기화
        
        Args:
            input_size: 입력 특성 크기 (MFCC 차원)
            hidden_size: LSTM 히든 레이어 크기
            num_layers: LSTM 레이어 수
        """
        super(VADNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=4,
            dropout=0.1
        )
        
        # 분류 레이어
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 (batch_size, sequence_length, input_size)
        
        Returns:
            출력 텐서 (batch_size, sequence_length, 1)
        """
        # LSTM 처리
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size * 2)
        
        # 어텐션 적용
        attn_out, _ = self.attention(
            lstm_out.transpose(0, 1),  # (seq_len, batch_size, hidden_size * 2)
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        attn_out = attn_out.transpose(0, 1)  # (batch_size, seq_len, hidden_size * 2)
        
        # 분류
        output = self.classifier(attn_out)
        
        return output

class VADDataset(Dataset):
    """VAD 학습을 위한 데이터셋"""
    
    def __init__(self, audio_files: List[str], labels: List[str], 
                 sample_rate: int = 16000, frame_duration: float = 0.025):
        """
        VADDataset 초기화
        
        Args:
            audio_files: 오디오 파일 경로 리스트
            labels: 라벨 파일 경로 리스트 (시간 기반)
            sample_rate: 샘플링 레이트
            frame_duration: 프레임 길이
        """
        self.audio_files = audio_files
        self.labels = labels
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration)
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # 오디오 로드
        audio, sr = librosa.load(self.audio_files[idx], sr=self.sample_rate)
        
        # MFCC 추출
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=13,
            hop_length=self.frame_size,
            n_fft=2048
        )
        
        # 델타와 델타-델타 추가
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 특성 결합
        features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
        features = features.T  # (time, features)
        
        # 라벨 로드 및 정렬
        labels = self._load_labels(self.labels[idx], len(features))
        
        return torch.FloatTensor(features), torch.FloatTensor(labels)
    
    def _load_labels(self, label_file: str, num_frames: int) -> np.ndarray:
        """라벨 파일 로드 및 프레임 단위로 변환"""
        # 실제 구현에서는 시간 기반 라벨을 프레임 단위로 변환
        # 여기서는 간단한 예시로 구현
        labels = np.zeros(num_frames)
        
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    start_time, end_time, label = line.strip().split()
                    start_frame = int(float(start_time) / 0.025)
                    end_frame = int(float(end_time) / 0.025)
                    
                    if label == 'speech':
                        labels[start_frame:end_frame] = 1.0
        except:
            # 라벨 파일이 없는 경우 기본값
            pass
        
        return labels

class DeepLearningVAD:
    """딥러닝 기반 VAD 클래스"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 sample_rate: int = 16000,
                 frame_duration: float = 0.025,
                 device: str = 'cpu'):
        """
        DeepLearningVAD 초기화
        
        Args:
            model_path: 사전 훈련된 모델 경로
            sample_rate: 샘플링 레이트
            frame_duration: 프레임 길이
            device: 사용할 디바이스 (cpu/cuda)
        """
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 모델 초기화
        self.model = VADNet(input_size=39, hidden_size=64, num_layers=2)
        self.model.to(self.device)
        
        # 사전 훈련된 모델 로드
        if model_path and torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        elif model_path:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        self.model.eval()
        
        # 오디오 버퍼
        self.audio_buffer = deque(maxlen=self.frame_size * 100)  # 2.5초 버퍼
        self.feature_buffer = deque(maxlen=100)  # 100프레임 버퍼
        
        # 상태 변수
        self.is_speech = False
        self.speech_probability = 0.0
        self.confidence_threshold = 0.7
        
    def extract_features(self, audio_frame: np.ndarray) -> np.ndarray:
        """오디오 프레임에서 MFCC 특성 추출"""
        # MFCC 추출
        mfcc = librosa.feature.mfcc(
            y=audio_frame, 
            sr=self.sample_rate, 
            n_mfcc=13,
            hop_length=self.frame_size,
            n_fft=2048
        )
        
        # 델타와 델타-델타 추가
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 특성 결합 및 정규화
        features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
        features = features.T  # (time, features)
        
        # 정규화
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        
        return features
    
    def predict(self, audio_frame: np.ndarray) -> float:
        """단일 프레임에 대한 음성 확률 예측"""
        with torch.no_grad():
            # 특성 추출
            features = self.extract_features(audio_frame)
            
            if len(features) == 0:
                return 0.0
            
            # 텐서 변환
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # 예측
            output = self.model(features_tensor)
            probability = output.mean().item()
            
            return probability
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> List[Tuple[bool, np.ndarray]]:
        """오디오 청크 처리 및 음성 구간 반환"""
        segments = []
        
        # 오디오 버퍼에 추가
        self.audio_buffer.extend(audio_chunk)
        
        # 프레임 단위로 처리
        for i in range(0, len(audio_chunk), self.frame_size):
            frame = audio_chunk[i:i + self.frame_size]
            
            if len(frame) < self.frame_size:
                # 마지막 프레임이 불완전한 경우 패딩
                frame = np.pad(frame, (0, self.frame_size - len(frame)))
            
            # 음성 확률 예측
            speech_prob = self.predict(frame)
            self.speech_probability = speech_prob
            
            # 음성/무음 판단
            is_speech_frame = speech_prob > self.confidence_threshold
            
            if is_speech_frame and not self.is_speech:
                # 음성 시작
                self.is_speech = True
                speech_start = len(self.audio_buffer) - len(audio_chunk) + i
                
            elif not is_speech_frame and self.is_speech:
                # 음성 종료
                if self.is_speech:
                    speech_end = len(self.audio_buffer) - len(audio_chunk) + i
                    speech_audio = np.array(list(self.audio_buffer)[speech_start:speech_end])
                    
                    if len(speech_audio) > self.frame_size * 2:  # 최소 50ms
                        segments.append((True, speech_audio))
                
                self.is_speech = False
        
        return segments
    
    def set_confidence_threshold(self, threshold: float):
        """신뢰도 임계값 설정"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def get_speech_probability(self) -> float:
        """현재 음성 확률 반환"""
        return self.speech_probability
    
    def reset(self):
        """VAD 상태 초기화"""
        self.is_speech = False
        self.speech_probability = 0.0
        self.audio_buffer.clear()
        self.feature_buffer.clear()

class VADTrainer:
    """VAD 모델 훈련 클래스"""
    
    def __init__(self, model: VADNet, device: str = 'cpu'):
        """
        VADTrainer 초기화
        
        Args:
            model: 훈련할 VAD 모델
            device: 사용할 디바이스
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 손실 함수와 옵티마이저
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 50, save_path: str = 'vad_model.pth'):
        """모델 훈련"""
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # 훈련
            train_loss = self._train_epoch(train_loader)
            
            # 검증
            val_loss = self._validate_epoch(val_loader)
            
            # 학습률 조정
            self.scheduler.step(val_loss)
            
            # 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs.squeeze(), labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """한 에포크 검증"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs.squeeze(), labels)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)

# 사용 예시
def create_vad_dataset(audio_dir: str, label_dir: str) -> VADDataset:
    """VAD 데이터셋 생성"""
    import os
    import glob
    
    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    labels = []
    
    for audio_file in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        label_file = os.path.join(label_dir, f"{base_name}.txt")
        labels.append(label_file)
    
    return VADDataset(audio_files, labels)

def train_vad_model(audio_dir: str, label_dir: str, model_save_path: str = "vad_model.pth"):
    """VAD 모델 훈련"""
    # 데이터셋 생성
    dataset = create_vad_dataset(audio_dir, label_dir)
    
    # 데이터 로더 생성
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 모델 및 훈련기 생성
    model = VADNet()
    trainer = VADTrainer(model)
    
    # 훈련
    trainer.train(train_loader, val_loader, epochs=50, save_path=model_save_path)
    
    print(f"모델이 {model_save_path}에 저장되었습니다.") 