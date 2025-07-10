import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
from typing import List, Tuple, Optional, Dict, Any
import threading
import queue
import time
from collections import deque
import pickle
import os

class EmotionClassifier(nn.Module):
    """감정 분류 신경망"""
    
    def __init__(self, input_size: int = 128, hidden_size: int = 256, 
                 num_emotions: int = 7, num_layers: int = 3):
        """
        EmotionClassifier 초기화
        
        Args:
            input_size: 입력 특성 크기
            hidden_size: 히든 레이어 크기
            num_emotions: 감정 클래스 수
            num_layers: LSTM 레이어 수
        """
        super(EmotionClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_emotions = num_emotions
        self.num_layers = num_layers
        
        # 특성 추출 레이어
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=8,
            dropout=0.1
        )
        
        # 감정 분류기
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size // 2, num_emotions),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 (batch_size, sequence_length, input_size)
        
        Returns:
            감정 확률 (batch_size, num_emotions)
        """
        batch_size, seq_len, _ = x.shape
        
        # 특성 추출
        features = self.feature_extractor(x)  # (batch_size, seq_len, hidden_size)
        
        # LSTM 처리
        lstm_out, _ = self.lstm(features)  # (batch_size, seq_len, hidden_size * 2)
        
        # 어텐션 적용
        attn_out, _ = self.attention(
            lstm_out.transpose(0, 1),  # (seq_len, batch_size, hidden_size * 2)
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        attn_out = attn_out.transpose(0, 1)  # (batch_size, seq_len, hidden_size * 2)
        
        # 글로벌 평균 풀링
        pooled = torch.mean(attn_out, dim=1)  # (batch_size, hidden_size * 2)
        
        # 감정 분류
        emotion_probs = self.classifier(pooled)
        
        return emotion_probs

class EmotionRecognition:
    """감정 인식 클래스"""
    
    def __init__(self, model_path: Optional[str] = None,
                 sample_rate: int = 16000,
                 frame_duration: float = 0.025,
                 device: str = 'cpu'):
        """
        EmotionRecognition 초기화
        
        Args:
            model_path: 사전 훈련된 모델 경로
            sample_rate: 샘플링 레이트
            frame_duration: 프레임 길이
            device: 사용할 디바이스
        """
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 감정 클래스 정의
        self.emotions = {
            0: 'angry',      # 분노
            1: 'disgust',    # 혐오
            2: 'fear',       # 두려움
            3: 'happy',      # 기쁨
            4: 'sad',        # 슬픔
            5: 'surprise',   # 놀람
            6: 'neutral'     # 중립
        }
        
        self.emotion_colors = {
            'angry': '#FF4444',      # 빨강
            'disgust': '#8B4513',    # 갈색
            'fear': '#800080',       # 보라
            'happy': '#FFD700',      # 노랑
            'sad': '#4169E1',        # 파랑
            'surprise': '#FF69B4',   # 분홍
            'neutral': '#808080'     # 회색
        }
        
        # 감정 분류기 초기화
        self.classifier = EmotionClassifier(num_emotions=len(self.emotions))
        self.classifier.to(self.device)
        
        # 사전 훈련된 모델 로드
        if model_path and os.path.exists(model_path):
            if torch.cuda.is_available():
                self.classifier.load_state_dict(torch.load(model_path))
            else:
                self.classifier.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        self.classifier.eval()
        
        # 오디오 버퍼
        self.audio_buffer = deque(maxlen=self.frame_size * 200)  # 5초 버퍼
        self.emotion_history = deque(maxlen=50)  # 50개 감정 히스토리
        
        # 현재 감정 상태
        self.current_emotion = 'neutral'
        self.emotion_confidence = 0.0
        self.emotion_probabilities = {}
        
    def extract_features(self, audio_frame: np.ndarray) -> np.ndarray:
        """오디오 프레임에서 감정 특성 추출"""
        # MFCC 추출
        mfcc = librosa.feature.mfcc(
            y=audio_frame, 
            sr=self.sample_rate, 
            n_mfcc=13,
            hop_length=self.frame_size,
            n_fft=2048
        )
        
        # 스펙트럼 중심 주파수
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_frame, sr=self.sample_rate, hop_length=self.frame_size
        )
        
        # 스펙트럼 롤오프
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_frame, sr=self.sample_rate, hop_length=self.frame_size
        )
        
        # 제로 크로싱 레이트
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            audio_frame, hop_length=self.frame_size
        )
        
        # 스펙트럼 대비
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio_frame, sr=self.sample_rate, hop_length=self.frame_size
        )
        
        # 특성 결합
        features = np.concatenate([
            mfcc,
            spectral_centroids,
            spectral_rolloff,
            zero_crossing_rate,
            spectral_contrast
        ], axis=0)
        
        # 정규화
        features = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-8)
        
        return features.T  # (time, features)
    
    def predict_emotion(self, audio_frame: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """감정 예측"""
        with torch.no_grad():
            # 특성 추출
            features = self.extract_features(audio_frame)
            
            if len(features) == 0:
                return 'neutral', 0.0, {emotion: 0.0 for emotion in self.emotions.values()}
            
            # 텐서 변환
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # 감정 예측
            emotion_probs = self.classifier(features_tensor)
            probabilities = emotion_probs.cpu().numpy().flatten()
            
            # 최고 확률 감정 선택
            predicted_idx = np.argmax(probabilities)
            predicted_emotion = self.emotions[predicted_idx]
            confidence = probabilities[predicted_idx]
            
            # 모든 감정 확률을 딕셔너리로 변환
            emotion_probs_dict = {self.emotions[i]: prob for i, prob in enumerate(probabilities)}
            
            return predicted_emotion, confidence, emotion_probs_dict
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """오디오 청크 처리 및 감정 분석"""
        # 오디오 버퍼에 추가
        self.audio_buffer.extend(audio_chunk)
        
        # 프레임 단위로 처리
        emotions_in_chunk = []
        
        for i in range(0, len(audio_chunk), self.frame_size):
            frame = audio_chunk[i:i + self.frame_size]
            
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))
            
            # 감정 예측
            emotion, confidence, probabilities = self.predict_emotion(frame)
            
            emotions_in_chunk.append({
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': probabilities,
                'timestamp': len(self.audio_buffer) / self.sample_rate
            })
        
        # 감정 히스토리 업데이트
        if emotions_in_chunk:
            # 가장 높은 신뢰도의 감정을 현재 감정으로 설정
            best_emotion = max(emotions_in_chunk, key=lambda x: x['confidence'])
            self.current_emotion = best_emotion['emotion']
            self.emotion_confidence = best_emotion['confidence']
            self.emotion_probabilities = best_emotion['probabilities']
            
            self.emotion_history.append(best_emotion)
        
        return {
            'current_emotion': self.current_emotion,
            'confidence': self.emotion_confidence,
            'probabilities': self.emotion_probabilities,
            'emotions_in_chunk': emotions_in_chunk,
            'emotion_history': list(self.emotion_history)
        }
    
    def get_emotion_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """감정 변화 추세 분석"""
        if len(self.emotion_history) < window_size:
            return {'trend': 'insufficient_data', 'dominant_emotion': self.current_emotion}
        
        # 최근 감정들 분석
        recent_emotions = [item['emotion'] for item in list(self.emotion_history)[-window_size:]]
        
        # 감정 빈도 계산
        emotion_counts = {}
        for emotion in self.emotions.values():
            emotion_counts[emotion] = recent_emotions.count(emotion)
        
        # 주요 감정
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        dominant_count = emotion_counts[dominant_emotion]
        
        # 감정 안정성 계산
        unique_emotions = len(set(recent_emotions))
        stability = 1.0 - (unique_emotions - 1) / (len(self.emotions) - 1)
        
        # 감정 변화 추세
        if dominant_count >= window_size * 0.7:  # 70% 이상이 같은 감정
            trend = 'stable'
        elif dominant_count >= window_size * 0.5:  # 50% 이상이 같은 감정
            trend = 'moderate'
        else:
            trend = 'unstable'
        
        return {
            'trend': trend,
            'dominant_emotion': dominant_emotion,
            'stability': stability,
            'emotion_distribution': emotion_counts,
            'recent_emotions': recent_emotions
        }
    
    def get_emotion_intensity(self) -> float:
        """감정 강도 계산"""
        if not self.emotion_probabilities:
            return 0.0
        
        # 중립을 제외한 감정들의 평균 확률
        non_neutral_probs = [
            prob for emotion, prob in self.emotion_probabilities.items() 
            if emotion != 'neutral'
        ]
        
        if not non_neutral_probs:
            return 0.0
        
        return np.mean(non_neutral_probs)
    
    def get_emotion_color(self, emotion: str) -> str:
        """감정에 해당하는 색상 반환"""
        return self.emotion_colors.get(emotion, '#808080')
    
    def save_emotion_model(self, filepath: str):
        """감정 모델 저장"""
        torch.save(self.classifier.state_dict(), filepath)
        print(f"감정 모델이 {filepath}에 저장되었습니다.")
    
    def load_emotion_model(self, filepath: str):
        """감정 모델 로드"""
        if os.path.exists(filepath):
            if torch.cuda.is_available():
                self.classifier.load_state_dict(torch.load(filepath))
            else:
                self.classifier.load_state_dict(torch.load(filepath, map_location='cpu'))
            print(f"감정 모델이 {filepath}에서 로드되었습니다.")
    
    def get_emotion_statistics(self) -> Dict[str, Any]:
        """감정 통계 정보 반환"""
        if not self.emotion_history:
            return {}
        
        # 감정별 빈도 계산
        emotion_counts = {}
        for emotion in self.emotions.values():
            emotion_counts[emotion] = 0
        
        for item in self.emotion_history:
            emotion_counts[item['emotion']] += 1
        
        # 평균 신뢰도 계산
        avg_confidence = np.mean([item['confidence'] for item in self.emotion_history])
        
        # 감정 변화 횟수 계산
        emotion_changes = 0
        for i in range(1, len(self.emotion_history)):
            if self.emotion_history[i]['emotion'] != self.emotion_history[i-1]['emotion']:
                emotion_changes += 1
        
        return {
            'total_analyses': len(self.emotion_history),
            'emotion_distribution': emotion_counts,
            'average_confidence': avg_confidence,
            'emotion_changes': emotion_changes,
            'current_emotion': self.current_emotion,
            'current_confidence': self.emotion_confidence
        }

class EmotionRecognitionTrainer:
    """감정 인식 모델 훈련 클래스"""
    
    def __init__(self, model: EmotionClassifier, device: str = 'cpu'):
        """
        EmotionRecognitionTrainer 초기화
        
        Args:
            model: 훈련할 감정 분류 모델
            device: 사용할 디바이스
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 손실 함수와 옵티마이저
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, save_path: str = 'emotion_model.pth'):
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
            loss = self.criterion(outputs, labels)
            
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
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)

class EmotionDataset(Dataset):
    """감정 인식 데이터셋"""
    
    def __init__(self, audio_files: List[str], emotion_labels: List[int], 
                 sample_rate: int = 16000, frame_duration: float = 0.025):
        """
        EmotionDataset 초기화
        
        Args:
            audio_files: 오디오 파일 경로 리스트
            emotion_labels: 감정 라벨 리스트
            sample_rate: 샘플링 레이트
            frame_duration: 프레임 길이
        """
        self.audio_files = audio_files
        self.emotion_labels = emotion_labels
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration)
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # 오디오 로드
        audio, sr = librosa.load(self.audio_files[idx], sr=self.sample_rate)
        
        # 특성 추출 (감정 인식용)
        features = self._extract_emotion_features(audio)
        
        # 라벨
        label = self.emotion_labels[idx]
        
        return torch.FloatTensor(features), torch.LongTensor([label])
    
    def _extract_emotion_features(self, audio: np.ndarray) -> np.ndarray:
        """감정 인식용 특성 추출"""
        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate, 
            n_mfcc=13,
            hop_length=self.frame_size,
            n_fft=2048
        )
        
        # 스펙트럼 중심 주파수
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.frame_size
        )
        
        # 스펙트럼 롤오프
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=self.frame_size
        )
        
        # 제로 크로싱 레이트
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.frame_size
        )
        
        # 스펙트럼 대비
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio, sr=self.sample_rate, hop_length=self.frame_size
        )
        
        # 특성 결합
        features = np.concatenate([
            mfcc,
            spectral_centroids,
            spectral_rolloff,
            zero_crossing_rate,
            spectral_contrast
        ], axis=0)
        
        # 정규화
        features = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-8)
        
        return features.T  # (time, features)

# 사용 예시
def create_emotion_dataset(audio_dir: str, emotion_labels: Dict[str, int]) -> EmotionDataset:
    """감정 데이터셋 생성"""
    import os
    import glob
    
    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    labels = []
    
    for audio_file in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        emotion_id = emotion_labels.get(base_name, 6)  # 기본값: neutral
        labels.append(emotion_id)
    
    return EmotionDataset(audio_files, labels)

def train_emotion_model(audio_dir: str, emotion_labels: Dict[str, int], 
                       model_save_path: str = "emotion_model.pth"):
    """감정 인식 모델 훈련"""
    # 데이터셋 생성
    dataset = create_emotion_dataset(audio_dir, emotion_labels)
    
    # 데이터 로더 생성
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 모델 및 훈련기 생성
    model = EmotionClassifier()
    trainer = EmotionRecognitionTrainer(model)
    
    # 훈련
    trainer.train(train_loader, val_loader, epochs=100, save_path=model_save_path)
    
    print(f"감정 모델이 {model_save_path}에 저장되었습니다.") 