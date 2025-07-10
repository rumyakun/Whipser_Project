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
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class SpeakerEncoder(nn.Module):
    """화자 임베딩을 생성하는 신경망"""
    
    def __init__(self, input_size: int = 80, hidden_size: int = 128, embedding_size: int = 256):
        """
        SpeakerEncoder 초기화
        
        Args:
            input_size: 입력 특성 크기 (MFCC 차원)
            hidden_size: LSTM 히든 레이어 크기
            embedding_size: 화자 임베딩 크기
        """
        super(SpeakerEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=3,
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
        
        # 화자 임베딩 생성
        self.embedding_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, embedding_size),
            nn.Tanh()
        )
        
        # 정규화
        self.layer_norm = nn.LayerNorm(embedding_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 (batch_size, sequence_length, input_size)
        
        Returns:
            화자 임베딩 (batch_size, embedding_size)
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
        
        # 글로벌 평균 풀링
        pooled = torch.mean(attn_out, dim=1)  # (batch_size, hidden_size * 2)
        
        # 임베딩 생성
        embedding = self.embedding_layer(pooled)
        
        # 정규화
        embedding = self.layer_norm(embedding)
        
        # L2 정규화
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding

class SpeakerDiarization:
    """화자 구분 클래스"""
    
    def __init__(self, model_path: Optional[str] = None,
                 sample_rate: int = 16000,
                 frame_duration: float = 0.025,
                 embedding_size: int = 256,
                 device: str = 'cpu'):
        """
        SpeakerDiarization 초기화
        
        Args:
            model_path: 사전 훈련된 모델 경로
            sample_rate: 샘플링 레이트
            frame_duration: 프레임 길이
            embedding_size: 임베딩 크기
            device: 사용할 디바이스
        """
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration)
        self.embedding_size = embedding_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 화자 인코더 초기화
        self.encoder = SpeakerEncoder(embedding_size=embedding_size)
        self.encoder.to(self.device)
        
        # 사전 훈련된 모델 로드
        if model_path and os.path.exists(model_path):
            if torch.cuda.is_available():
                self.encoder.load_state_dict(torch.load(model_path))
            else:
                self.encoder.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        self.encoder.eval()
        
        # 클러스터링 모델
        self.clustering_model = None
        self.speaker_embeddings = []
        self.speaker_labels = []
        
        # 오디오 버퍼
        self.audio_buffer = deque(maxlen=self.frame_size * 1000)  # 25초 버퍼
        self.embedding_buffer = deque(maxlen=100)  # 100개 임베딩 버퍼
        
        # 화자 정보
        self.speakers = {}
        self.current_speaker = None
        self.speaker_change_threshold = 0.7
        
    def extract_features(self, audio_frame: np.ndarray) -> np.ndarray:
        """오디오 프레임에서 MFCC 특성 추출"""
        # MFCC 추출 (더 많은 차원 사용)
        mfcc = librosa.feature.mfcc(
            y=audio_frame, 
            sr=self.sample_rate, 
            n_mfcc=80,  # 화자 구분에는 더 많은 차원 사용
            hop_length=self.frame_size,
            n_fft=2048
        )
        
        # 정규화
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
        
        return mfcc.T  # (time, features)
    
    def extract_speaker_embedding(self, audio_frame: np.ndarray) -> np.ndarray:
        """화자 임베딩 추출"""
        with torch.no_grad():
            # 특성 추출
            features = self.extract_features(audio_frame)
            
            if len(features) == 0:
                return np.zeros(self.embedding_size)
            
            # 텐서 변환
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # 임베딩 생성
            embedding = self.encoder(features_tensor)
            
            return embedding.cpu().numpy().flatten()
    
    def cluster_speakers(self, embeddings: List[np.ndarray], 
                        method: str = 'kmeans', n_speakers: Optional[int] = None) -> List[int]:
        """
        화자 클러스터링
        
        Args:
            embeddings: 화자 임베딩 리스트
            method: 클러스터링 방법 ('kmeans', 'agglomerative')
            n_speakers: 화자 수 (None이면 자동 추정)
        
        Returns:
            화자 라벨 리스트
        """
        if len(embeddings) < 2:
            return [0] * len(embeddings)
        
        embeddings_array = np.array(embeddings)
        
        # 화자 수 자동 추정 (Silhouette Score 기반)
        if n_speakers is None:
            n_speakers = self._estimate_number_of_speakers(embeddings_array)
        
        # 클러스터링
        if method == 'kmeans':
            clustering = KMeans(n_clusters=n_speakers, random_state=42)
        elif method == 'agglomerative':
            clustering = AgglomerativeClustering(n_clusters=n_speakers)
        else:
            raise ValueError(f"지원하지 않는 클러스터링 방법: {method}")
        
        labels = clustering.fit_predict(embeddings_array)
        
        return labels.tolist()
    
    def _estimate_number_of_speakers(self, embeddings: np.ndarray) -> int:
        """화자 수 자동 추정"""
        from sklearn.metrics import silhouette_score
        
        max_speakers = min(10, len(embeddings) // 5)  # 최대 10명, 최소 5개 임베딩당 1명
        
        best_score = -1
        best_n = 2
        
        for n in range(2, max_speakers + 1):
            try:
                clustering = KMeans(n_clusters=n, random_state=42)
                labels = clustering.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                
                if score > best_score:
                    best_score = score
                    best_n = n
            except:
                continue
        
        return best_n
    
    def detect_speaker_change(self, current_embedding: np.ndarray, 
                            previous_embedding: np.ndarray) -> bool:
        """화자 변경 감지"""
        if previous_embedding is None:
            return False
        
        # 코사인 유사도 계산
        similarity = cosine_similarity(
            current_embedding.reshape(1, -1), 
            previous_embedding.reshape(1, -1)
        )[0, 0]
        
        # 유사도가 임계값보다 낮으면 화자 변경으로 판단
        return similarity < self.speaker_change_threshold
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> List[Dict[str, Any]]:
        """오디오 청크 처리 및 화자 구분"""
        segments = []
        
        # 오디오 버퍼에 추가
        self.audio_buffer.extend(audio_chunk)
        
        # 프레임 단위로 처리
        for i in range(0, len(audio_chunk), self.frame_size):
            frame = audio_chunk[i:i + self.frame_size]
            
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))
            
            # 화자 임베딩 추출
            embedding = self.extract_speaker_embedding(frame)
            self.embedding_buffer.append(embedding)
            
            # 화자 변경 감지
            if len(self.embedding_buffer) >= 2:
                current_embedding = embedding
                previous_embedding = self.embedding_buffer[-2]
                
                if self.detect_speaker_change(current_embedding, previous_embedding):
                    # 화자 변경 감지됨
                    speaker_info = {
                        'timestamp': len(self.audio_buffer) / self.sample_rate,
                        'embedding': current_embedding,
                        'speaker_change': True
                    }
                    segments.append(speaker_info)
        
        return segments
    
    def identify_speakers(self, audio_segments: List[np.ndarray]) -> List[Dict[str, Any]]:
        """음성 세그먼트에서 화자 식별"""
        if not audio_segments:
            return []
        
        # 각 세그먼트에서 임베딩 추출
        embeddings = []
        for segment in audio_segments:
            embedding = self.extract_speaker_embedding(segment)
            embeddings.append(embedding)
        
        # 화자 클러스터링
        speaker_labels = self.cluster_speakers(embeddings)
        
        # 결과 생성
        results = []
        for i, (segment, embedding, label) in enumerate(zip(audio_segments, embeddings, speaker_labels)):
            result = {
                'segment_id': i,
                'speaker_id': f"Speaker_{label}",
                'embedding': embedding,
                'duration': len(segment) / self.sample_rate,
                'confidence': self._calculate_speaker_confidence(embedding, embeddings, speaker_labels)
            }
            results.append(result)
        
        return results
    
    def _calculate_speaker_confidence(self, embedding: np.ndarray, 
                                    all_embeddings: List[np.ndarray], 
                                    labels: List[int]) -> float:
        """화자 식별 신뢰도 계산"""
        # 같은 화자로 분류된 임베딩들과의 평균 유사도
        same_speaker_embeddings = []
        current_label = labels[all_embeddings.index(embedding)]
        
        for emb, label in zip(all_embeddings, labels):
            if label == current_label:
                same_speaker_embeddings.append(emb)
        
        if len(same_speaker_embeddings) <= 1:
            return 0.5
        
        similarities = []
        for other_embedding in same_speaker_embeddings:
            if not np.array_equal(embedding, other_embedding):
                similarity = cosine_similarity(
                    embedding.reshape(1, -1), 
                    other_embedding.reshape(1, -1)
                )[0, 0]
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    def save_speaker_model(self, filepath: str):
        """화자 모델 저장"""
        torch.save(self.encoder.state_dict(), filepath)
        print(f"화자 모델이 {filepath}에 저장되었습니다.")
    
    def load_speaker_model(self, filepath: str):
        """화자 모델 로드"""
        if os.path.exists(filepath):
            if torch.cuda.is_available():
                self.encoder.load_state_dict(torch.load(filepath))
            else:
                self.encoder.load_state_dict(torch.load(filepath, map_location='cpu'))
            print(f"화자 모델이 {filepath}에서 로드되었습니다.")
    
    def get_speaker_statistics(self) -> Dict[str, Any]:
        """화자 통계 정보 반환"""
        if not self.speaker_embeddings:
            return {}
        
        unique_speakers = set(self.speaker_labels)
        stats = {
            'total_segments': len(self.speaker_embeddings),
            'unique_speakers': len(unique_speakers),
            'speaker_distribution': {}
        }
        
        for speaker in unique_speakers:
            count = self.speaker_labels.count(speaker)
            stats['speaker_distribution'][f"Speaker_{speaker}"] = count
        
        return stats

class SpeakerDiarizationTrainer:
    """화자 구분 모델 훈련 클래스"""
    
    def __init__(self, model: SpeakerEncoder, device: str = 'cpu'):
        """
        SpeakerDiarizationTrainer 초기화
        
        Args:
            model: 훈련할 화자 인코더 모델
            device: 사용할 디바이스
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 손실 함수와 옵티마이저
        self.criterion = nn.TripletMarginLoss(margin=0.3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, save_path: str = 'speaker_model.pth'):
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
        
        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            
            self.optimizer.zero_grad()
            
            anchor_emb = self.model(anchor)
            positive_emb = self.model(positive)
            negative_emb = self.model(negative)
            
            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """한 에포크 검증"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)
                negative_emb = self.model(negative)
                
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)

# 사용 예시
def create_speaker_dataset(audio_dir: str, speaker_labels: Dict[str, str]) -> List[Tuple[np.ndarray, str]]:
    """화자 데이터셋 생성"""
    import os
    import glob
    
    dataset = []
    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    
    for audio_file in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        speaker_id = speaker_labels.get(base_name, "unknown")
        
        audio, sr = librosa.load(audio_file, sr=16000)
        dataset.append((audio, speaker_id))
    
    return dataset

def train_speaker_model(audio_dir: str, speaker_labels: Dict[str, str], 
                       model_save_path: str = "speaker_model.pth"):
    """화자 구분 모델 훈련"""
    # 데이터셋 생성
    dataset = create_speaker_dataset(audio_dir, speaker_labels)
    
    # 모델 및 훈련기 생성
    model = SpeakerEncoder()
    trainer = SpeakerDiarizationTrainer(model)
    
    # 훈련 (실제 구현에서는 데이터 로더 생성 필요)
    print("화자 구분 모델 훈련을 시작합니다...")
    print(f"데이터셋 크기: {len(dataset)}")
    
    # 여기서는 간단한 예시로 구현
    # 실제로는 triplet 데이터 로더를 구현해야 함
    
    print(f"모델이 {model_save_path}에 저장되었습니다.") 