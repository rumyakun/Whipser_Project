import numpy as np
import threading
import queue
import time
from typing import List, Tuple, Optional, Callable
from collections import deque

class SimpleVAD:
    """간단한 음성 활동 감지 (VAD) 클래스"""
    
    def __init__(self, sample_rate: int = 16000, 
                 frame_duration: float = 0.025,
                 silence_threshold: float = 0.01,
                 speech_threshold: float = 0.05,
                 min_speech_duration: float = 0.3,
                 min_silence_duration: float = 0.5):
        """
        SimpleVAD 초기화
        
        Args:
            sample_rate: 샘플링 레이트
            frame_duration: 프레임 길이 (초)
            silence_threshold: 무음 임계값
            speech_threshold: 음성 임계값
            min_speech_duration: 최소 음성 길이 (초)
            min_silence_duration: 최소 무음 길이 (초)
        """
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration)
        self.silence_threshold = silence_threshold
        self.speech_threshold = speech_threshold
        self.min_speech_frames = int(min_speech_duration / frame_duration)
        self.min_silence_frames = int(min_silence_duration / frame_duration)
        
        # 상태 변수
        self.is_speech = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.audio_buffer = deque(maxlen=self.frame_size)
        
    def process_audio(self, audio_data: np.ndarray) -> List[Tuple[bool, np.ndarray]]:
        """
        오디오 데이터를 처리하여 음성/무음 구간 반환
        
        Args:
            audio_data: 오디오 데이터
        
        Returns:
            [(is_speech, audio_segment), ...] 리스트
        """
        segments = []
        
        # 오디오 데이터를 프레임 단위로 처리
        for i in range(0, len(audio_data), self.frame_size):
            frame = audio_data[i:i + self.frame_size]
            
            if len(frame) < self.frame_size:
                # 마지막 프레임이 불완전한 경우 패딩
                frame = np.pad(frame, (0, self.frame_size - len(frame)))
            
            # 프레임 에너지 계산
            energy = np.sqrt(np.mean(frame**2))
            
            # 음성/무음 판단
            if energy > self.speech_threshold:
                self.speech_frames += 1
                self.silence_frames = 0
                
                if self.speech_frames >= self.min_speech_frames:
                    self.is_speech = True
                    
            elif energy < self.silence_threshold:
                self.silence_frames += 1
                self.speech_frames = 0
                
                if self.silence_frames >= self.min_silence_frames:
                    if self.is_speech:
                        # 음성 구간 종료
                        segments.append((True, np.array(list(self.audio_buffer))))
                        self.audio_buffer.clear()
                    self.is_speech = False
            else:
                # 임계값 사이의 영역 - 현재 상태 유지
                pass
            
            # 오디오 버퍼에 프레임 추가
            self.audio_buffer.extend(frame)
        
        return segments
    
    def reset(self):
        """VAD 상태 초기화"""
        self.is_speech = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.audio_buffer.clear()

class AdvancedVAD:
    """고급 음성 활동 감지 클래스 (스펙트럼 기반)"""
    
    def __init__(self, sample_rate: int = 16000,
                 frame_duration: float = 0.025,
                 hop_duration: float = 0.010,
                 num_bands: int = 8,
                 energy_threshold: float = 0.01,
                 spectral_threshold: float = 0.1):
        """
        AdvancedVAD 초기화
        
        Args:
            sample_rate: 샘플링 레이트
            frame_duration: 프레임 길이 (초)
            hop_duration: 프레임 간격 (초)
            num_bands: 주파수 대역 수
            energy_threshold: 에너지 임계값
            spectral_threshold: 스펙트럼 임계값
        """
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration)
        self.hop_size = int(sample_rate * hop_duration)
        self.num_bands = num_bands
        self.energy_threshold = energy_threshold
        self.spectral_threshold = spectral_threshold
        
        # FFT 설정
        self.fft_size = 2**int(np.ceil(np.log2(self.frame_size)))
        self.freq_bands = self._create_frequency_bands()
        
        # 상태 변수
        self.is_speech = False
        self.audio_buffer = deque(maxlen=self.frame_size)
        
    def _create_frequency_bands(self) -> List[Tuple[int, int]]:
        """주파수 대역 생성"""
        bands = []
        band_size = self.fft_size // 2 // self.num_bands
        
        for i in range(self.num_bands):
            start_freq = i * band_size
            end_freq = (i + 1) * band_size
            bands.append((start_freq, end_freq))
        
        return bands
    
    def _calculate_spectral_features(self, frame: np.ndarray) -> Tuple[float, float]:
        """스펙트럼 특징 계산"""
        # FFT 계산
        fft_data = np.fft.fft(frame, self.fft_size)
        magnitude = np.abs(fft_data[:self.fft_size // 2])
        
        # 에너지 계산
        energy = np.sum(magnitude**2)
        
        # 스펙트럼 중심 주파수 계산
        freqs = np.fft.fftfreq(self.fft_size, 1/self.sample_rate)[:self.fft_size // 2]
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
        
        return energy, spectral_centroid
    
    def process_audio(self, audio_data: np.ndarray) -> List[Tuple[bool, np.ndarray]]:
        """
        오디오 데이터를 처리하여 음성/무음 구간 반환
        
        Args:
            audio_data: 오디오 데이터
        
        Returns:
            [(is_speech, audio_segment), ...] 리스트
        """
        segments = []
        
        # 오디오 데이터를 프레임 단위로 처리
        for i in range(0, len(audio_data), self.hop_size):
            frame = audio_data[i:i + self.frame_size]
            
            if len(frame) < self.frame_size:
                # 마지막 프레임이 불완전한 경우 패딩
                frame = np.pad(frame, (0, self.frame_size - len(frame)))
            
            # 스펙트럼 특징 계산
            energy, spectral_centroid = self._calculate_spectral_features(frame)
            
            # 음성/무음 판단 (에너지와 스펙트럼 중심 주파수 기반)
            is_speech_frame = (energy > self.energy_threshold and 
                             spectral_centroid > self.spectral_threshold)
            
            if is_speech_frame and not self.is_speech:
                # 음성 시작
                self.is_speech = True
                self.audio_buffer.clear()
            
            elif not is_speech_frame and self.is_speech:
                # 음성 종료
                if len(self.audio_buffer) > 0:
                    segments.append((True, np.array(list(self.audio_buffer))))
                self.is_speech = False
                self.audio_buffer.clear()
            
            # 오디오 버퍼에 프레임 추가
            if self.is_speech:
                self.audio_buffer.extend(frame)
        
        return segments
    
    def reset(self):
        """VAD 상태 초기화"""
        self.is_speech = False
        self.audio_buffer.clear()

class VADProcessor:
    """VAD 처리를 위한 메인 클래스"""
    
    def __init__(self, sample_rate: int = 16000, 
                 use_advanced_vad: bool = False,
                 callback: Optional[Callable] = None):
        """
        VADProcessor 초기화
        
        Args:
            sample_rate: 샘플링 레이트
            use_advanced_vad: 고급 VAD 사용 여부
            callback: 음성 구간 감지 콜백 함수
        """
        self.sample_rate = sample_rate
        self.callback = callback
        
        # VAD 선택
        if use_advanced_vad:
            self.vad = AdvancedVAD(sample_rate)
        else:
            self.vad = SimpleVAD(sample_rate)
        
        # 오디오 버퍼
        self.audio_buffer = []
        self.is_processing = False
        
    def start_processing(self):
        """VAD 처리 시작"""
        self.is_processing = True
        self.audio_buffer = []
        self.vad.reset()
    
    def stop_processing(self):
        """VAD 처리 중지"""
        self.is_processing = False
        self.audio_buffer = []
    
    def process_audio_chunk(self, audio_chunk: np.ndarray):
        """오디오 청크 처리"""
        if not self.is_processing:
            return
        
        # 오디오 버퍼에 추가
        self.audio_buffer.extend(audio_chunk)
        
        # 충분한 오디오가 모이면 VAD 처리
        if len(self.audio_buffer) >= self.vad.frame_size * 10:  # 10프레임 이상
            audio_data = np.array(self.audio_buffer)
            segments = self.vad.process_audio(audio_data)
            
            # 음성 구간이 감지되면 콜백 호출
            for is_speech, audio_segment in segments:
                if is_speech and self.callback:
                    self.callback(audio_segment)
            
            # 처리된 오디오 제거
            self.audio_buffer = self.audio_buffer[self.vad.frame_size * 10:]
    
    def get_current_audio_level(self) -> float:
        """현재 오디오 레벨 반환"""
        if not self.audio_buffer:
            return 0.0
        
        audio_data = np.array(self.audio_buffer)
        return np.sqrt(np.mean(audio_data**2)) 