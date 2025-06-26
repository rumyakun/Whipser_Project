import sounddevice as sd
import numpy as np
import threading
import queue
import time
from typing import Callable, Optional

class AudioCapture:
    """실시간 오디오 캡처 클래스"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, 
                 chunk_size: int = 1024, callback: Optional[Callable] = None):
        """
        AudioCapture 초기화
        
        Args:
            sample_rate: 샘플링 레이트
            channels: 채널 수 (1=모노, 2=스테레오)
            chunk_size: 청크 크기
            callback: 오디오 데이터 콜백 함수
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.callback = callback
        
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        
        # 오디오 스트림 설정
        self.stream = None
    
    def start_recording(self):
        """녹음 시작"""
        if self.is_recording:
            return
        
        self.is_recording = True
        
        # 오디오 스트림 시작
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=self.chunk_size,
            callback=self._audio_callback
        )
        
        self.stream.start()
        
        # 오디오 처리 스레드 시작
        self.recording_thread = threading.Thread(target=self._audio_processing_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        print("오디오 녹음이 시작되었습니다.")
    
    def stop_recording(self):
        """녹음 중지"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # 오디오 스트림 중지
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # 스레드 종료 대기
        if self.recording_thread:
            self.recording_thread.join()
        
        print("오디오 녹음이 중지되었습니다.")
    
    def _audio_callback(self, indata, frames, time, status):
        """오디오 콜백 함수"""
        if status:
            print(f"오디오 스트림 상태: {status}")
        
        if self.is_recording:
            # 오디오 데이터를 큐에 추가
            audio_data = indata.copy()
            if self.channels == 2:
                # 스테레오를 모노로 변환
                audio_data = np.mean(audio_data, axis=1)
            
            self.audio_queue.put(audio_data.flatten())
    
    def _audio_processing_loop(self):
        """오디오 처리 루프"""
        while self.is_recording:
            try:
                # 오디오 데이터 가져오기
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # 콜백 함수 호출
                if self.callback:
                    self.callback(audio_data)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"오디오 처리 오류: {e}")
    
    def get_audio_level(self) -> float:
        """현재 오디오 레벨 반환 (0.0 ~ 1.0)"""
        if not self.audio_queue.empty():
            try:
                audio_data = self.audio_queue.get_nowait()
                return np.sqrt(np.mean(audio_data**2))
            except queue.Empty:
                pass
        return 0.0
    
    def list_audio_devices(self):
        """사용 가능한 오디오 장치 목록 출력"""
        devices = sd.query_devices()
        print("사용 가능한 오디오 장치:")
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']} (입력 채널: {device['max_inputs']}, 출력 채널: {device['max_outputs']})")
        
        # 기본 장치 정보
        print(f"\n기본 입력 장치: {sd.query_devices(kind='input')['name']}")
        print(f"기본 출력 장치: {sd.query_devices(kind='output')['name']}")

class AudioVisualizer:
    """오디오 시각화를 위한 클래스"""
    
    def __init__(self, sample_rate: int = 16000, fft_size: int = 1024):
        """
        AudioVisualizer 초기화
        
        Args:
            sample_rate: 샘플링 레이트
            fft_size: FFT 크기
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.audio_buffer = np.zeros(fft_size)
        self.buffer_index = 0
    
    def update(self, audio_data: np.ndarray) -> np.ndarray:
        """
        오디오 데이터 업데이트 및 스펙트럼 반환
        
        Args:
            audio_data: 오디오 데이터
        
        Returns:
            주파수 스펙트럼 (dB)
        """
        # 오디오 버퍼 업데이트
        for sample in audio_data:
            self.audio_buffer[self.buffer_index] = sample
            self.buffer_index = (self.buffer_index + 1) % self.fft_size
        
        # FFT 계산
        fft_data = np.fft.fft(self.audio_buffer)
        magnitude = np.abs(fft_data[:self.fft_size // 2])
        
        # dB로 변환
        spectrum = 20 * np.log10(magnitude + 1e-10)
        
        return spectrum
    
    def get_frequency_bands(self, audio_data: np.ndarray, num_bands: int = 8) -> np.ndarray:
        """
        주파수 대역별 에너지 반환
        
        Args:
            audio_data: 오디오 데이터
            num_bands: 대역 수
        
        Returns:
            각 대역의 에너지
        """
        spectrum = self.update(audio_data)
        
        # 주파수 대역 분할
        band_size = len(spectrum) // num_bands
        bands = []
        
        for i in range(num_bands):
            start_idx = i * band_size
            end_idx = start_idx + band_size
            band_energy = np.mean(spectrum[start_idx:end_idx])
            bands.append(band_energy)
        
        return np.array(bands) 