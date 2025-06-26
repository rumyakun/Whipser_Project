import subprocess
import json
import tempfile
import os
import wave
import numpy as np
from typing import Optional, List, Dict, Any
import threading
import queue
import time

class WhisperWrapper:
    """Whisper.cpp를 Python에서 사용하기 위한 래퍼 클래스"""
    
    def __init__(self, whisper_path: str = "./whisper.cpp", model_path: str = "./models/ggml-base.bin"):
        """
        WhisperWrapper 초기화
        
        Args:
            whisper_path: whisper.cpp 빌드된 디렉토리 경로
            model_path: GGML 모델 파일 경로
        """
        self.whisper_path = whisper_path
        self.model_path = model_path
        self.whisper_cli = os.path.join(whisper_path, "main")
        
        # Windows 환경에서는 .exe 확장자 추가
        if os.name == 'nt':
            self.whisper_cli += ".exe"
        
        # 모델 파일 존재 확인
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        # whisper-cli 실행 파일 존재 확인
        if not os.path.exists(self.whisper_cli):
            raise FileNotFoundError(f"Whisper CLI를 찾을 수 없습니다: {self.whisper_cli}")
    
    def transcribe_audio_file(self, audio_file: str, language: str = "ko", 
                            output_format: str = "json") -> Dict[str, Any]:
        """
        오디오 파일을 텍스트로 변환
        
        Args:
            audio_file: 오디오 파일 경로
            language: 언어 코드 (ko=한국어)
            output_format: 출력 형식 (json, txt, srt, vtt)
        
        Returns:
            변환 결과 딕셔너리
        """
        try:
            cmd = [
                self.whisper_cli,
                "-m", self.model_path,
                "-f", audio_file,
                "-l", language,
                "-of", output_format,
                "--output-txt",
                "--output-words"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if output_format == "json":
                # JSON 출력 파싱
                return json.loads(result.stdout)
            else:
                return {"text": result.stdout.strip()}
                
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Whisper 실행 오류: {e.stderr}")
    
    def transcribe_audio_data(self, audio_data: np.ndarray, sample_rate: int = 16000,
                            language: str = "ko") -> Dict[str, Any]:
        """
        오디오 데이터를 텍스트로 변환
        
        Args:
            audio_data: 오디오 데이터 (numpy array)
            sample_rate: 샘플링 레이트
            language: 언어 코드
        
        Returns:
            변환 결과 딕셔너리
        """
        # 임시 WAV 파일 생성
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
            
            # WAV 파일로 저장
            with wave.open(temp_filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # 모노
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
        
        try:
            result = self.transcribe_audio_file(temp_filename, language)
            return result
        finally:
            # 임시 파일 삭제
            os.unlink(temp_filename)
    
    def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        models_dir = os.path.join(self.whisper_path, "models")
        if not os.path.exists(models_dir):
            return []
        
        models = []
        for file in os.listdir(models_dir):
            if file.endswith(".bin"):
                models.append(file)
        return models

class RealTimeWhisper:
    """실시간 음성 인식을 위한 클래스"""
    
    def __init__(self, whisper_wrapper: WhisperWrapper, 
                 sample_rate: int = 16000, chunk_duration: float = 3.0):
        """
        RealTimeWhisper 초기화
        
        Args:
            whisper_wrapper: WhisperWrapper 인스턴스
            sample_rate: 샘플링 레이트
            chunk_duration: 청크 길이 (초)
        """
        self.whisper = whisper_wrapper
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio_buffer = []
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
    def start_recording(self):
        """녹음 시작"""
        self.is_recording = True
        self.audio_buffer = []
        
        # 오디오 처리 스레드 시작
        self.audio_thread = threading.Thread(target=self._audio_processing_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()
    
    def stop_recording(self):
        """녹음 중지"""
        self.is_recording = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join()
    
    def add_audio_chunk(self, audio_chunk: np.ndarray):
        """오디오 청크 추가"""
        if self.is_recording:
            self.audio_queue.put(audio_chunk)
    
    def _audio_processing_loop(self):
        """오디오 처리 루프"""
        while self.is_recording:
            try:
                # 오디오 청크 수집
                audio_chunk = self.audio_queue.get(timeout=0.1)
                self.audio_buffer.extend(audio_chunk)
                
                # 충분한 오디오가 모이면 처리
                if len(self.audio_buffer) >= self.chunk_size:
                    audio_data = np.array(self.audio_buffer[:self.chunk_size])
                    self.audio_buffer = self.audio_buffer[self.chunk_size:]
                    
                    # Whisper로 변환
                    try:
                        result = self.whisper.transcribe_audio_data(audio_data, self.sample_rate)
                        if result and 'text' in result and result['text'].strip():
                            self.result_queue.put(result)
                    except Exception as e:
                        print(f"음성 인식 오류: {e}")
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"오디오 처리 오류: {e}")
    
    def get_results(self) -> List[Dict[str, Any]]:
        """처리된 결과 반환"""
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results 