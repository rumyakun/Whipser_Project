import numpy as np
import torch
import threading
import queue
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import json
import os

from speaker_diarization import SpeakerDiarization
from emotion_recognition import EmotionRecognition
from audio_capture import AudioCapture
from vad_processor import VADProcessor

class IntegratedAudioProcessor:
    """화자 구분과 감정 분석을 통합한 오디오 처리 클래스"""
    
    def __init__(self, 
                 speaker_model_path: Optional[str] = None,
                 emotion_model_path: Optional[str] = None,
                 sample_rate: int = 16000,
                 device: str = 'cpu',
                 enable_speaker_diarization: bool = True,
                 enable_emotion_recognition: bool = True,
                 enable_vad: bool = True):
        """
        IntegratedAudioProcessor 초기화
        
        Args:
            speaker_model_path: 화자 구분 모델 경로
            emotion_model_path: 감정 분석 모델 경로
            sample_rate: 샘플링 레이트
            device: 사용할 디바이스
            enable_speaker_diarization: 화자 구분 활성화
            enable_emotion_recognition: 감정 분석 활성화
            enable_vad: 음성 활동 감지 활성화
        """
        self.sample_rate = sample_rate
        self.device = device
        self.enable_speaker_diarization = enable_speaker_diarization
        self.enable_emotion_recognition = enable_emotion_recognition
        self.enable_vad = enable_vad
        
        # 오디오 캡처
        self.audio_capture = AudioCapture(sample_rate=sample_rate)
        
        # VAD 프로세서
        if enable_vad:
            self.vad_processor = VADProcessor(sample_rate=sample_rate)
        
        # 화자 구분
        if enable_speaker_diarization:
            self.speaker_diarization = SpeakerDiarization(
                model_path=speaker_model_path,
                sample_rate=sample_rate,
                device=device
            )
        
        # 감정 분석
        if enable_emotion_recognition:
            self.emotion_recognition = EmotionRecognition(
                model_path=emotion_model_path,
                sample_rate=sample_rate,
                device=device
            )
        
        # 처리 결과 저장
        self.results = {
            'speaker_segments': [],
            'emotion_segments': [],
            'vad_segments': [],
            'integrated_results': []
        }
        
        # 실시간 처리 버퍼
        self.audio_buffer = deque(maxlen=sample_rate * 10)  # 10초 버퍼
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # 처리 상태
        self.is_processing = False
        self.processing_thread = None
        
        # 콜백 함수
        self.on_speaker_change = None
        self.on_emotion_change = None
        self.on_vad_change = None
        self.on_integrated_result = None
    
    def start_processing(self):
        """실시간 오디오 처리 시작"""
        if self.is_processing:
            return
        
        self.is_processing = True
        
        # 오디오 캡처 시작
        self.audio_capture.start_capture()
        
        # 처리 스레드 시작
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("통합 오디오 처리가 시작되었습니다.")
    
    def stop_processing(self):
        """실시간 오디오 처리 중지"""
        if not self.is_processing:
            return
        
        self.is_processing = False
        
        # 오디오 캡처 중지
        self.audio_capture.stop_capture()
        
        # 처리 스레드 대기
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        
        print("통합 오디오 처리가 중지되었습니다.")
    
    def _processing_loop(self):
        """실시간 처리 루프"""
        while self.is_processing:
            try:
                # 오디오 데이터 가져오기
                audio_chunk = self.audio_capture.get_audio_chunk()
                
                if audio_chunk is not None and len(audio_chunk) > 0:
                    # 오디오 버퍼에 추가
                    self.audio_buffer.extend(audio_chunk)
                    
                    # 통합 처리
                    result = self._process_audio_chunk(audio_chunk)
                    
                    # 결과 저장
                    if result:
                        self.results['integrated_results'].append(result)
                        
                        # 콜백 호출
                        if self.on_integrated_result:
                            self.on_integrated_result(result)
                
                time.sleep(0.01)  # 10ms 대기
                
            except Exception as e:
                print(f"처리 중 오류 발생: {e}")
                continue
    
    def _process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[Dict[str, Any]]:
        """오디오 청크 통합 처리"""
        result = {
            'timestamp': time.time(),
            'audio_duration': len(audio_chunk) / self.sample_rate,
            'speaker_info': None,
            'emotion_info': None,
            'vad_info': None
        }
        
        # VAD 처리
        if self.enable_vad:
            vad_result = self.vad_processor.process_audio_chunk(audio_chunk)
            result['vad_info'] = vad_result
            
            # VAD 콜백
            if self.on_vad_change and vad_result.get('speech_detected'):
                self.on_vad_change(vad_result)
        
        # 음성이 감지된 경우에만 화자 구분과 감정 분석 수행
        if not self.enable_vad or (result['vad_info'] and result['vad_info'].get('speech_detected')):
            
            # 화자 구분
            if self.enable_speaker_diarization:
                speaker_result = self.speaker_diarization.process_audio_chunk(audio_chunk)
                result['speaker_info'] = speaker_result
                
                # 화자 변경 콜백
                if self.on_speaker_change and speaker_result:
                    for segment in speaker_result:
                        if segment.get('speaker_change'):
                            self.on_speaker_change(segment)
            
            # 감정 분석
            if self.enable_emotion_recognition:
                emotion_result = self.emotion_recognition.process_audio_chunk(audio_chunk)
                result['emotion_info'] = emotion_result
                
                # 감정 변경 콜백
                if self.on_emotion_change and emotion_result:
                    current_emotion = emotion_result.get('current_emotion')
                    if current_emotion and current_emotion != 'neutral':
                        self.on_emotion_change(emotion_result)
        
        return result
    
    def get_current_speaker(self) -> Optional[str]:
        """현재 화자 반환"""
        if not self.enable_speaker_diarization:
            return None
        
        if self.speaker_diarization.current_speaker:
            return self.speaker_diarization.current_speaker
        return None
    
    def get_current_emotion(self) -> Optional[str]:
        """현재 감정 반환"""
        if not self.enable_emotion_recognition:
            return None
        
        return self.emotion_recognition.current_emotion
    
    def get_speaker_statistics(self) -> Dict[str, Any]:
        """화자 통계 반환"""
        if not self.enable_speaker_diarization:
            return {}
        
        return self.speaker_diarization.get_speaker_statistics()
    
    def get_emotion_statistics(self) -> Dict[str, Any]:
        """감정 통계 반환"""
        if not self.enable_emotion_recognition:
            return {}
        
        return self.emotion_recognition.get_emotion_statistics()
    
    def get_emotion_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """감정 변화 추세 반환"""
        if not self.enable_emotion_recognition:
            return {}
        
        return self.emotion_recognition.get_emotion_trend(window_size)
    
    def get_integrated_analysis(self, duration: float = 10.0) -> Dict[str, Any]:
        """통합 분석 결과 반환"""
        current_time = time.time()
        recent_results = [
            result for result in self.results['integrated_results']
            if current_time - result['timestamp'] <= duration
        ]
        
        if not recent_results:
            return {}
        
        # 화자 분석
        speaker_analysis = {}
        if self.enable_speaker_diarization:
            speaker_segments = [r for r in recent_results if r['speaker_info']]
            if speaker_segments:
                speaker_analysis = {
                    'total_speaker_changes': len(speaker_segments),
                    'current_speaker': self.get_current_speaker(),
                    'speaker_statistics': self.get_speaker_statistics()
                }
        
        # 감정 분석
        emotion_analysis = {}
        if self.enable_emotion_recognition:
            emotion_segments = [r for r in recent_results if r['emotion_info']]
            if emotion_segments:
                emotion_analysis = {
                    'current_emotion': self.get_current_emotion(),
                    'emotion_trend': self.get_emotion_trend(),
                    'emotion_intensity': self.emotion_recognition.get_emotion_intensity(),
                    'emotion_statistics': self.get_emotion_statistics()
                }
        
        # VAD 분석
        vad_analysis = {}
        if self.enable_vad:
            vad_segments = [r for r in recent_results if r['vad_info']]
            if vad_segments:
                speech_detected = sum(1 for r in vad_segments if r['vad_info'].get('speech_detected'))
                total_segments = len(vad_segments)
                vad_analysis = {
                    'speech_ratio': speech_detected / total_segments if total_segments > 0 else 0,
                    'total_speech_segments': speech_detected,
                    'total_segments': total_segments
                }
        
        return {
            'duration': duration,
            'total_processed_chunks': len(recent_results),
            'speaker_analysis': speaker_analysis,
            'emotion_analysis': emotion_analysis,
            'vad_analysis': vad_analysis,
            'timestamp': current_time
        }
    
    def save_results(self, filepath: str):
        """결과를 JSON 파일로 저장"""
        # 결과를 JSON 직렬화 가능한 형태로 변환
        serializable_results = {
            'speaker_segments': [],
            'emotion_segments': [],
            'vad_segments': [],
            'integrated_results': []
        }
        
        # 통합 결과 변환
        for result in self.results['integrated_results']:
            serializable_result = {
                'timestamp': result['timestamp'],
                'audio_duration': result['audio_duration'],
                'speaker_info': result['speaker_info'],
                'emotion_info': result['emotion_info'],
                'vad_info': result['vad_info']
            }
            serializable_results['integrated_results'].append(serializable_result)
        
        # 파일 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"결과가 {filepath}에 저장되었습니다.")
    
    def load_results(self, filepath: str):
        """JSON 파일에서 결과 로드"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_results = json.load(f)
            
            self.results.update(loaded_results)
            print(f"결과가 {filepath}에서 로드되었습니다.")
    
    def set_callbacks(self, 
                     on_speaker_change=None,
                     on_emotion_change=None,
                     on_vad_change=None,
                     on_integrated_result=None):
        """콜백 함수 설정"""
        self.on_speaker_change = on_speaker_change
        self.on_emotion_change = on_emotion_change
        self.on_vad_change = on_vad_change
        self.on_integrated_result = on_integrated_result
    
    def get_audio_buffer(self) -> np.ndarray:
        """현재 오디오 버퍼 반환"""
        return np.array(list(self.audio_buffer))
    
    def clear_results(self):
        """결과 초기화"""
        self.results = {
            'speaker_segments': [],
            'emotion_segments': [],
            'vad_segments': [],
            'integrated_results': []
        }
        self.audio_buffer.clear()
    
    def get_processing_status(self) -> Dict[str, Any]:
        """처리 상태 반환"""
        return {
            'is_processing': self.is_processing,
            'enable_speaker_diarization': self.enable_speaker_diarization,
            'enable_emotion_recognition': self.enable_emotion_recognition,
            'enable_vad': self.enable_vad,
            'audio_buffer_size': len(self.audio_buffer),
            'total_results': len(self.results['integrated_results']),
            'current_speaker': self.get_current_speaker(),
            'current_emotion': self.get_current_emotion()
        }

# 사용 예시
def example_usage():
    """통합 오디오 처리 사용 예시"""
    
    # 콜백 함수 정의
    def on_speaker_change(speaker_info):
        print(f"화자 변경 감지: {speaker_info}")
    
    def on_emotion_change(emotion_info):
        print(f"감정 변경 감지: {emotion_info['current_emotion']} (신뢰도: {emotion_info['confidence']:.2f})")
    
    def on_vad_change(vad_info):
        print(f"음성 활동 감지: {vad_info}")
    
    def on_integrated_result(result):
        print(f"통합 결과: 화자={result.get('speaker_info')}, 감정={result.get('emotion_info', {}).get('current_emotion')}")
    
    # 통합 프로세서 생성
    processor = IntegratedAudioProcessor(
        speaker_model_path='speaker_model.pth',
        emotion_model_path='emotion_model.pth',
        enable_speaker_diarization=True,
        enable_emotion_recognition=True,
        enable_vad=True
    )
    
    # 콜백 설정
    processor.set_callbacks(
        on_speaker_change=on_speaker_change,
        on_emotion_change=on_emotion_change,
        on_vad_change=on_vad_change,
        on_integrated_result=on_integrated_result
    )
    
    # 처리 시작
    processor.start_processing()
    
    try:
        # 30초간 처리
        time.sleep(30)
        
        # 통합 분석 결과 출력
        analysis = processor.get_integrated_analysis(duration=30.0)
        print("통합 분석 결과:")
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
        
        # 결과 저장
        processor.save_results('audio_analysis_results.json')
        
    finally:
        # 처리 중지
        processor.stop_processing()

if __name__ == "__main__":
    example_usage() 