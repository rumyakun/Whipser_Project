import numpy as np
import threading
import queue
import time
from typing import List, Tuple, Optional, Callable
from collections import deque

# 기존 VAD 클래스들 import
from vad_processor import SimpleVAD, AdvancedVAD

# 딥러닝 VAD import (선택적)
try:
    from deep_learning_vad import DeepLearningVAD
    DL_VAD_AVAILABLE = True
except ImportError:
    DL_VAD_AVAILABLE = False
    print("딥러닝 VAD를 사용하려면 torch와 librosa를 설치하세요: pip install torch librosa")

class EnhancedVADProcessor:
    """향상된 VAD 프로세서 - 여러 VAD 방법을 통합"""
    
    def __init__(self, sample_rate: int = 16000, 
                 vad_type: str = "hybrid",
                 use_advanced_vad: bool = False,
                 callback: Optional[Callable] = None):
        """
        EnhancedVADProcessor 초기화
        
        Args:
            sample_rate: 샘플링 레이트
            vad_type: VAD 타입 ("simple", "advanced", "deep", "hybrid")
            use_advanced_vad: 고급 VAD 사용 여부
            callback: 음성 구간 감지 콜백 함수
        """
        self.sample_rate = sample_rate
        self.vad_type = vad_type
        self.callback = callback
        
        # VAD 인스턴스들 초기화
        self.vad_instances = {}
        
        # 기본 VAD
        self.vad_instances['simple'] = SimpleVAD(sample_rate)
        self.vad_instances['advanced'] = AdvancedVAD(sample_rate)
        
        # 딥러닝 VAD (가능한 경우)
        if vad_type in ["deep", "hybrid"] and DL_VAD_AVAILABLE:
            try:
                self.vad_instances['deep'] = DeepLearningVAD(sample_rate=sample_rate)
                print("딥러닝 VAD가 로드되었습니다.")
            except Exception as e:
                print(f"딥러닝 VAD 로드 실패: {e}")
                if vad_type == "deep":
                    vad_type = "advanced"
        
        # 현재 사용할 VAD 설정
        self.current_vad = self.vad_instances.get(vad_type, self.vad_instances['simple'])
        
        # 오디오 버퍼
        self.audio_buffer = []
        self.is_processing = False
        
        # 통계 정보
        self.stats = {
            'total_frames': 0,
            'speech_frames': 0,
            'silence_frames': 0,
            'detection_accuracy': 0.0
        }
        
    def set_vad_type(self, vad_type: str):
        """VAD 타입 변경"""
        if vad_type in self.vad_instances:
            self.current_vad = self.vad_instances[vad_type]
            self.vad_type = vad_type
            print(f"VAD 타입이 {vad_type}으로 변경되었습니다.")
        else:
            print(f"지원하지 않는 VAD 타입: {vad_type}")
    
    def start_processing(self):
        """VAD 처리 시작"""
        self.is_processing = True
        self.audio_buffer = []
        
        # 모든 VAD 인스턴스 초기화
        for vad in self.vad_instances.values():
            if hasattr(vad, 'reset'):
                vad.reset()
        
        print(f"VAD 처리 시작 - 타입: {self.vad_type}")
    
    def stop_processing(self):
        """VAD 처리 중지"""
        self.is_processing = False
        self.audio_buffer = []
        print("VAD 처리 중지")
    
    def process_audio_chunk(self, audio_chunk: np.ndarray):
        """오디오 청크 처리"""
        if not self.is_processing:
            return
        
        # 오디오 버퍼에 추가
        self.audio_buffer.extend(audio_chunk)
        
        # 하이브리드 모드인 경우 여러 VAD 결과를 결합
        if self.vad_type == "hybrid":
            segments = self._hybrid_vad_process(audio_chunk)
        else:
            # 단일 VAD 처리
            if hasattr(self.current_vad, 'process_audio_chunk'):
                segments = self.current_vad.process_audio_chunk(audio_chunk)
            else:
                segments = self.current_vad.process_audio(audio_chunk)
        
        # 음성 구간이 감지되면 콜백 호출
        for is_speech, audio_segment in segments:
            if is_speech and self.callback:
                self.callback(audio_segment)
        
        # 통계 업데이트
        self._update_stats(len(audio_chunk), len(segments))
    
    def _hybrid_vad_process(self, audio_chunk: np.ndarray) -> List[Tuple[bool, np.ndarray]]:
        """하이브리드 VAD 처리 - 여러 VAD 결과를 결합"""
        all_segments = []
        vad_results = {}
        
        # 각 VAD의 결과 수집
        for vad_name, vad_instance in self.vad_instances.items():
            try:
                if hasattr(vad_instance, 'process_audio_chunk'):
                    segments = vad_instance.process_audio_chunk(audio_chunk)
                else:
                    segments = vad_instance.process_audio(audio_chunk)
                
                vad_results[vad_name] = segments
                all_segments.extend(segments)
                
            except Exception as e:
                print(f"{vad_name} VAD 처리 오류: {e}")
        
        # 결과 결합 및 필터링
        combined_segments = self._combine_vad_results(vad_results)
        
        return combined_segments
    
    def _combine_vad_results(self, vad_results: dict) -> List[Tuple[bool, np.ndarray]]:
        """여러 VAD 결과를 결합"""
        if not vad_results:
            return []
        
        # 모든 음성 구간 수집
        speech_segments = []
        for vad_name, segments in vad_results.items():
            for is_speech, audio_segment in segments:
                if is_speech:
                    speech_segments.append({
                        'audio': audio_segment,
                        'vad_name': vad_name,
                        'confidence': self._get_vad_confidence(vad_name)
                    })
        
        # 중복 제거 및 병합
        merged_segments = self._merge_overlapping_segments(speech_segments)
        
        return [(True, segment['audio']) for segment in merged_segments]
    
    def _get_vad_confidence(self, vad_name: str) -> float:
        """VAD별 신뢰도 반환"""
        confidence_scores = {
            'simple': 0.6,
            'advanced': 0.8,
            'deep': 0.9
        }
        return confidence_scores.get(vad_name, 0.5)
    
    def _merge_overlapping_segments(self, segments: List[dict]) -> List[dict]:
        """중복되는 음성 구간 병합"""
        if not segments:
            return []
        
        # 시간 순으로 정렬
        segments.sort(key=lambda x: len(x['audio']), reverse=True)
        
        merged = []
        used_indices = set()
        
        for i, segment in enumerate(segments):
            if i in used_indices:
                continue
            
            current_audio = segment['audio']
            current_vads = [segment['vad_name']]
            used_indices.add(i)
            
            # 중복되는 구간 찾기
            for j, other_segment in enumerate(segments[i+1:], i+1):
                if j in used_indices:
                    continue
                
                other_audio = other_segment['audio']
                
                # 오디오 유사도 계산 (간단한 방법)
                similarity = self._calculate_audio_similarity(current_audio, other_audio)
                
                if similarity > 0.7:  # 70% 이상 유사하면 병합
                    current_audio = self._merge_audio_segments(current_audio, other_audio)
                    current_vads.append(other_segment['vad_name'])
                    used_indices.add(j)
            
            merged.append({
                'audio': current_audio,
                'vad_names': current_vads,
                'confidence': max(self._get_vad_confidence(vad) for vad in current_vads)
            })
        
        return merged
    
    def _calculate_audio_similarity(self, audio1: np.ndarray, audio2: np.ndarray) -> float:
        """두 오디오 세그먼트의 유사도 계산"""
        # 간단한 상관계수 기반 유사도
        min_len = min(len(audio1), len(audio2))
        if min_len == 0:
            return 0.0
        
        audio1_trimmed = audio1[:min_len]
        audio2_trimmed = audio2[:min_len]
        
        correlation = np.corrcoef(audio1_trimmed, audio2_trimmed)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def _merge_audio_segments(self, audio1: np.ndarray, audio2: np.ndarray) -> np.ndarray:
        """두 오디오 세그먼트 병합"""
        # 더 긴 세그먼트를 우선
        if len(audio1) >= len(audio2):
            return audio1
        else:
            return audio2
    
    def _update_stats(self, total_frames: int, speech_segments: int):
        """통계 정보 업데이트"""
        self.stats['total_frames'] += total_frames
        self.stats['speech_frames'] += speech_segments
        
        if self.stats['total_frames'] > 0:
            self.stats['detection_accuracy'] = (
                self.stats['speech_frames'] / self.stats['total_frames']
            )
    
    def get_current_audio_level(self) -> float:
        """현재 오디오 레벨 반환"""
        if not self.audio_buffer:
            return 0.0
        
        audio_data = np.array(self.audio_buffer)
        return np.sqrt(np.mean(audio_data**2))
    
    def get_speech_probability(self) -> float:
        """현재 음성 확률 반환"""
        if hasattr(self.current_vad, 'get_speech_probability'):
            return self.current_vad.get_speech_probability()
        elif hasattr(self.current_vad, 'is_speech'):
            return 1.0 if self.current_vad.is_speech else 0.0
        else:
            return 0.0
    
    def get_stats(self) -> dict:
        """VAD 통계 정보 반환"""
        return self.stats.copy()
    
    def set_confidence_threshold(self, threshold: float):
        """신뢰도 임계값 설정"""
        for vad_name, vad_instance in self.vad_instances.items():
            if hasattr(vad_instance, 'set_confidence_threshold'):
                vad_instance.set_confidence_threshold(threshold)
    
    def get_available_vad_types(self) -> List[str]:
        """사용 가능한 VAD 타입 목록 반환"""
        return list(self.vad_instances.keys())

class AdaptiveVADProcessor:
    """적응형 VAD 프로세서 - 환경에 따라 자동 조정"""
    
    def __init__(self, sample_rate: int = 16000, callback: Optional[Callable] = None):
        """
        AdaptiveVADProcessor 초기화
        
        Args:
            sample_rate: 샘플링 레이트
            callback: 음성 구간 감지 콜백 함수
        """
        self.sample_rate = sample_rate
        self.callback = callback
        
        # 여러 VAD 인스턴스
        self.vad_processors = {
            'simple': SimpleVAD(sample_rate),
            'advanced': AdvancedVAD(sample_rate)
        }
        
        if DL_VAD_AVAILABLE:
            try:
                self.vad_processors['deep'] = DeepLearningVAD(sample_rate=sample_rate)
            except:
                pass
        
        # 현재 사용할 VAD
        self.current_vad_name = 'simple'
        self.current_vad = self.vad_processors[self.current_vad_name]
        
        # 환경 분석을 위한 변수들
        self.noise_level_history = deque(maxlen=100)
        self.speech_detection_history = deque(maxlen=100)
        self.adaptation_threshold = 0.1
        
        # 오디오 버퍼
        self.audio_buffer = []
        self.is_processing = False
    
    def start_processing(self):
        """적응형 VAD 처리 시작"""
        self.is_processing = True
        self.audio_buffer = []
        
        for vad in self.vad_processors.values():
            if hasattr(vad, 'reset'):
                vad.reset()
        
        print("적응형 VAD 처리 시작")
    
    def stop_processing(self):
        """적응형 VAD 처리 중지"""
        self.is_processing = False
        self.audio_buffer = []
    
    def process_audio_chunk(self, audio_chunk: np.ndarray):
        """오디오 청크 처리 및 자동 VAD 선택"""
        if not self.is_processing:
            return
        
        # 오디오 버퍼에 추가
        self.audio_buffer.extend(audio_chunk)
        
        # 환경 분석
        noise_level = self._analyze_environment(audio_chunk)
        self.noise_level_history.append(noise_level)
        
        # VAD 자동 선택
        self._adapt_vad_selection()
        
        # 선택된 VAD로 처리
        if hasattr(self.current_vad, 'process_audio_chunk'):
            segments = self.current_vad.process_audio_chunk(audio_chunk)
        else:
            segments = self.current_vad.process_audio(audio_chunk)
        
        # 결과 처리
        for is_speech, audio_segment in segments:
            if is_speech and self.callback:
                self.callback(audio_segment)
        
        # 감지 히스토리 업데이트
        self.speech_detection_history.append(len(segments) > 0)
    
    def _analyze_environment(self, audio_chunk: np.ndarray) -> float:
        """환경 노이즈 레벨 분석"""
        # RMS 에너지 계산
        rms_energy = np.sqrt(np.mean(audio_chunk**2))
        
        # 스펙트럼 분석
        if len(audio_chunk) >= 1024:
            fft_data = np.fft.fft(audio_chunk[:1024])
            magnitude = np.abs(ffft_data[:512])
            spectral_centroid = np.sum(np.arange(512) * magnitude) / (np.sum(magnitude) + 1e-10)
        else:
            spectral_centroid = 0.0
        
        # 노이즈 레벨 계산 (에너지 + 스펙트럼 중심 주파수)
        noise_level = rms_energy * 0.7 + (spectral_centroid / 512) * 0.3
        
        return noise_level
    
    def _adapt_vad_selection(self):
        """환경에 따른 VAD 자동 선택"""
        if len(self.noise_level_history) < 10:
            return
        
        # 평균 노이즈 레벨 계산
        avg_noise = np.mean(list(self.noise_level_history))
        
        # 음성 감지 성공률 계산
        if len(self.speech_detection_history) > 0:
            speech_rate = np.mean(list(self.speech_detection_history))
        else:
            speech_rate = 0.0
        
        # VAD 선택 로직
        new_vad_name = self._select_best_vad(avg_noise, speech_rate)
        
        if new_vad_name != self.current_vad_name:
            self.current_vad_name = new_vad_name
            self.current_vad = self.vad_processors[new_vad_name]
            print(f"VAD가 {new_vad_name}으로 자동 변경되었습니다. (노이즈: {avg_noise:.3f}, 음성률: {speech_rate:.3f})")
    
    def _select_best_vad(self, noise_level: float, speech_rate: float) -> str:
        """최적의 VAD 선택"""
        available_vads = list(self.vad_processors.keys())
        
        if noise_level < 0.02:  # 조용한 환경
            if 'deep' in available_vads:
                return 'deep'
            else:
                return 'simple'
        
        elif noise_level < 0.05:  # 보통 환경
            if 'advanced' in available_vads:
                return 'advanced'
            else:
                return 'simple'
        
        else:  # 시끄러운 환경
            return 'simple'  # 가장 안정적인 기본 VAD
    
    def get_current_vad_info(self) -> dict:
        """현재 VAD 정보 반환"""
        return {
            'current_vad': self.current_vad_name,
            'available_vads': list(self.vad_processors.keys()),
            'noise_level': np.mean(list(self.noise_level_history)) if self.noise_level_history else 0.0,
            'speech_rate': np.mean(list(self.speech_detection_history)) if self.speech_detection_history else 0.0
        }
    
    def get_current_audio_level(self) -> float:
        """현재 오디오 레벨 반환"""
        if not self.audio_buffer:
            return 0.0
        
        audio_data = np.array(self.audio_buffer)
        return np.sqrt(np.mean(audio_data**2)) 