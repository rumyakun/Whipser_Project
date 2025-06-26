import sys
import os
import threading
import time
from typing import Optional
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QProgressBar, QComboBox,
    QGroupBox, QGridLayout, QMessageBox, QFileDialog, QSlider,
    QCheckBox, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import QTimer, QThread, pyqtSignal, Qt
from PySide6.QtGui import QFont, QPalette, QColor, QPixmap, QIcon

from whisper_wrapper import WhisperWrapper, RealTimeWhisper
from audio_capture import AudioCapture, AudioVisualizer

class AudioProcessingThread(QThread):
    """오디오 처리를 위한 별도 스레드"""
    
    audio_level_changed = pyqtSignal(float)
    transcription_result = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, whisper_wrapper: WhisperWrapper, audio_capture: AudioCapture):
        super().__init__()
        self.whisper_wrapper = whisper_wrapper
        self.audio_capture = audio_capture
        self.real_time_whisper = RealTimeWhisper(whisper_wrapper)
        self.is_running = False
        
    def run(self):
        """스레드 실행"""
        self.is_running = True
        
        # 실시간 Whisper 시작
        self.real_time_whisper.start_recording()
        
        # 오디오 캡처 시작
        self.audio_capture.start_recording()
        
        # 오디오 레벨 모니터링 타이머
        timer = QTimer()
        timer.timeout.connect(self._update_audio_level)
        timer.start(100)  # 100ms마다 업데이트
        
        # 메인 루프
        while self.is_running:
            try:
                # Whisper 결과 확인
                results = self.real_time_whisper.get_results()
                for result in results:
                    if 'text' in result and result['text'].strip():
                        self.transcription_result.emit(result['text'])
                
                time.sleep(0.1)
                
            except Exception as e:
                self.error_occurred.emit(str(e))
                break
    
    def stop(self):
        """스레드 중지"""
        self.is_running = False
        self.real_time_whisper.stop_recording()
        self.audio_capture.stop_recording()
    
    def _update_audio_level(self):
        """오디오 레벨 업데이트"""
        level = self.audio_capture.get_audio_level()
        self.audio_level_changed.emit(level)

class MainWindow(QMainWindow):
    """메인 윈도우 클래스"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("실시간 한국어 음성 인식 프로그램")
        self.setGeometry(100, 100, 800, 600)
        
        # Whisper 설정
        self.whisper_wrapper = None
        self.audio_capture = None
        self.audio_thread = None
        
        # UI 초기화
        self.init_ui()
        self.init_whisper()
        
        # 타이머 설정
        self.audio_level_timer = QTimer()
        self.audio_level_timer.timeout.connect(self.update_audio_level)
        self.audio_level_timer.start(50)  # 50ms마다 업데이트
    
    def init_ui(self):
        """UI 초기화"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)
        
        # 제목
        title_label = QLabel("실시간 한국어 음성 인식")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 설정 그룹
        settings_group = QGroupBox("설정")
        settings_layout = QGridLayout(settings_group)
        
        # Whisper 모델 선택
        settings_layout.addWidget(QLabel("Whisper 모델:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["base", "small", "medium", "large"])
        settings_layout.addWidget(self.model_combo, 0, 1)
        
        # 언어 선택
        settings_layout.addWidget(QLabel("언어:"), 0, 2)
        self.language_combo = QComboBox()
        self.language_combo.addItems(["ko", "en", "ja", "zh"])
        self.language_combo.setCurrentText("ko")
        settings_layout.addWidget(self.language_combo, 0, 3)
        
        # 샘플링 레이트
        settings_layout.addWidget(QLabel("샘플링 레이트:"), 1, 0)
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 48000)
        self.sample_rate_spin.setValue(16000)
        self.sample_rate_spin.setSuffix(" Hz")
        settings_layout.addWidget(self.sample_rate_spin, 1, 1)
        
        # 청크 크기
        settings_layout.addWidget(QLabel("청크 크기:"), 1, 2)
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(512, 4096)
        self.chunk_size_spin.setValue(1024)
        self.chunk_size_spin.setSuffix(" 샘플")
        settings_layout.addWidget(self.chunk_size_spin, 1, 3)
        
        main_layout.addWidget(settings_group)
        
        # 오디오 레벨 표시
        audio_group = QGroupBox("오디오 레벨")
        audio_layout = QVBoxLayout(audio_group)
        
        self.audio_level_bar = QProgressBar()
        self.audio_level_bar.setRange(0, 100)
        self.audio_level_bar.setValue(0)
        audio_layout.addWidget(self.audio_level_bar)
        
        self.audio_level_label = QLabel("레벨: 0%")
        audio_layout.addWidget(self.audio_level_label)
        
        main_layout.addWidget(audio_group)
        
        # 제어 버튼
        control_layout = QHBoxLayout()
        
        self.start_button = QPushButton("녹음 시작")
        self.start_button.clicked.connect(self.start_recording)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("녹음 중지")
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        control_layout.addWidget(self.stop_button)
        
        self.clear_button = QPushButton("텍스트 지우기")
        self.clear_button.clicked.connect(self.clear_text)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        control_layout.addWidget(self.clear_button)
        
        main_layout.addLayout(control_layout)
        
        # 결과 텍스트 영역
        result_group = QGroupBox("인식 결과")
        result_layout = QVBoxLayout(result_group)
        
        self.result_text = QTextEdit()
        self.result_text.setFont(QFont("맑은 고딕", 12))
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        
        main_layout.addWidget(result_group)
        
        # 상태 표시줄
        self.status_label = QLabel("준비됨")
        self.status_label.setStyleSheet("color: gray; padding: 5px;")
        main_layout.addWidget(self.status_label)
    
    def init_whisper(self):
        """Whisper 초기화"""
        try:
            # Whisper.cpp 경로 설정
            whisper_path = "./whisper.cpp"
            model_name = self.model_combo.currentText()
            model_path = f"./whisper.cpp/models/ggml-{model_name}.bin"
            
            # Whisper 래퍼 초기화
            self.whisper_wrapper = WhisperWrapper(whisper_path, model_path)
            
            # 오디오 캡처 초기화
            sample_rate = self.sample_rate_spin.value()
            chunk_size = self.chunk_size_spin.value()
            
            self.audio_capture = AudioCapture(
                sample_rate=sample_rate,
                channels=1,
                chunk_size=chunk_size
            )
            
            self.status_label.setText("Whisper 초기화 완료")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"Whisper 초기화 실패: {str(e)}")
            self.status_label.setText("초기화 실패")
    
    def start_recording(self):
        """녹음 시작"""
        if not self.whisper_wrapper:
            QMessageBox.warning(self, "경고", "Whisper가 초기화되지 않았습니다.")
            return
        
        try:
            # 오디오 처리 스레드 시작
            self.audio_thread = AudioProcessingThread(self.whisper_wrapper, self.audio_capture)
            self.audio_thread.audio_level_changed.connect(self.update_audio_level)
            self.audio_thread.transcription_result.connect(self.add_transcription_result)
            self.audio_thread.error_occurred.connect(self.handle_error)
            self.audio_thread.start()
            
            # UI 상태 변경
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("녹음 중...")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"녹음 시작 실패: {str(e)}")
    
    def stop_recording(self):
        """녹음 중지"""
        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread.wait()
            self.audio_thread = None
        
        # UI 상태 변경
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("녹음 중지됨")
    
    def update_audio_level(self):
        """오디오 레벨 업데이트"""
        if self.audio_capture and self.audio_capture.is_recording:
            level = self.audio_capture.get_audio_level()
            level_percent = int(level * 100)
            
            self.audio_level_bar.setValue(level_percent)
            self.audio_level_label.setText(f"레벨: {level_percent}%")
    
    def add_transcription_result(self, text: str):
        """음성 인식 결과 추가"""
        current_text = self.result_text.toPlainText()
        if current_text:
            current_text += "\n"
        current_text += text
        
        self.result_text.setPlainText(current_text)
        
        # 스크롤을 맨 아래로
        scrollbar = self.result_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_text(self):
        """텍스트 지우기"""
        self.result_text.clear()
    
    def handle_error(self, error_msg: str):
        """오류 처리"""
        QMessageBox.warning(self, "오류", f"음성 인식 오류: {error_msg}")
        self.stop_recording()
    
    def closeEvent(self, event):
        """창 닫기 이벤트"""
        self.stop_recording()
        event.accept()

def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    
    # 애플리케이션 스타일 설정
    app.setStyle('Fusion')
    
    # 다크 테마 설정
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    # 메인 윈도우 생성 및 표시
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 