#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실시간 한국어 음성 인식 프로그램
Whisper.cpp를 기반으로 한 실시간 음성 인식 GUI 애플리케이션
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

def check_whisper_cpp():
    """Whisper.cpp 설치 확인"""
    whisper_path = Path("./whisper.cpp")
    
    if not whisper_path.exists():
        print("Whisper.cpp가 설치되지 않았습니다.")
        print("다음 명령어로 설치하세요:")
        print("git clone https://github.com/ggml-org/whisper.cpp.git")
        print("cd whisper.cpp")
        print("make")
        return False
    
    # 실행 파일 확인
    if platform.system() == "Windows":
        main_exe = whisper_path / "main.exe"
    else:
        main_exe = whisper_path / "main"
    
    if not main_exe.exists():
        print("Whisper.cpp가 빌드되지 않았습니다.")
        print("whisper.cpp 디렉토리에서 'make' 명령을 실행하세요.")
        return False
    
    return True

def check_models():
    """Whisper 모델 확인"""
    models_dir = Path("./whisper.cpp/models")
    
    if not models_dir.exists():
        print("모델 디렉토리가 없습니다.")
        return False
    
    # 기본 모델 확인
    base_model = models_dir / "ggml-base.bin"
    if not base_model.exists():
        print("기본 Whisper 모델이 없습니다.")
        print("다음 명령어로 다운로드하세요:")
        print("cd whisper.cpp")
        print("./models/download-ggml-model.sh base")
        return False
    
    return True

def download_model():
    """모델 다운로드"""
    print("Whisper 모델을 다운로드합니다...")
    
    try:
        # whisper.cpp 디렉토리로 이동
        os.chdir("./whisper.cpp")
        
        # 모델 다운로드 스크립트 실행
        if platform.system() == "Windows":
            cmd = ["./models/download-ggml-model.cmd", "base"]
        else:
            cmd = ["./models/download-ggml-model.sh", "base"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("모델 다운로드 완료!")
            return True
        else:
            print(f"모델 다운로드 실패: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"모델 다운로드 오류: {e}")
        return False
    finally:
        # 원래 디렉토리로 복귀
        os.chdir("..")

def main():
    """메인 함수"""
    print("실시간 한국어 음성 인식 프로그램")
    print("=" * 50)
    
    # Whisper.cpp 확인
    if not check_whisper_cpp():
        return 1
    
    # 모델 확인
    if not check_models():
        print("\n모델을 다운로드하시겠습니까? (y/n): ", end="")
        response = input().lower().strip()
        
        if response in ['y', 'yes', '예']:
            if not download_model():
                return 1
        else:
            print("모델이 필요합니다. 프로그램을 종료합니다.")
            return 1
    
    print("모든 준비가 완료되었습니다!")
    print("GUI를 시작합니다...")
    
    # GUI 시작
    try:
        from main_window import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"GUI 모듈을 불러올 수 없습니다: {e}")
        print("필요한 패키지를 설치하세요: pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"GUI 실행 오류: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 