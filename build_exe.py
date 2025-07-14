#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실행 파일 빌드 스크립트
PyInstaller를 사용하여 Whisper 애플리케이션을 .exe 파일로 패키징
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_pyinstaller():
    """PyInstaller 설치"""
    print("PyInstaller를 설치합니다...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("PyInstaller 설치 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"PyInstaller 설치 실패: {e}")
        return False

def build_executable():
    """실행 파일 빌드"""
    print("실행 파일을 빌드합니다...")
    
    # PyInstaller 명령어 구성
    cmd = [
        "pyinstaller",
        "--onefile",  # 단일 실행 파일로 생성
        "--windowed",  # 콘솔 창 숨김 (GUI 앱용)
        "--name=WhisperApp",  # 실행 파일 이름
        "--icon=icon.ico",  # 아이콘 (있는 경우)
        "--add-data=whisper.cpp;whisper.cpp",  # whisper.cpp 폴더 포함
        "--hidden-import=sounddevice",
        "--hidden-import=soundfile", 
        "--hidden-import=librosa",
        "--hidden-import=scipy",
        "--hidden-import=torch",
        "--hidden-import=torchaudio",
        "--hidden-import=sklearn",
        "--hidden-import=matplotlib",
        "--hidden-import=seaborn",
        "--hidden-import=requests",
        "main.py"
    ]
    
    try:
        # PyInstaller 실행
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("빌드 성공!")
            print(f"실행 파일 위치: dist/WhisperApp.exe")
            return True
        else:
            print(f"빌드 실패: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"빌드 오류: {e}")
        return False

def create_installer():
    """설치 프로그램 생성 (선택사항)"""
    print("설치 프로그램을 생성합니다...")
    
    # NSIS 스크립트 생성 (NSIS가 설치된 경우)
    nsis_script = """
!include "MUI2.nsh"

Name "Whisper 음성 인식 앱"
OutFile "WhisperApp_Setup.exe"
InstallDir "$PROGRAMFILES\\WhisperApp"
RequestExecutionLevel admin

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_LANGUAGE "Korean"

Section "Main Application" SecMain
    SetOutPath "$INSTDIR"
    File "dist\\WhisperApp.exe"
    File /r "whisper.cpp"
    
    WriteUninstaller "$INSTDIR\\Uninstall.exe"
    
    CreateDirectory "$SMPROGRAMS\\WhisperApp"
    CreateShortCut "$SMPROGRAMS\\WhisperApp\\WhisperApp.lnk" "$INSTDIR\\WhisperApp.exe"
    CreateShortCut "$DESKTOP\\WhisperApp.lnk" "$INSTDIR\\WhisperApp.exe"
    
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\WhisperApp" \\
                     "DisplayName" "Whisper 음성 인식 앱"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\WhisperApp" \\
                     "UninstallString" "$INSTDIR\\Uninstall.exe"
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\\WhisperApp.exe"
    RMDir /r "$INSTDIR\\whisper.cpp"
    Delete "$INSTDIR\\Uninstall.exe"
    RMDir "$INSTDIR"
    
    Delete "$SMPROGRAMS\\WhisperApp\\WhisperApp.lnk"
    RMDir "$SMPROGRAMS\\WhisperApp"
    Delete "$DESKTOP\\WhisperApp.lnk"
    
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\WhisperApp"
SectionEnd
"""
    
    with open("installer.nsi", "w", encoding="utf-8") as f:
        f.write(nsis_script)
    
    print("NSIS 스크립트가 생성되었습니다: installer.nsi")
    print("NSIS가 설치되어 있다면 'makensis installer.nsi' 명령으로 설치 프로그램을 생성할 수 있습니다.")

def main():
    """메인 함수"""
    print("Whisper 앱 실행 파일 빌더")
    print("=" * 50)
    
    # PyInstaller 설치 확인 및 설치
    try:
        import PyInstaller
        print("PyInstaller가 이미 설치되어 있습니다.")
    except ImportError:
        if not install_pyinstaller():
            return 1
    
    # 빌드 실행
    if not build_executable():
        return 1
    
    print("\n빌드 완료!")
    print("=" * 50)
    print("생성된 파일:")
    print("- dist/WhisperApp.exe: 메인 실행 파일")
    print("\n사용자 배포 시 주의사항:")
    print("1. whisper.cpp 폴더와 모델 파일들이 실행 파일과 함께 있어야 합니다.")
    print("2. 처음 실행 시 모델 다운로드가 필요할 수 있습니다.")
    print("3. Windows Defender나 다른 백신이 오탐할 수 있습니다.")
    
    # 설치 프로그램 생성 여부 확인
    print("\n설치 프로그램을 생성하시겠습니까? (y/n): ", end="")
    response = input().lower().strip()
    
    if response in ['y', 'yes', '예']:
        create_installer()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 