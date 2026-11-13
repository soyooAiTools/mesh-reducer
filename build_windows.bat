@echo off
REM Build dist\MeshReducer\MeshReducer.exe (one-folder).  Needs Python 3.10+ on PATH.
chcp 65001 >nul
setlocal
cd /d "%~dp0"
python -m pip install --upgrade pip || goto :err
python -m pip install -r requirements.txt pyinstaller || goto :err
python -m PyInstaller --noconfirm mesh_reducer.spec || goto :err
echo.
echo Build OK: dist\MeshReducer\MeshReducer.exe
echo Copy the whole dist\MeshReducer folder to the target machine (Blender 5.2 must be installed there).
exit /b 0
:err
echo BUILD FAILED
exit /b 1
