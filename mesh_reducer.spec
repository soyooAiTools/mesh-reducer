# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec: one-folder, windowed build -> dist/MeshReducer/MeshReducer(.exe)
# runner.py, evaluate.py and core/ are bundled as plain data files so that the
# exe can hand them to the user's blender.exe (they run in Blender's Python,
# not in the exe's Python).
import os
import glob

root = os.path.abspath(os.getcwd())
datas = [(os.path.join(root, 'runner.py'), '.'), (os.path.join(root, 'evaluate.py'), '.')]
for f in glob.glob(os.path.join(root, 'core', '*.py')):
    datas.append((f, 'core'))

a = Analysis(
    [os.path.join(root, 'app', 'main.py')],
    pathex=[root],
    binaries=[],
    datas=datas,
    hiddenimports=['app.mainwindow', 'app.jobs', 'app.paths', 'app.blender_finder'],
    hookspath=[],
    runtime_hooks=[],
    excludes=['bpy', 'numpy', 'tkinter', 'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
              'PySide6.QtQml', 'PySide6.QtQuick', 'PySide6.Qt3DCore', 'PySide6.QtMultimedia',
              'PySide6.QtPdf', 'PySide6.QtCharts', 'PySide6.QtDataVisualization', 'PySide6.QtSql',
              'PySide6.QtTest', 'PySide6.QtBluetooth', 'PySide6.QtNetwork'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MeshReducer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='MeshReducer')
