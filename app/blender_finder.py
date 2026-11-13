# -*- coding: utf-8 -*-
"""Locate an installed Blender (5.2 preferred) without starting a GUI."""
import os
import re
import sys
import glob
import shutil
import subprocess


def _ver_key(path):
    m = re.search(r'(\d+)\.(\d+)', os.path.basename(os.path.dirname(path)))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _steam_libraries():
    libs = []
    roots = [os.path.join(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'), 'Steam'),
             os.path.join(os.environ.get('ProgramFiles', r'C:\Program Files'), 'Steam'),
             os.path.expanduser('~/.steam/steam'), os.path.expanduser('~/.local/share/Steam'),
             os.path.expanduser('~/Library/Application Support/Steam')]
    try:
        import winreg
        for hive, key in ((winreg.HKEY_CURRENT_USER, r'Software\Valve\Steam'),
                          (winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\WOW6432Node\Valve\Steam'),
                          (winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Valve\Steam')):
            for name in ('SteamPath', 'InstallPath'):
                try:
                    with winreg.OpenKey(hive, key) as k:
                        roots.append(winreg.QueryValueEx(k, name)[0])
                except OSError:
                    pass
    except ImportError:
        pass
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        libs.append(r)
        vdf = os.path.join(r, 'steamapps', 'libraryfolders.vdf')
        try:
            with open(vdf, 'r', encoding='utf-8', errors='replace') as f:
                for m in re.finditer(r'"path"\s+"([^"]+)"', f.read()):
                    libs.append(m.group(1).replace('\\\\', '\\'))
        except OSError:
            pass
    return libs


def candidates():
    """Possible blender executables, best first (existing files only)."""
    out = []

    def add(p):
        if p and os.path.isfile(p) and os.path.normcase(os.path.abspath(p)) not in \
                [os.path.normcase(os.path.abspath(x)) for x in out]:
            out.append(p)
    add(os.environ.get('BLENDER_EXE', ''))
    if sys.platform.startswith('win'):
        found = []
        for env in ('ProgramFiles', 'ProgramW6432', 'ProgramFiles(x86)'):
            base = os.environ.get(env)
            if base:
                found += glob.glob(os.path.join(base, 'Blender Foundation', 'Blender*', 'blender.exe'))
        for drive in 'CDEFG':
            found += glob.glob(drive + r':\Program Files\Blender Foundation\Blender*\blender.exe')
            found += glob.glob(drive + r':\Blender*\blender.exe')
        # 5.2 first, then newest 5.x, then anything else
        found = sorted(set(found), key=lambda p: (_ver_key(p) != (5, 2), _ver_key(p)[0] != 5,
                                                  tuple(-x for x in _ver_key(p))))
        for p in found:
            add(p)
        try:
            import winreg
            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    with winreg.OpenKey(hive, r'SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\blender.exe') as k:
                        add(winreg.QueryValue(k, None))
                except OSError:
                    pass
            try:
                with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r'blendfile\shell\open\command') as k:
                    cmd = winreg.QueryValue(k, None)
                    m = re.match(r'\s*"([^"]+)"', cmd or '')
                    if m:
                        launcher = m.group(1)
                        add(os.path.join(os.path.dirname(launcher), 'blender.exe'))
            except OSError:
                pass
        except ImportError:
            pass
        for lib in _steam_libraries():
            add(os.path.join(lib, 'steamapps', 'common', 'Blender', 'blender.exe'))
        add(shutil.which('blender.exe') or '')
    elif sys.platform == 'darwin':
        add('/Applications/Blender.app/Contents/MacOS/Blender')
        for p in glob.glob('/Applications/Blender*.app/Contents/MacOS/Blender'):
            add(p)
        for lib in _steam_libraries():
            add(os.path.join(lib, 'steamapps', 'common', 'Blender', 'Blender.app', 'Contents', 'MacOS', 'Blender'))
        add(shutil.which('blender') or '')
    else:
        add(shutil.which('blender') or '')
        pats = ['/usr/bin/blender', '/usr/local/bin/blender', '/snap/bin/blender', '/opt/blender*/blender',
                os.path.expanduser('~/blender*/blender'), os.path.expanduser('~/Applications/blender*/blender'),
                os.path.expanduser('~/Downloads/blender*/blender')]
        for pat in pats:
            for p in sorted(glob.glob(pat), reverse=True):
                add(p)
        for lib in _steam_libraries():
            add(os.path.join(lib, 'steamapps', 'common', 'Blender', 'blender'))
    return out


def is_python_fallback(path):
    """a python interpreter that has the `bpy` module can stand in for Blender"""
    return os.path.basename(path).lower().startswith('python')


def no_window_kwargs():
    if sys.platform.startswith('win'):
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        return {'startupinfo': si, 'creationflags': 0x08000000}   # CREATE_NO_WINDOW
    return {}


def query_version(path, timeout=40):
    """-> 'x.y.z' or None"""
    if not path or not os.path.isfile(path):
        return None
    try:
        if is_python_fallback(path):
            cmd = [path, '-c', 'import bpy;print("Blender", bpy.app.version_string)']
        else:
            cmd = [path, '--version']
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                           **no_window_kwargs())
        text = p.stdout.decode('utf-8', errors='replace')
        m = re.search(r'Blender\s+(\d+\.\d+(?:\.\d+)?)', text)
        return m.group(1) if m else None
    except Exception:
        return None


def find_blender(saved=''):
    """-> (path, version) of the best usable Blender, ('' , None) if none."""
    tried = []
    if saved:
        v = query_version(saved)
        if v:
            return saved, v
    for p in candidates():
        if p in tried:
            continue
        tried.append(p)
        v = query_version(p)
        if v:
            return p, v
    return '', None
