# -*- coding: utf-8 -*-
"""Where the bundled runner / core live (source tree or PyInstaller bundle)."""
import os
import sys
import json

APP_NAME = 'MeshReducer'


def resource_dir():
    base = getattr(sys, '_MEIPASS', None)
    if base and os.path.isfile(os.path.join(base, 'runner.py')):
        return base
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.dirname(here), here, os.path.dirname(os.path.abspath(sys.argv[0]))):
        if os.path.isfile(os.path.join(cand, 'runner.py')):
            return cand
    return os.path.dirname(here)


def runner_path():
    return os.path.join(resource_dir(), 'runner.py')


def config_dir():
    if sys.platform.startswith('win'):
        base = os.environ.get('APPDATA') or os.path.expanduser('~')
    elif sys.platform == 'darwin':
        base = os.path.expanduser('~/Library/Application Support')
    else:
        base = os.environ.get('XDG_CONFIG_HOME') or os.path.expanduser('~/.config')
    d = os.path.join(base, APP_NAME)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d


def load_settings():
    try:
        with open(os.path.join(config_dir(), 'settings.json'), 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(d):
    try:
        with open(os.path.join(config_dir(), 'settings.json'), 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
    except Exception:
        pass
