# -*- coding: utf-8 -*-
"""Package the Blender add-on:  python build_addon.py [output.zip]
Copies ../core into the add-on and zips it (blender_manifest.toml at the zip
root = Blender 4.2+/5.x extension; install with Preferences > Get Extensions >
Install from Disk, or drag the zip into Blender)."""
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def build(out=None):
    out = out or os.path.join(ROOT, 'dist', 'mesh_reducer_addon.zip')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        src = os.path.join(HERE, 'mesh_reducer')
        for name in ('__init__.py', 'blender_manifest.toml'):
            z.write(os.path.join(src, name), name)
        for name in sorted(os.listdir(os.path.join(ROOT, 'core'))):
            if name.endswith('.py'):
                z.write(os.path.join(ROOT, 'core', name), 'core/' + name)
    return out


if __name__ == '__main__':
    print(build(sys.argv[1] if len(sys.argv) > 1 else None))
