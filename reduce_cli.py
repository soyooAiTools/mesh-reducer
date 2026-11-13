# -*- coding: utf-8 -*-
"""Simple command line front end (needs the `bpy` python module, or run it with
Blender:  blender -b --factory-startup --python reduce_cli.py -- in.fbx out.fbx --ratio 0.4)

    python3 reduce_cli.py in.fbx out.fbx --ratio 0.4 --mode character
    python3 reduce_cli.py in.fbx out.fbx --faces 1500 --mode character --renders previews --report r.json
"""
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import runner  # noqa: E402


def main():
    argv = sys.argv
    argv = argv[argv.index('--') + 1:] if '--' in argv else argv[1:]
    pos = [a for a in argv[:2] if not a.startswith('--')]
    if len(pos) < 2:
        print(__doc__)
        return 2
    return runner.main(['--in', pos[0], '--out', pos[1]] + argv[2:])


if __name__ == '__main__':
    sys.exit(main())
