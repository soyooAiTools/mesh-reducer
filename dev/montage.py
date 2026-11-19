"""Stack images vertically (each scaled to one width) into one JPEG for a single look.

    python3 dev/montage.py OUT.jpg a.png b.png ... [--width 1150]

Used for the key_*.jpg / key9_*.jpg review sheets of a skeleton-retopo run
(see dev/skelretopo_round.sh). Missing inputs are skipped with a warning.
"""
import os
import sys

import bpy  # noqa: F401  (image IO without extra deps)
import numpy as np


def montage(out, files, width=1150):
    rows = []
    for f in files:
        if not os.path.exists(f):
            print("montage: missing", f, file=sys.stderr)
            continue
        img = bpy.data.images.load(f)
        w, h = img.size
        a = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(a)
        a = a.reshape(h, w, 4)[::-1]
        nh = int(h * width / w)
        yi = (np.arange(nh) * h / nh).astype(int)
        xi = (np.arange(width) * w / width).astype(int)
        rows.append(a[yi][:, xi])
        rows.append(np.ones((6, width, 4), np.float32))
    if not rows:
        print("montage: nothing to write for", out, file=sys.stderr)
        return False
    m = np.concatenate(rows[:-1])[::-1]
    o = bpy.data.images.new("m", width, m.shape[0])
    o.pixels.foreach_set(np.ascontiguousarray(m).ravel())
    o.filepath_raw = out
    o.file_format = "JPEG"
    bpy.context.scene.render.image_settings.quality = 88
    o.save()
    return True


if __name__ == "__main__":
    args = sys.argv[1:]
    width = 1150
    if "--width" in args:
        i = args.index("--width")
        width = int(args[i + 1])
        del args[i:i + 2]
    if len(args) < 2:
        sys.exit(__doc__)
    sys.exit(0 if montage(args[0], args[1:], width) else 1)
