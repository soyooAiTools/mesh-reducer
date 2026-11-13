"""Texture look-up used to make the reduction aware of what is painted on the
model: a loop that carries a colour border in the texture (strap, sole, hem,
eye line) is worth more than a loop on a uniformly coloured area."""
import os
import numpy as np


def load(image_or_path, max_size=512):
    """-> float32 array (h, w, 3), low-pass filtered and reduced to max_size, or None"""
    import bpy
    img = image_or_path
    try:
        if isinstance(img, str):
            if not img or not os.path.isfile(img):
                return None
            img = bpy.data.images.load(img, check_existing=True)
        w, h = img.size
        if w == 0 or h == 0:
            return None
        ch = img.channels
        a = np.empty(w * h * ch, dtype=np.float32)
        img.pixels.foreach_get(a)
        a = a.reshape(h, w, ch)[:, :, :3]
    except Exception:
        return None
    if a.shape[2] < 3:
        a = np.repeat(a[:, :, :1], 3, axis=2)
    # block average down to max_size (also acts as the low-pass filter)
    f = max(1, int(max(h, w) // max_size))
    if f > 1:
        hh, ww = (h // f) * f, (w // f) * f
        a = a[:hh, :ww].reshape(hh // f, f, ww // f, f, 3).mean(axis=(1, 3))
    return np.ascontiguousarray(a, dtype=np.float32)


def find_image(obj):
    """base colour image of the object's first material that has one"""
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.node_tree:
            continue
        nodes = mat.node_tree.nodes
        bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf is not None:
            inp = bsdf.inputs.get('Base Color')
            if inp is not None and inp.is_linked:
                n = inp.links[0].from_node
                if n.type == 'TEX_IMAGE' and n.image and n.image.size[0] > 0:
                    return n.image
        for n in nodes:
            if n.type == 'TEX_IMAGE' and n.image and n.image.size[0] > 0:
                return n.image
    return None


def sample(tex, uvs):
    """nearest texel colours for an (n,2) uv array (wrapping)"""
    h, w = tex.shape[:2]
    x = np.floor((uvs[:, 0] % 1.0) * w).astype(np.int64) % w
    y = np.floor((uvs[:, 1] % 1.0) * h).astype(np.int64) % h
    return tex[y, x]
