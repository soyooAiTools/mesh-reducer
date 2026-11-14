"""preview renders for dev/retopo_vh.py: textured side by side (left result, right source) and wireframes of the result"""
import os
import bpy
import numpy as np
from core import preview

# close-ups of the character: name -> (view, centre (world m), half size)
CLOSE = {'coat_back': ('back', (0, 0, 0.95), 0.75), 'head': ('front', (0, 0, 1.75), 0.3), 'head_back': ('back', (0, 0, 1.72), 0.3),
         'head_side': ('left', (0, 0.05, 1.72), 0.3), 'head_34': ('front34', (0, 0, 1.70), 0.27),
         'hand_L': ('front', (0.66, 0, 1.2), 0.2), 'hand_R': ('front', (-0.66, 0, 1.2), 0.2), 'torso': ('front', (0, 0, 1.32), 0.42),
         'legs': ('front', (0, 0, 0.5), 0.55), 'side': ('left', (0, 0.1, 1.2), 0.55)}


def _grey(o):
    m = bpy.data.materials.new('rt_grey')
    m.diffuse_color = (0.75, 0.75, 0.75, 1)
    c = o.copy()
    c.data = o.data.copy()
    bpy.context.scene.collection.objects.link(c)
    c.data.materials.clear()
    c.data.materials.append(m)
    for p in c.data.polygons:
        p.material_index = 0
    return c


def render_all(done, pv, stem, res=1300):
    for part, low, high in done:
        tag = '%s_%s' % (stem, part)
        g = _grey(low)
        if part == 'character':
            for v in ('front', 'back', 'left'):
                preview.render_views([low, high], os.path.join(pv, '%s_tex_%s.png' % (tag, v)), views=(v,), res=res, wire=False)
                preview.render_views([g], os.path.join(pv, '%s_wire_%s.png' % (tag, v)), views=(v,), res=1700, wire=True, textured=False, ref=low)
            for n, (v, c, h) in CLOSE.items():
                preview.render_views([g], os.path.join(pv, '%s_wire_%s.png' % (tag, n)), views=(v,), res=1300, wire=True, textured=False, zoom_box=(c, h))
                preview.render_views([low, high], os.path.join(pv, '%s_tex_%s.png' % (tag, n)), views=(v,), res=900, wire=False, zoom_box=(c, h))
        else:
            co = np.array([high.matrix_world @ v.co for v in high.data.vertices])
            lo, hi = co.min(0), co.max(0)
            ctr = tuple(((lo + hi) / 2).tolist())
            h = float((hi - lo).max() / 2 * 1.08)
            for v in ('front', 'back'):
                preview.render_views([low, high], os.path.join(pv, '%s_tex_%s.png' % (tag, v)), views=(v,), res=1100, wire=False, zoom_box=(ctr, h))
                preview.render_views([g, high], os.path.join(pv, '%s_wire_%s.png' % (tag, v)), views=(v,), res=1300, wire=True, textured=False, zoom_box=(ctr, h))
            preview.render_views([g], os.path.join(pv, '%s_wire_left.png' % tag), views=('left', 'top'), res=1100, wire=True, textured=False, zoom_box=(ctr, h))
            # detail: upper right quarter (crossbow: a limb with its hooks), result | source
            dc = (ctr[0] + 0.42 * h, ctr[1], ctr[2] + 0.32 * h)
            preview.render_views([g, high], os.path.join(pv, '%s_wire_detail.png' % tag), views=('front',), res=1200, wire=True, textured=False, zoom_box=(dc, 0.36 * h))
            low.hide_render = high.hide_render = False
            preview.render_views([low, high], os.path.join(pv, '%s_lit_detail.png' % tag), views=('front',), res=1000, wire=False, engine='CYCLES', zoom_box=(dc, 0.36 * h))
            low.hide_render = high.hide_render = True
        bpy.data.objects.remove(g)
        # lit comparison (Cycles): shows the baked normal map working
        low.hide_render = high.hide_render = False
        try:
            for v in ('front', 'back'):
                preview.render_views([low, high], os.path.join(pv, '%s_lit_%s.png' % (tag, v)), views=(v,), res=1000, wire=False, engine='CYCLES',
                                     **({} if part == 'character' else {'zoom_box': (ctr, h)}))
            if part == 'character':
                for n in ('head', 'head_34', 'head_back', 'head_side', 'torso', 'coat_back'):
                    v, c, h_ = CLOSE[n]
                    preview.render_views([low, high], os.path.join(pv, '%s_lit_%s.png' % (tag, n)), views=(v,), res=800, wire=False, engine='CYCLES', zoom_box=(c, h_))
        except Exception as ex:
            print('WARN lit render', ex)
        low.hide_render = high.hide_render = True
