"""previews for dev/skelretopo_vh.py (left = result, right = source): wire, textured, lit (Cycles, shows the baked
normal map), close-ups placed relative to the model's size, silhouette difference per view, posed wireframes."""
import os, math
import bpy
import numpy as np
from mathutils import Matrix, Vector
from core import preview


def _grey(o, name='sv_grey'):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (0.75, 0.75, 0.75, 1)
    c = o.copy()
    c.data = o.data.copy()
    bpy.context.scene.collection.objects.link(c)
    c.data.materials.clear()
    c.data.materials.append(m)
    for p in c.data.polygons:
        p.material_index = 0
    return c


def _bbox(o):
    co = np.array([(o.matrix_world @ v.co)[:] for v in o.data.vertices])
    return co.min(0), co.max(0)


def silhouettes(g1, g0, ref, tmp, res=420):
    out = {}
    for v in ('front', 'left', 'top', 'front34'):
        preview.render_views([g1, g0], tmp, views=(v,), res=res, wire=False, textured=False, ref=ref, bg=(0.0, 1.0, 0.0))
        im = preview._read_png(tmp)
        w = (im.shape[1] - 6) // 2
        a, b = im[:, :w, :3], im[:, w + 6:w + 6 + w, :3]
        ma = ~((a[..., 1] > 0.8) & (a[..., 0] < 0.3) & (a[..., 2] < 0.3))
        mb = ~((b[..., 1] > 0.8) & (b[..., 0] < 0.3) & (b[..., 2] < 0.3))
        any_ = (ma | mb).sum()
        out[v] = round(100.0 * float((ma ^ mb).sum()) / float(any_), 2) if any_ else None
    return out


def _posed_copy(o, name):
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
    c = bpy.data.objects.new(name, me)
    c.matrix_world = o.matrix_world.copy()
    bpy.context.scene.collection.objects.link(c)
    return c


def _rotate_bone(arm, name, axis_world, angle):
    pb = arm.pose.bones.get(name)
    if pb is None:
        return
    mw = arm.matrix_world
    ax = (mw.inverted().to_3x3() @ Vector(axis_world)).normalized()
    head = pb.matrix.translation.copy()
    R = Matrix.Translation(head) @ Matrix.Rotation(angle, 4, ax) @ Matrix.Translation(-head)
    pb.matrix = R @ pb.matrix
    bpy.context.view_layer.update()


def pose_check(low, high, arm, limbs, pv, tag, res=1000):
    """elbow 90, knee 90, shoulder raised: wire of the posed result | posed source"""
    if not limbs:
        return
    arm.data.pose_position = 'POSE'
    bpy.context.view_layer.update()
    head = {b.name: arm.matrix_world @ b.head_local for b in arm.data.bones}
    def d(a, b):
        return (head[b] - head[a]).normalized()
    up = Vector((0, 0, 1))
    poses = {}
    al, ll = limbs.get('arm_L'), limbs.get('leg_L')
    ar, lr = limbs.get('arm_R'), limbs.get('leg_R')
    front = None
    if ll and len(ll) >= 3:
        # the toes point to the front
        kids = [c for c in arm.data.bones[ll[2]].children]
        f = (arm.matrix_world @ kids[0].head_local - head[ll[2]]) if kids else Vector((0, -1, 0))
        f.z = 0
        front = f.normalized() if f.length > 1e-6 else Vector((0, -1, 0))
    front = front or Vector((0, -1, 0))
    if al and len(al) >= 2:
        poses['elbow90'] = [(al[1], d(al[0], al[1]).cross(front), math.radians(90))] + ([(ar[1], d(ar[0], ar[1]).cross(front), math.radians(90))] if ar else [])
        poses['shoulder_up'] = [(al[0], front, math.radians(50))] + ([(ar[0], front, math.radians(-50))] if ar else [])
    if ll and len(ll) >= 2:
        lat = up.cross(front)
        poses['knee90'] = [(ll[1], lat, math.radians(90))] + ([(lr[0], lat, math.radians(-60)), (lr[1], lat, math.radians(90))] if lr else [])
    zooms = {}
    if al and len(al) >= 3:
        # fingers curled (every bone below the hand 40 degrees about the line through the finger roots) + elbow 60
        hand = arm.data.bones[al[2]]
        roots = list(hand.children)
        if len(roots) >= 2:
            axis = (head[roots[-1].name] - head[roots[0].name])
            if axis.length > 1e-6:
                axis.normalize()
                rots = [(al[1], d(al[0], al[1]).cross(front), math.radians(60))]
                st = list(roots)
                while st:
                    b = st.pop(0)
                    rots.append((b.name, axis, math.radians(40)))
                    st.extend(b.children)
                poses['fingers'] = rots
                zooms['fingers'] = al[2]
    for name, rots in poses.items():
        for pb in arm.pose.bones:
            pb.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()
        for bone, axis, ang in rots:
            _rotate_bone(arm, bone, axis, ang)
        a, b = _posed_copy(low, 'pose_low'), _posed_copy(high, 'pose_high')
        ga, gb = _grey(a), _grey(b)
        zb = None
        if name in zooms:
            c = arm.matrix_world @ arm.pose.bones[zooms[name]].matrix.translation
            hh = max((arm.matrix_world @ Vector(x)).z for x in arm.bound_box) - min((arm.matrix_world @ Vector(x)).z for x in arm.bound_box)
            zb = (tuple(c), 0.09 * max(hh, 1e-6))
        for v in (('front', 'left') if name != 'knee90' else ('left', 'front34')):
            preview.render_views([ga, gb], os.path.join(pv, '%s_pose_%s_%s.png' % (tag, name, v)), views=(v,), res=res, wire=True, textured=False, ref=gb, zoom_box=zb)
        for o in (a, b, ga, gb):
            bpy.data.objects.remove(o)
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    arm.data.pose_position = 'REST'
    bpy.context.view_layer.update()


def render_all(done, pv, stem, report, arm, res=1200):
    for part, low, high, rep in done:
        tag = '%s_%s' % (stem, part)
        g, gh = _grey(low), _grey(high)
        lo, hi = _bbox(high)
        ctr = (lo + hi) / 2
        H = float(hi[2] - lo[2])
        rep['silhouette_diff_pct'] = silhouettes(g, gh, gh, os.path.join(pv, '_sil.png'))
        try:
            os.remove(os.path.join(pv, '_sil.png'))
        except OSError:
            pass
        if part == 'character':
            close = {'neck': ('front', (ctr[0], ctr[1], lo[2] + 0.84 * H), 0.17 * H), 'neck_side': ('left', (ctr[0], ctr[1], lo[2] + 0.84 * H), 0.2 * H),
                     'torso': ('front', (ctr[0], ctr[1], lo[2] + 0.64 * H), 0.2 * H), 'torso_back': ('back', (ctr[0], ctr[1], lo[2] + 0.64 * H), 0.2 * H),
                     'legs': ('front', (ctr[0], ctr[1], lo[2] + 0.24 * H), 0.27 * H), 'legs_side': ('left', (ctr[0], ctr[1], lo[2] + 0.2 * H), 0.24 * H),
                     'coat_back': ('back', (ctr[0], ctr[1], lo[2] + 0.42 * H), 0.45 * H), 'front34': ('front34', (ctr[0], ctr[1], lo[2] + 0.5 * H), 0.55 * H)}
            for v in ('front', 'back', 'left'):
                preview.render_views([low, high], os.path.join(pv, '%s_tex_%s.png' % (tag, v)), views=(v,), res=res, wire=False, ref=high)
                preview.render_views([g, gh], os.path.join(pv, '%s_wire_%s.png' % (tag, v)), views=(v,), res=res, wire=True, textured=False, ref=high)
            for n, (v, c, h) in close.items():
                preview.render_views([g, gh], os.path.join(pv, '%s_wire_%s.png' % (tag, n)), views=(v,), res=1000, wire=True, textured=False, zoom_box=(c, h))
                preview.render_views([low, high], os.path.join(pv, '%s_tex_%s.png' % (tag, n)), views=(v,), res=900, wire=False, zoom_box=(c, h))
            low.hide_render = high.hide_render = False
            try:
                for v in ('front', 'back'):
                    preview.render_views([low, high], os.path.join(pv, '%s_lit_%s.png' % (tag, v)), views=(v,), res=1000, wire=False, engine='CYCLES', ref=high)
                for n in ('neck', 'torso', 'legs', 'coat_back'):
                    v, c, h = close[n]
                    preview.render_views([low, high], os.path.join(pv, '%s_lit_%s.png' % (tag, n)), views=(v,), res=800, wire=False, engine='CYCLES', zoom_box=(c, h))
            except Exception as ex:
                print('WARN lit render', ex)
            low.hide_render = high.hide_render = True
            limbs = rep['build'].get('skeleton', {}).get('limbs', {})
            try:
                pose_check(low, high, arm, limbs, pv, tag)
            except Exception as ex:
                print('WARN pose check', ex)
        else:
            c = tuple(ctr.tolist())
            h = float((hi - lo).max() / 2 * 1.08)
            for v in ('front', 'left', 'top'):
                preview.render_views([low, high], os.path.join(pv, '%s_tex_%s.png' % (tag, v)), views=(v,), res=1000, wire=False, zoom_box=(c, h))
                preview.render_views([g, gh], os.path.join(pv, '%s_wire_%s.png' % (tag, v)), views=(v,), res=1000, wire=True, textured=False, zoom_box=(c, h))
            low.hide_render = high.hide_render = False
            try:
                preview.render_views([low, high], os.path.join(pv, '%s_lit_front.png' % tag), views=('front',), res=1000, wire=False, engine='CYCLES', zoom_box=(c, h))
            except Exception as ex:
                print('WARN lit render', ex)
            low.hide_render = high.hide_render = True
        bpy.data.objects.remove(g)
        bpy.data.objects.remove(gh)
