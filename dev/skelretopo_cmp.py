"""dev/skelretopo_cmp.py OURS_DIR MANUAL.json OUT_DIR [--no-renders]

Objective comparison of a dev/skelretopo_vh.py result with a hand made low model of the same character.
OURS_DIR holds report.json and previews/skel_debug.blend (results + sources).  MANUAL.json:
  {"character": {"fbx": "...", "textures": {"<object name>": "diffuse image", ...}}, "axe": {...}, ...}
(paths relative to the json).  The manual meshes are static, in their own units / place: they are aligned
to the source by a similarity ICP (weapons: also rotation, started from the 4 proper PCA flips).

Writes into OUT_DIR: cmp_<part>_{wire,tex}.png (manual | ours | original, rows front / left / back),
close-ups cmp_<part>_<region>_{wire,tex}.png, and adds 'manual_comparison' to OURS_DIR/report.json:
triangles per region (by the nearest source polygon: head/hat, torso, arms, hands, legs, feet, cloth, hair,
other), silhouette difference to the source per view (own rasteriser; 'geo' = geometry only, 'alpha' = with
the texture alpha cut out), Hausdorff / mean distance to the source."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from core import io_util, preview, metrics, retopo, skelretopo

REGIONS = ['head_hat', 'torso', 'arms', 'hands', 'legs', 'feet', 'cloth', 'hair', 'other']


# ---------------------------------------------------------------- helpers
def world_tris(o):
    """-> verts (n,3) world, tris (m,3), tri uv (m,3,2) or None, tri material index"""
    me = o.data
    mw = np.array(o.matrix_world)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    me.calc_loop_triangles()
    lt = me.loop_triangles
    tri = np.empty(len(lt) * 3, dtype=np.int64)
    lt.foreach_get('vertices', tri)
    lo = np.empty(len(lt) * 3, dtype=np.int64)
    lt.foreach_get('loops', lo)
    mat = np.empty(len(lt), dtype=np.int64)
    lt.foreach_get('material_index', mat)
    uv = None
    if me.uv_layers.active:
        u = np.empty(len(me.loops) * 2)
        me.uv_layers.active.data.foreach_get('uv', u)
        uv = u.reshape(-1, 2)[lo.reshape(-1, 3)]
    return co, tri.reshape(-1, 3), uv, mat


def sample(o, n=30000, seed=5):
    co, tri, _uv, _m = world_tris(o)
    a, b, c = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
    ar = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    ti = rng.choice(len(tri), size=n, p=ar / ar.sum())
    u, v = rng.random(n), rng.random(n)
    f = u + v > 1
    u[f], v[f] = 1 - u[f], 1 - v[f]
    return a[ti] + (b[ti] - a[ti]) * u[:, None] + (c[ti] - a[ti]) * v[:, None]


def bvh_of(o):
    co, tri, _uv, _m = world_tris(o)
    return BVHTree.FromPolygons([tuple(p) for p in co], [tuple(int(i) for i in t) for t in tri]), co, tri


def nearest(bvh, P):
    out = np.zeros((len(P), 3))
    idx = np.zeros(len(P), dtype=np.int64)
    for k, p in enumerate(P):
        loc, _n, i, _d = bvh.find_nearest(Vector(p))
        out[k] = loc
        idx[k] = i
    return out, idx


def icp(P, bvh, rot=False, iters=25, trim=0.85, scale=True):
    """similarity that moves the points P onto the surface `bvh` -> 4x4, trimmed mean distance"""
    M = np.eye(4)
    Q = P.copy()
    d = None
    for _ in range(iters):
        T, _i = nearest(bvh, Q)
        d = np.linalg.norm(T - Q, axis=1)
        keep = d <= np.quantile(d, trim)
        A, Bt = Q[keep], T[keep]
        ca, cb = A.mean(0), Bt.mean(0)
        A0, B0 = A - ca, Bt - cb
        if rot:
            U, S, Vt = np.linalg.svd(A0.T @ B0)
            D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
            R = Vt.T @ D @ U.T
            s = float((S * np.diag(D)).sum() / (A0 ** 2).sum()) if scale else 1.0
        else:
            R = np.eye(3)
            s = float((A0 * B0).sum() / (A0 ** 2).sum())
        step = np.eye(4)
        step[:3, :3] = s * R
        step[:3, 3] = cb - s * R @ ca
        M = step @ M
        Q = P @ M[:3, :3].T + M[:3, 3]
    return M, float(np.mean(np.sort(d)[:int(len(d) * trim)]))


def align(objs, target, rot):
    """moves the objects (as one rigid group) onto `target`"""
    P = np.concatenate([sample(o, 6000) for o in objs])
    T = sample(target, 20000)
    bvh, _c, _t = bvh_of(target)
    best = None
    if rot:
        cp, Vp = skelretopo._pca(P)
        ct, Vt = skelretopo._pca(T)
        sp = np.subtract(*np.percentile((P - cp) @ Vp[0], [99.5, 0.5]))
        stt = np.subtract(*np.percentile((T - ct) @ Vt[0], [99.5, 0.5]))
        bvh_p = BVHTree.FromPolygons([tuple(p) for o in objs for p in world_tris(o)[0]],
                                     [tuple(int(i) + off for i in t) for o, off in zip(objs, np.cumsum([0] + [len(o.data.vertices) for o in objs[:-1]])) for t in world_tris(o)[1]])
        for sg in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            Fp = np.array([Vp[0], Vp[1], np.cross(Vp[0], Vp[1])])
            Ft = np.array([Vt[0] * sg[0], Vt[1] * sg[1], np.cross(Vt[0] * sg[0], Vt[1] * sg[1])])
            R = Ft.T @ Fp
            M0 = np.eye(4)
            M0[:3, :3] = R * (stt / sp)
            M0[:3, 3] = ct - M0[:3, :3] @ cp
            M1, d = icp(P @ M0[:3, :3].T + M0[:3, 3], bvh, rot=True, scale=False)      # the size is that of the main axis
            M = M1 @ M0
            # both directions: the source must be covered by the manual mesh too
            Mi = np.linalg.inv(M)
            back = nearest(bvh_p, T[::4] @ Mi[:3, :3].T + Mi[:3, 3])[0] @ M[:3, :3].T + M[:3, 3]
            d = d + float(np.mean(np.linalg.norm(back - T[::4], axis=1)))
            if best is None or d < best[1]:
                best = (M, d)
    else:
        s = np.ptp(T[:, 2]) / np.ptp(P[:, 2])
        M0 = np.eye(4)
        M0[:3, :3] *= s
        M0[:3, 3] = np.median(T, axis=0) - s * np.median(P, axis=0)
        M0[2, 3] = T[:, 2].min() - s * P[:, 2].min()
        M1, d = icp(P @ M0[:3, :3].T + M0[:3, 3], bvh, rot=False)
        best = (M1 @ M0, d)
    M = Matrix(best[0].tolist())
    for o in objs:
        o.matrix_world = M @ o.matrix_world
    bpy.context.view_layer.update()
    return best[1]


def join(objs, name):
    for o in bpy.context.scene.objects:
        o.select_set(False)
    for o in objs:
        o.hide_set(False)
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    o = objs[0]
    o.name = name
    return o


def tex_material(name, path):
    img = bpy.data.images.load(path, check_existing=True)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    node = nt.nodes.new('ShaderNodeTexImage')
    node.image = img
    nt.nodes.active = node
    bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    nt.links.new(node.outputs['Color'], bsdf.inputs['Base Color'])
    if img.channels == 4:
        px = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
        if float(px[3::4].min()) < 0.5:
            nt.links.new(node.outputs['Alpha'], bsdf.inputs['Alpha'])
            try:
                mat.surface_render_method = 'DITHERED'
            except Exception:
                pass
    return mat


def alpha_of(mat, cache={}):
    """alpha channel (h, w) top row last, or None when the material is opaque"""
    if mat is None or not mat.use_nodes:
        return None
    if mat.name in cache:
        return cache[mat.name]
    out = None
    b = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if b is not None and b.inputs['Alpha'].is_linked:
        n = b.inputs['Alpha'].links[0].from_node
        if n.type == 'TEX_IMAGE' and n.image and n.image.size[0] > 0:
            img = n.image
            px = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
            img.pixels.foreach_get(px)
            out = px.reshape(img.size[1], img.size[0], 4)[..., 3]
    cache[mat.name] = out
    return out


VIEW_AXES = {'front': ((1, 0, 0), (0, 0, 1)), 'left': ((0, 1, 0), (0, 0, 1)), 'top': ((1, 0, 0), (0, 1, 0)),
             'front34': ((0.7071, 0.7071, 0), (0, 0, 1))}


def raster(o, ex, ey, c, half, res=384, use_alpha=True):
    """orthographic coverage mask of the object (projection on ex, ey around c, +-half)"""
    co, tri, uv, mat = world_tris(o)
    x = ((co - c) @ np.array(ex) / half * 0.5 + 0.5) * res
    y = ((co - c) @ np.array(ey) / half * 0.5 + 0.5) * res
    alphas = [alpha_of(s.material) if use_alpha else None for s in o.material_slots] or [None]
    mask = np.zeros((res, res), dtype=bool)
    X, Y = x[tri], y[tri]
    x0 = np.clip(np.floor(X.min(1)).astype(int), 0, res - 1)
    x1 = np.clip(np.ceil(X.max(1)).astype(int), 0, res - 1)
    y0 = np.clip(np.floor(Y.min(1)).astype(int), 0, res - 1)
    y1 = np.clip(np.ceil(Y.max(1)).astype(int), 0, res - 1)
    den = (Y[:, 1] - Y[:, 2]) * (X[:, 0] - X[:, 2]) + (X[:, 2] - X[:, 1]) * (Y[:, 0] - Y[:, 2])
    for k in range(len(tri)):
        if abs(den[k]) < 1e-9 or X[k].max() < 0 or X[k].min() > res or Y[k].max() < 0 or Y[k].min() > res:
            continue
        gx, gy = np.meshgrid(np.arange(x0[k], x1[k] + 1) + 0.5, np.arange(y0[k], y1[k] + 1) + 0.5)
        a = ((Y[k, 1] - Y[k, 2]) * (gx - X[k, 2]) + (X[k, 2] - X[k, 1]) * (gy - Y[k, 2])) / den[k]
        b = ((Y[k, 2] - Y[k, 0]) * (gx - X[k, 2]) + (X[k, 0] - X[k, 2]) * (gy - Y[k, 2])) / den[k]
        ins = (a >= 0) & (b >= 0) & (a + b <= 1)
        if not ins.any():
            continue
        A = alphas[min(mat[k], len(alphas) - 1)]
        if A is not None and uv is not None:
            w = 1 - a - b
            uu = a * uv[k, 0, 0] + b * uv[k, 1, 0] + w * uv[k, 2, 0]
            vv = a * uv[k, 0, 1] + b * uv[k, 1, 1] + w * uv[k, 2, 1]
            px = np.clip((uu % 1.0 * A.shape[1]).astype(int), 0, A.shape[1] - 1)
            py = np.clip((vv % 1.0 * A.shape[0]).astype(int), 0, A.shape[0] - 1)
            ins &= A[py, px] > 0.5
        mask[y0[k]:y1[k] + 1, x0[k]:x1[k] + 1] |= ins
    return mask


def silhouettes(objs, ref, res=384, axes=None, dump=None):
    """{name: {view: {'geo': %, 'alpha': %}}}: XOR / union against the reference (the source with its own alpha).
    dump: path prefix; writes <dump>_<view>.png, one panel per object: yellow = both, red = only the low
    model, green = only the source (with alpha)"""
    co, _t, _u, _m = world_tris(ref)
    c = (co.min(0) + co.max(0)) / 2
    half = float(np.linalg.norm(np.ptp(co, axis=0))) * 0.52
    out = {n: {} for n in objs}
    for v, (ex, ey) in (axes or VIEW_AXES).items():
        mr = raster(ref, ex, ey, c, half, res)
        panels = []
        for n, o in objs.items():
            r = {}
            for key, ua in (('geo', False), ('alpha', True)):
                m = raster(o, ex, ey, c, half, res, ua)
                r[key] = round(100.0 * float((m ^ mr).sum()) / max(float((m | mr).sum()), 1.0), 2)
            out[n][v] = r
            img = np.zeros((res, res, 4), np.float32)
            img[..., :3] = 0.12
            img[..., 3] = 1
            img[m & mr, :3] = (0.85, 0.8, 0.3)
            img[m & ~mr, :3] = (0.9, 0.15, 0.15)
            img[~m & mr, :3] = (0.15, 0.8, 0.25)
            panels.append(img[::-1])
            panels.append(np.ones((res, 4, 4), np.float32))
        if dump:
            preview.save_png(np.concatenate(panels[:-1], axis=1), '%s_%s.png' % (dump, v))
    return out


def geometry(high, low):
    """Hausdorff / mean distance in % of the source diagonal; both in world space"""
    tmp = []
    for o in (high, low):
        me = o.data.copy()
        me.transform(o.matrix_world)
        tmp.append(me)
    g = metrics.evaluate(tmp[0], tmp[1])['geometry']
    for me in tmp:
        bpy.data.meshes.remove(me)
    return {'hausdorff_pct': g['hausdorff_pct'], 'hausdorff_high_to_low_pct': g['hausdorff_orig_to_result_pct'],
            'hausdorff_low_to_high_pct': g['hausdorff_result_to_orig_pct'], 'mean_pct': g['mean_dist_pct'], 'p99_pct': g['p99_dist_pct']}


def region_map(high, arm):
    """region index of every polygon of the source character, by the skeleton analysis of core.skelretopo"""
    hs = skelretopo.High(high, n_samples=60000)
    sk = skelretopo.analyse_skeleton(arm, hs, log=lambda *a: None)
    owners, own = skelretopo.face_owners(sk, hs)
    H = hs.H
    reg = np.full(len(own), REGIONS.index('other'))

    def sub(b):
        out, st = set(), [b]
        while st:
            x = st.pop()
            out.add(x)
            st.extend(sk.kids[x])
        return out

    zsh = float(np.mean([c.joints[0][2] for c in sk.arms])) if sk.arms else hs.lo[2] + 0.8 * H
    head = set()
    for b in sk.torso.core:
        if sk.head[b][2] >= zsh - 0.02 * H:
            head |= {x for x in sub(b) if sk.owner.get(x) == 'torso'}
    for i, n in enumerate(owners):
        m = own == i
        if n == 'torso':
            reg[m] = np.where(np.isin(sk.pdom[m], list(head)), REGIONS.index('head_hat'), REGIONS.index('torso'))
        elif n.startswith('arm') or n.startswith('leg'):
            lc = next(l for l in sk.legs + sk.arms if l.name == n)
            end = sub(lc.core[-1])
            a, b = ('arms', 'hands') if n.startswith('arm') else ('legs', 'feet')
            reg[m] = np.where(np.isin(sk.pdom[m], list(end)), REGIONS.index(b), REGIONS.index(a))
        else:
            ar = hs.poly_area[m]
            if ar.sum() <= 0:
                continue
            ks = np.flatnonzero(m)
            P = hs.pts[np.isin(hs.pt_poly, ks)]
            if len(P) < 20:
                continue
            ext = float(np.ptp(P, axis=0).max())
            if np.median(P[:, 2]) > zsh:
                reg[m] = REGIONS.index('hair')
            elif ar.sum() > 0.015 * hs.area and ext > 0.15 * H:
                reg[m] = REGIONS.index('cloth')
    # polygon index of the High -> index of the mesh polygon
    full = np.full(len(high.data.polygons), REGIONS.index('other'))
    full[hs.poly_index] = reg
    return full, hs


def region_tris(o, bvh_high, tri_poly_high, reg):
    co, tri, _uv, _m = world_tris(o)
    cen = co[tri].mean(1)
    _loc, idx = nearest(bvh_high, cen)
    r = reg[tri_poly_high[idx]]
    return {REGIONS[i]: int((r == i).sum()) for i in range(len(REGIONS))}, r


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    ours_dir, man_json, out = argv[0], argv[1], argv[2]
    renders = '--no-renders' not in sys.argv
    os.makedirs(out, exist_ok=True)
    spec = json.load(open(man_json))
    base = os.path.dirname(os.path.abspath(man_json))
    rep_path = os.path.join(ours_dir, 'report.json')
    report = json.load(open(rep_path))
    bpy.ops.wm.open_mainfile(filepath=os.path.join(ours_dir, 'previews', 'skel_debug.blend'))
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    highs = [o for o in bpy.data.objects if o.type == 'MESH' and o.name.startswith('high_')]
    stem = os.path.splitext(report['source'])[0]
    res = {}
    for part, sp in spec.items():
        low = bpy.data.objects.get('%s_%s' % (stem, part))
        if low is None or part not in report['parts']:
            continue
        lc = np.array(preview._world_bbox(low)).mean(0)
        high = min(highs, key=lambda h: float(np.linalg.norm(np.array(preview._world_bbox(h)).mean(0) - lc)))
        before = set(bpy.data.objects)
        bpy.ops.import_scene.fbx(filepath=os.path.join(base, sp['fbx']))
        mobjs = [o for o in bpy.data.objects if o not in before and o.type == 'MESH']
        for o in mobjs:
            t = sp.get('textures', {}).get(o.name)
            if t:
                mat = tex_material('manual_' + o.name, os.path.join(base, t))
                o.data.materials.clear()
                o.data.materials.append(mat)
        tris_by_obj = {o.name: sum(len(p.vertices) - 2 for p in o.data.polygons) for o in mobjs}
        d = align(mobjs, high, rot=part != 'character')
        mcopies = []
        for o in mobjs:
            cpy = o.copy()
            cpy.data = o.data.copy()
            bpy.context.scene.collection.objects.link(cpy)
            cpy.hide_render = True
            mcopies.append((o.name, cpy))
        man = join(mobjs, 'manual_' + part)
        for o in (low, high, man):
            o.hide_render = False
            o.hide_set(False)
        r = {'manual_objects': tris_by_obj, 'align_trimmed_mean_dist': round(d, 5),
             'tris': {'manual': sum(len(p.vertices) - 2 for p in man.data.polygons), 'ours': sum(len(p.vertices) - 2 for p in low.data.polygons),
                      'source': sum(len(p.vertices) - 2 for p in high.data.polygons)},
             'faces': {n: {'quads': sum(len(p.vertices) == 4 for p in o.data.polygons), 'tris': sum(len(p.vertices) == 3 for p in o.data.polygons),
                           'ngons': sum(len(p.vertices) > 4 for p in o.data.polygons)} for n, o in (('manual', man), ('ours', low))}}
        axes = None
        if part != 'character':           # a prop: seen along its own principal axes
            c_, V_ = skelretopo._pca(sample(high, 8000))
            e3_ = np.cross(V_[0], V_[1])
            axes = {'flat': (tuple(V_[1]), tuple(V_[0])), 'side': (tuple(e3_), tuple(V_[0])), 'end': (tuple(V_[1]), tuple(e3_))}
        r['silhouette_diff_pct'] = silhouettes({'manual': man, 'ours': low}, high, axes=axes, dump=os.path.join(out, 'sil_%s' % part))
        r['geometry'] = {'manual': geometry(high, man), 'ours': geometry(high, low)}
        zoom = {}
        if part == 'character':
            reg, hs = region_map(high, arm)
            bvh, _c, _t = bvh_of(high)
            high.data.calc_loop_triangles()
            tp = np.array([t.polygon_index for t in high.data.loop_triangles])
            r['region_tris'] = {}
            for n, o in [('manual', man), ('ours', low), ('source', high)] + [('manual:' + n_, c_) for n_, c_ in mcopies]:
                r['region_tris'][n], _ = region_tris(o, bvh, tp, reg)
            preg = reg[hs.poly_index][hs.pt_poly]
            H = hs.H

            def box(mask, pad=1.25, lo_=0.06):
                P = hs.pts[mask]
                a, b = np.percentile(P, 0.5, axis=0), np.percentile(P, 99.5, axis=0)
                return tuple(((a + b) / 2).tolist()), max(float((b - a).max()) / 2 * pad, lo_ * H)

            left = hs.pts[:, 0] > np.median(hs.pts[:, 0])
            R = REGIONS.index
            zoom = {'hat': ('front', box(preg == R('head_hat'))), 'hat_side': ('left', box(preg == R('head_hat'))),
                    'hat_34': ('front34', box(preg == R('head_hat'))),
                    'hand': ('front', box((preg == R('hands')) & left, 1.5)), 'hand_top': ('top', box((preg == R('hands')) & left, 1.5)),
                    'hand_side': ('left', box((preg == R('hands')) & left, 1.5)),
                    'feather': ('front34', box((preg == R('head_hat')) & (hs.pts[:, 2] > np.percentile(hs.pts[preg == R('head_hat'), 2], 70)), 1.3)),
                    'feather_side': ('left', box((preg == R('head_hat')) & (hs.pts[:, 2] > np.percentile(hs.pts[preg == R('head_hat'), 2], 70)), 1.3)),
                    'bracer': ('front', box((preg == R('arms')) & left)), 'bracer_top': ('top', box((preg == R('arms')) & left)),
                    'boots': ('front', box(preg == R('feet'), 1.3)), 'boots_side': ('left', box(preg == R('feet'), 1.3)),
                    'legs': ('front', box((preg == R('legs')) | (preg == R('feet')), 1.1)),
                    'torso': ('front', box(preg == R('torso'), 1.0)),
                    'cape_back': ('back', box(preg == R('cloth'), 1.05)), 'cape_side': ('left', box(preg == R('cloth'), 1.05)),
                    'cape_34': ('back34', box(preg == R('cloth'), 1.05))}
            # round 5 (artist feedback 3): neck / hair, top of the cape, hand from below, quiver
            hz = hs.pts[preg == R('head_hat'), 2]
            tz = hs.pts[preg == R('torso')]
            if len(hz) and len(tz):
                nc_ = (float(np.median(tz[:, 0])), float(np.median(tz[:, 1])), float(np.percentile(hz, 3)))
                zoom.update({'neck_back': ('back', (nc_, 0.13 * H)), 'neck_34': ('back34', (nc_, 0.13 * H)), 'neck_side': ('left', (nc_, 0.13 * H)),
                             'cape_top': ('back', ((nc_[0], nc_[1], nc_[2] - 0.07 * H), 0.2 * H)),
                             'cape_top_34': ('back34', ((nc_[0], nc_[1], nc_[2] - 0.07 * H), 0.2 * H))})
            zoom['hand_bottom'] = ('bottom', box((preg == R('hands')) & left, 1.5))
            # round 9 (artist feedback 5): boots from above / below, one boot from the side, the fingers close up
            zoom['boots_top'] = ('top', box(preg == R('feet'), 1.3))
            zoom['boots_bottom'] = ('bottom', box(preg == R('feet'), 1.3))
            zoom['boot_side'] = ('left', box((preg == R('feet')) & left, 1.25))
            zoom['boot_front'] = ('front', box((preg == R('feet')) & left, 1.25))
            zoom['fingers_top'] = ('top', box((preg == R('hands')) & left, 0.95))
            zoom['fingers_side'] = ('left', box((preg == R('hands')) & left, 0.95))
            zoom['hand_back'] = ('back', box((preg == R('hands')) & left, 1.5))
            zoom['hat_top'] = ('top', box(preg == R('head_hat')))
            zoom['bracer_back'] = ('back', box((preg == R('arms')) & left))
            names = report['parts'][part].get('groups') or []
            comp = [e['name'] for e in report['parts'][part].get('build', {}).get('extras', []) if e['kind'] != 'arc']
            ga = low.data.attributes.get('sr_group')
            if ga is not None and comp and names:
                gv = np.zeros(len(low.data.polygons), dtype=np.int32)
                ga.data.foreach_get('value', gv)
                for cn in comp[:1]:
                    fs = np.flatnonzero(gv == names.index(cn)) if cn in names else []
                    if len(fs):
                        pc = np.array([(low.matrix_world @ low.data.polygons[int(i)].center)[:] for i in fs])
                        qc, qh = tuple(((pc.min(0) + pc.max(0)) / 2).tolist()), max(float(np.ptp(pc, axis=0).max()) * 0.9, 0.06 * H)
                        zoom.update({'quiver': ('front', (qc, qh)), 'quiver_side': ('left', (qc, qh)), 'quiver_34': ('back34', (qc, qh)),
                                     'quiver_right': ('right', (qc, qh))})
        for _n, c_ in mcopies:
            bpy.data.objects.remove(c_)
        res[part] = r
        print('CMP', part, json.dumps(r, ensure_ascii=False), flush=True)
        if renders:
            grey = bpy.data.materials.new('cmp_grey')
            grey.diffuse_color = (0.75, 0.75, 0.75, 1)
            gs = []
            for o in (man, low, high):
                c = o.copy()
                c.data = o.data.copy()
                bpy.context.scene.collection.objects.link(c)
                c.data.materials.clear()
                c.data.materials.append(grey)
                for p in c.data.polygons:
                    p.material_index = 0
                gs.append(c)
            objs = [man, low, high]
            if part == 'character':
                preview.render_views(gs, os.path.join(out, 'cmp_%s_wire.png' % part), views=('front', 'left', 'back'), res=1100, wire=True, textured=False, ref=high)
                preview.render_views(objs, os.path.join(out, 'cmp_%s_tex.png' % part), views=('front', 'left', 'back'), res=1100, wire=False, ref=high)
                for n, (v, (c, h)) in zoom.items():
                    # backface culling: a flipped face shows as a hole
                    preview.render_views(gs, os.path.join(out, 'cmp_%s_%s_wire.png' % (part, n)), views=(v,), res=800, wire=True, textured=False, zoom_box=(c, h), cull=True)
                    preview.render_views(objs, os.path.join(out, 'cmp_%s_%s_tex.png' % (part, n)), views=(v,), res=800, wire=False, zoom_box=(c, h))
            else:
                # weapons: looked at along their three principal axes
                P = sample(high, 8000)
                c, V = skelretopo._pca(P)
                h = float(np.abs((P - c) @ V[0]).max()) * 1.1
                Rm = Matrix((tuple(V[1]), tuple(np.cross(V[0], V[1]) * -1), tuple(V[0]))).to_4x4()      # main axis -> Z, flat side faces -Y
                Tm = Rm @ Matrix.Translation(Vector(-c))
                old = [(o, o.matrix_world.copy()) for o in gs + objs]
                for o, m in old:
                    o.matrix_world = Tm @ m
                bpy.context.view_layer.update()
                # per object framing: every panel is centred on its own object (the manual weapon has another
                # shape and size than the source, the alignment by ICP does not put it at the same place), same scale
                for o, m in old:
                    bb = np.array(preview._world_bbox(o))
                    o.matrix_world = Matrix.Translation(Vector(-(bb[0] + bb[1]) / 2)) @ o.matrix_world
                bpy.context.view_layer.update()
                hb = max(float(np.linalg.norm(np.ptp(np.array(preview._world_bbox(o)), axis=0))) for o in gs) * 0.52
                h = max(h, hb) if hb < 1.5 * h else h
                for v in ('front', 'left', 'top', 'front34', 'back34'):
                    preview.render_views(gs, os.path.join(out, 'cmp_%s_%s_wire.png' % (part, v)), views=(v,), res=900, wire=True, textured=False, zoom_box=((0, 0, 0), h), cull=True)
                    preview.render_views(objs, os.path.join(out, 'cmp_%s_%s_tex.png' % (part, v)), views=(v,), res=900, wire=False, zoom_box=((0, 0, 0), h))
                # round 6: close-ups of the two halves (shaft / blade edge, limbs / disc)
                for zn, zc in (('upper', (0, 0, 0.45 * h)), ('lower', (0, 0, -0.45 * h)), ('centre', (0, 0, 0))):
                    preview.render_views(gs, os.path.join(out, 'cmp_%s_zoom_%s_wire.png' % (part, zn)), views=('front',), res=900, wire=True, textured=False, zoom_box=(zc, 0.5 * h), cull=True)
                    preview.render_views(objs, os.path.join(out, 'cmp_%s_zoom_%s_tex.png' % (part, zn)), views=('front',), res=900, wire=False, zoom_box=(zc, 0.5 * h))
                # round 9: the end of a limb / the far end across the main axis
                hv = np.array([(gs[2].matrix_world @ v_.co)[:] for v_ in gs[2].data.vertices])
                far = hv[hv[:, 0] > 0.55 * hv[:, 0].max()]
                if len(far) > 10 and hv[:, 0].max() > 0.25 * h:
                    zc = (float(0.72 * hv[:, 0].max()), 0, float(np.median(far[:, 2])))
                    preview.render_views(gs, os.path.join(out, 'cmp_%s_zoom_end_wire.png' % part), views=('front',), res=900, wire=True, textured=False, zoom_box=(zc, 0.3 * h), cull=True)
                    preview.render_views(objs, os.path.join(out, 'cmp_%s_zoom_end_tex.png' % part), views=('front',), res=900, wire=False, zoom_box=(zc, 0.3 * h))
                for o, m in old:
                    o.matrix_world = m
            for c in gs:
                bpy.data.objects.remove(c)
        for o in (low, high, man):
            o.hide_render = True
    report['manual_comparison'] = res
    json.dump(report, open(rep_path, 'w'), ensure_ascii=False, indent=1, default=float)
    # markdown table
    L = []
    for part, r in res.items():
        L.append('### %s: manual %d tris / ours %d / source %d' % (part, r['tris']['manual'], r['tris']['ours'], r['tris']['source']))
        if 'region_tris' in r:
            L.append('| region | manual | ours |\n|---|---|---|')
            for k in REGIONS:
                L.append('| %s | %d | %d |' % (k, r['region_tris']['manual'][k], r['region_tris']['ours'][k]))
        L.append('\n| metric | manual | ours |\n|---|---|---|')
        for v in r['silhouette_diff_pct']['manual']:
            L.append('| silhouette %s geo / alpha %% | %s / %s | %s / %s |' % (v, r['silhouette_diff_pct']['manual'][v]['geo'], r['silhouette_diff_pct']['manual'][v]['alpha'],
                                                                           r['silhouette_diff_pct']['ours'][v]['geo'], r['silhouette_diff_pct']['ours'][v]['alpha']))
        for k in ('hausdorff_pct', 'hausdorff_high_to_low_pct', 'hausdorff_low_to_high_pct', 'mean_pct', 'p99_pct'):
            L.append('| %s | %s | %s |' % (k, r['geometry']['manual'][k], r['geometry']['ours'][k]))
        L.append('')
    open(os.path.join(out, 'cmp.md'), 'w').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
