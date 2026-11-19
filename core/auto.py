"""Automatic choice of the reduction pipeline, per mesh object.

What matters is not how many quads the file contains but whether the mesh HAS
a quad grid, i.e. edge loops that can be removed:

    weld the vertices the exporter split along UV seams, put the triangle
    pairs back together (join_triangles, never across a UV seam), then
    grid = share of the inner vertices that have 4 edges and only quads
           around them ("regular" vertices)

    grid >= GRID_MIN (0.25)
        native quad share >= NATIVE_QUAD_MIN (0.5) and not split along seams
                          -> 'character'  ring collapse on the mesh as it is
        else              -> 'rejoin'     weld + rejoin quads, then ring collapse
    grid <  GRID_MIN      -> 'prop'       true triangle soup (scan, sculpt,
                                          decimated mesh): Blender Decimate

Measured (see HANDOFF 2d): artist quad meshes 0.48-0.95, triangulated quad
meshes 0.28-0.47, triangle soups 0.05-0.17.  The share of triangles that CAN be
paired to quads does not separate them (soups still reach 0.65-0.7).

All three keep the UVs, materials and textures of the source.  The bake
pipeline (core.retopo) is a separate, explicit choice: it replaces the textures.
"""
import math
import numpy as np
import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

GRID_MIN = 0.25
NATIVE_QUAD_MIN = 0.5
WELD_MIN = 0.9          # welded / original vertex count below this: the file is split along seams
SMALL_OBJECT = 60       # inner vertices; below that the grid score of the whole model is used
JOIN_WARP = 0.0025      # rejoined quads may be warped by this share of the object's diagonal (see retopo.join_quads)
PROP_BOTH_MAX = 20000   # faces: up to here both prop decimators are run and compared (ours is slow beyond)
PROP_STRICT_OK = 1.0    # % of the object's diagonal: Hausdorff our own decimator may always have


def _diag(me):
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    return float(np.linalg.norm(co.max(0) - co.min(0))) if len(co) else 0.0


def _dup(obj, name):
    c = obj.copy()
    c.data = obj.data.copy()
    c.name = c.data.name = name
    bpy.context.scene.collection.objects.link(c)
    return c


def probe_object(obj):
    """-> dict with the face statistics and the grid score of one mesh object"""
    from . import retopo
    me = obj.data
    q0 = sum(1 for p in me.polygons if p.loop_total == 4)
    t0 = sum(1 for p in me.polygons if p.loop_total == 3)
    te = sum(p.loop_total - 2 for p in me.polygons)
    c = _dup(obj, 'mr_probe')
    retopo.weld(c, manifold=True)
    v1 = len(c.data.vertices)
    retopo.join_quads(c.data, max_warp=JOIN_WARP * _diag(c.data))
    bm = bmesh.new()
    bm.from_mesh(c.data)
    q1 = sum(1 for f in bm.faces if len(f.verts) == 4)
    inner = reg = 0
    for v in bm.verts:
        if v.is_boundary or not v.link_faces:
            continue
        inner += 1
        if len(v.link_edges) == 4 and all(len(f.verts) == 4 for f in v.link_faces):
            reg += 1
    nf1 = len(bm.faces)
    bm.free()
    me1 = c.data
    bpy.data.objects.remove(c)
    bpy.data.meshes.remove(me1)
    return {'object': obj.name, 'faces': len(me.polygons), 'tris': t0, 'quads': q0, 'tri_equiv': te,
            'native_quad_share': round(2 * q0 / max(te, 1), 3), 'faces_rejoined': nf1, 'quads_rejoined': q1,
            'rejoin_quad_share': round(2 * q1 / max(te, 1), 3), 'inner_verts': inner, 'regular_verts': reg,
            'grid': round(reg / max(inner, 1), 3), 'weld_ratio': round(v1 / max(len(me.vertices), 1), 3)}


def choose_mode(st, model_grid=None):
    grid = st['grid']
    if st['inner_verts'] < SMALL_OBJECT and model_grid is not None:
        grid = model_grid
    if grid < GRID_MIN:
        return 'prop'
    if st['native_quad_share'] >= NATIVE_QUAD_MIN and st['weld_ratio'] >= WELD_MIN:
        return 'character'
    return 'rejoin'


def probe(objs):
    """-> {'objects': [per object stats + 'mode'], 'grid': model score, 'mode': mode of most faces}"""
    sts = [probe_object(o) for o in objs]
    inner = sum(s['inner_verts'] for s in sts)
    grid = sum(s['regular_verts'] for s in sts) / max(inner, 1)
    by = {}
    for s in sts:
        s['mode'] = choose_mode(s, grid)
        by[s['mode']] = by.get(s['mode'], 0) + s['tri_equiv']
    te = sum(s['tri_equiv'] for s in sts)
    return {'objects': sts, 'grid': round(grid, 3), 'tri_equiv': te, 'faces': sum(s['faces'] for s in sts),
            'native_quad_share': round(sum(2 * s['quads'] for s in sts) / max(te, 1), 3),
            'mode': max(by, key=by.get) if by else 'prop'}


# ------------------------------------------------------------ LOD copies
def _world_bvh(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    pts = [(f.calc_center_median(), f.calc_area()) for f in bm.faces]
    if len(pts) > 4000:
        step = len(pts) // 4000 + 1
        pts = pts[::step]
    t = BVHTree.FromBMesh(bm)
    area = sum(f.calc_area() for f in bm.faces)
    bm.free()
    return t, pts, area


def lod_duplicates(objs, tol=0.015, cover=0.9, back=0.8):
    """Objects that are lower detail copies of other objects of the same file
    (LOD1..n next to LOD0, exported together): X is one when >= cover of its
    surface lies within tol * diagonal of the other objects AND it covers >=
    back of them in return (an accessory lying on the body is covered by the
    body but does not cover it).  Of several such objects the one with the
    fewest faces per area goes first, then the test is repeated.  -> [objects]"""
    objs = [o for o in objs if len(o.data.polygons)]
    if len(objs) < 2:
        return []
    data = {o.name: _world_bvh(o) for o in objs}
    lo = Vector((min((o.matrix_world @ Vector(c))[i] for o in objs for c in o.bound_box) for i in range(3)))
    hi = Vector((max((o.matrix_world @ Vector(c))[i] for o in objs for c in o.bound_box) for i in range(3)))
    d = tol * (hi - lo).length
    left = list(objs)
    out = []

    def covered(pts, trees):
        a = sum(ar for _q, ar in pts)
        return sum(ar for q, ar in pts if any(t.find_nearest(q, d)[0] is not None for t in trees)) / max(a, 1e-30)
    while len(left) > 1:
        cand = []
        for x in left:
            tx, px, ax = data[x.name]
            others = [o for o in left if o is not x]
            if covered(px, [data[o.name][0] for o in others]) < cover:
                continue
            tot = sum(data[o.name][2] for o in others)
            got = sum(data[o.name][2] * covered(data[o.name][1], [tx]) for o in others)
            if got >= back * tot:
                cand.append((len(x.data.polygons) / max(ax, 1e-30), x))
        if not cand:
            break
        x = min(cand, key=lambda c: c[0])[1]
        left.remove(x)
        out.append(x)
    return out


# ------------------------------------------------------- one mesh object
def bone_parent_to_skin(obj):
    """a mesh that hangs rigidly on a bone (parent_type BONE, no skin) gets a vertex group of
    that bone with weight 1 and an armature modifier: it can then be joined with skinned meshes"""
    if obj.parent is None or obj.parent.type != 'ARMATURE' or obj.parent_type != 'BONE' or not obj.parent_bone:
        return False
    if any(m.type == 'ARMATURE' for m in obj.modifiers):
        return False
    arm, bone = obj.parent, obj.parent_bone
    mw = obj.matrix_world.copy()
    g = obj.vertex_groups.get(bone) or obj.vertex_groups.new(name=bone)
    g.add(list(range(len(obj.data.vertices))), 1.0, 'REPLACE')
    obj.parent_type = 'OBJECT'
    obj.parent_bone = ''
    obj.matrix_world = mw
    m = obj.modifiers.new('Armature', 'ARMATURE')
    m.object = arm
    return True


def merge_objects(objs, name='merged'):
    """one mesh object from copies of objs (world space kept; rigid bone children become
    skinned): the input of core.retopo, which works on the connected components of ONE mesh"""
    cs = []
    for o in objs:
        c = _dup(o, 'mr_merge')
        bpy.context.view_layer.update()
        bone_parent_to_skin(c)
        cs.append(c)
    # the copy with an armature leads: its parent / modifier survive the join
    cs.sort(key=lambda c: (not any(m.type == 'ARMATURE' for m in c.modifiers), -len(c.data.polygons)))
    uv = next((c.data.uv_layers[0].name for c in cs if c.data.uv_layers), None)
    for c in cs:
        if c.data.uv_layers and uv:
            c.data.uv_layers[0].name = uv       # same name: the join keeps one layer
    for o in bpy.context.scene.objects:
        o.select_set(False)
    for c in cs:
        c.hide_set(False)
        c.select_set(True)
    bpy.context.view_layer.objects.active = cs[0]
    if len(cs) > 1:
        bpy.ops.object.join()
    out = cs[0]
    out.name = out.data.name = name
    return out


def _tri_count(me):
    return sum(p.loop_total - 2 for p in me.polygons)


def _loops_to_ratio(src, ratio, symmetry, progress, name, extra, retries=1, tol=0.05):
    """ring collapse of src down to `ratio` of its TRIANGLE count.  The reducer counts
    polygons and removes quads and triangles in unknown proportion, so the polygon target
    is corrected once from the result (the reduction itself is deterministic)."""
    from . import api, retopo
    n = len(src.data.polygons)
    want = ratio * _tri_count(src.data)
    tgt = max(4, int(round(n * ratio)))
    best = None
    for k in range(retries + 1):
        new, info = api.reduce_character(src, target_faces=tgt, symmetry=symmetry, progress=progress, name=name, **extra)
        got = _tri_count(new.data)
        if best is None or abs(got - want) < abs(best[2] - want):
            if best is not None:
                bpy.data.objects.remove(best[0])
            best = (new, info, got)
        else:
            bpy.data.objects.remove(new)
        if abs(best[2] - want) <= tol * want or info.get('stuck') or k == retries:
            break
        t2 = max(4, int(round(tgt * want / max(got, 1))))
        if t2 == tgt or t2 >= n:
            break
        tgt = t2
    return best[0], best[1]


def reduce_object(obj, ratio, mode, symmetry='AUTO', progress=None, **extra):
    """reduce one mesh object to about `ratio` of its triangle count with the given mode.
    -> (new object, info)"""
    from . import api, prop, retopo
    n = len(obj.data.polygons)
    if mode == 'prop':
        tgt = max(4, int(round(n * ratio)))
        new, info = prop.reduce_prop(obj, target_faces=tgt, symmetry='NONE', method='FAST', progress=progress)
        if n <= PROP_BOTH_MAX:
            # two decimators: ours keeps every UV seam exact but can fold or drop small closed
            # pieces of a low poly mesh; Blender's is geometrically tighter but merges small UV
            # islands.  Both are run, ours is kept unless it is clearly further from the source
            from . import metrics
            new2, info2 = prop.reduce_prop(obj, target_faces=tgt, symmetry='NONE', method='STRICT')
            h1 = metrics.evaluate(obj.data, new.data, n_samples=6000)['geometry']['hausdorff_pct']
            h2 = metrics.evaluate(obj.data, new2.data, n_samples=6000)['geometry']['hausdorff_pct']
            if h2 <= max(PROP_STRICT_OK, 1.5 * h1) and not info2.get('stuck'):
                new, new2, info = new2, new, info2
            info['hausdorff_fast_strict'] = [h1, h2]
            me2 = new2.data
            bpy.data.objects.remove(new2)
            bpy.data.meshes.remove(me2)
        extra = {}
    elif mode == 'character':
        new, info = _loops_to_ratio(obj, ratio, symmetry, progress, None, extra)
    elif mode == 'rejoin':
        src = _dup(obj, obj.name + '_welded')
        retopo.weld(src, manifold=True)
        retopo.join_quads(src.data, max_warp=JOIN_WARP * _diag(src.data))
        n1 = len(src.data.polygons)
        new, info = _loops_to_ratio(src, ratio, symmetry, progress, obj.name + '_reduced', extra)
        retopo.heal_tri_pairs(new.data)
        info['faces_before'] = n
        info['faces_rejoined'] = n1
        info['faces_after'] = len(new.data.polygons)
        info['quads'] = sum(1 for p in new.data.polygons if p.loop_total == 4)
        info['tris'] = sum(1 for p in new.data.polygons if p.loop_total == 3)
        me = src.data
        bpy.data.objects.remove(src)
        bpy.data.meshes.remove(me)
    else:
        raise ValueError('unknown mode ' + mode)
    info['mode'] = mode
    info['tris_before'] = _tri_count(obj.data)
    info['tris_after'] = _tri_count(new.data)
    if len(obj.vertex_groups):
        api.limit_skin_weights(new, max(4, api.max_influences(obj)))
    return new, info


def reduce_model(objs, ratio, mode='auto', log=print, progress=None, **extra):
    """reduce every mesh object of a model to `ratio` of its triangles.
    mode: 'auto' (per object, see module docstring) | 'character' | 'rejoin' | 'prop'.
    -> ([(new, original, info)], probe result).  LOD copies are left out (info only)."""
    pr = probe(objs)
    lods = set(o.name for o in lod_duplicates(objs))
    pr['lod_copies'] = sorted(lods)
    out = []
    for o, st in zip(objs, pr['objects']):
        if o.name in lods:
            log('%s: lower LOD copy of other objects, skipped' % o.name)
            st['mode'] = 'skipped (LOD copy)'
            continue
        m = st['mode'] if mode == 'auto' else mode
        n = len(o.data.polygons)
        if n * ratio < 4:
            new = _dup(o, o.name + '_reduced')
            info = {'mode': m, 'faces_before': n, 'faces_after': n, 'unchanged': True}
        else:
            new, info = reduce_object(o, ratio, m, progress=progress, **extra)
        info['object'] = o.name
        info['grid'] = st['grid']
        log('%s: %s %d -> %d faces%s' % (o.name, m, n, info['faces_after'], ' (stuck)' if info.get('stuck') else ''))
        out.append((new, o, info))
    return out, pr
