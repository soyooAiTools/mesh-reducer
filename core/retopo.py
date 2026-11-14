"""Retopology + bake pipeline (PROTOTYPE) for triangulated game characters.

Follows the artists' "拓扑烘焙" rule: build a clean, lighter topology, drop
detail whose height is small and put it back with baked maps.

    high model (triangles, several materials, skinned)
      1. weld, split into parts (by bone sets) and connected components
      2. classify the components of a part
           cards   alpha material (feathers): cannot become a solid, kept as
                   cards with their own UVs / texture, only thinned out
           relief  small piece lying on a bigger one (hair strands on the hair
                   mass, buckles, badges): removed, baked into the host
           bundle  cluster of tiny pieces (arrows): one merged shell
           shell   piece that carries many relief pieces (hair mass under its
                   strands): host + strands become one closed envelope shell
           keep    everything else
      3. geometry
           keep    the triangles are put back together into the quads the
                   artist modelled (join_triangles), then whole edge loops are
                   removed with the ring collapse of core.character, so the
                   edge flow of the source survives
           bundle  convex hull
           shell   voxel union of host + strands, closed (dilate / erode),
                   QuadriFlow quads, faces nobody can see removed
      4. new UV atlas (one per part), Cycles selected-to-active bakes from the
         high model: base colour, tangent normal (incl. the high model's
         normal maps), AO.  Per-vertex cage: extruded as far as the relief
         needs but never through the underside of another cloth layer
      5. skin weights by data transfer (nearest face interpolated), 4
         influences, normalised; same armature

Nothing here changes core.character; it is only called.
"""
import math
import os
import time
import collections
import numpy as np
import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import api


SRC_FACE = 'rt_src_face'
FOLD_ANGLE = 100.0       # degrees: sharper folds are hard edges and are split in the bake cage
INNER_SHIFT = 16.0      # UV offset of the rebuilt inner cloth side (keeps its islands apart from the outer side)


# ----------------------------------------------------------------- helpers
def _activate(obj):
    for o in bpy.context.scene.objects:
        o.select_set(False)
    obj.hide_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _dup(obj, name):
    c = obj.copy()
    c.data = obj.data.copy()
    c.name = name
    c.data.name = name
    bpy.context.scene.collection.objects.link(c)
    return c


def _keep_faces(obj, keep):
    """delete every polygon whose index is not in `keep` (set)"""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.index not in keep], context='FACES')
    bm.to_mesh(obj.data)
    bm.free()


def _strip_unused_materials(obj):
    me = obj.data
    used = sorted({p.material_index for p in me.polygons})
    if len(used) == len(me.materials):
        return
    mats = [me.materials[i] for i in used]
    remap = {o: n for n, o in enumerate(used)}
    idx = [remap[p.material_index] for p in me.polygons]
    me.materials.clear()
    for m in mats:
        me.materials.append(m)
    for p, i in zip(me.polygons, idx):
        p.material_index = i


def weld(obj, rel=1e-6):
    """merge vertices at the same position (the FBX is split along every UV
    seam / smoothing group border).  Distance is relative to the bbox diagonal
    and large enough to also remove the zero length edges QuadriFlow rejects."""
    me = obj.data
    co = np.array([v.co[:] for v in me.vertices])
    diag = float(np.linalg.norm(co.max(0) - co.min(0)))
    bm = bmesh.new()
    bm.from_mesh(me)
    lay = bm.faces.layers.int.new(SRC_FACE)
    for f in bm.faces:
        f[lay] = f.index
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=diag * rel)
    bm.to_mesh(me)
    bm.free()


def source_faces(me):
    """polygon index of the welded mesh -> polygon index before weld()"""
    a = me.attributes.get(SRC_FACE)
    out = np.empty(len(me.polygons), dtype=np.int32)
    a.data.foreach_get('value', out)
    return out


def join_quads(me, angle=60.0):
    """put the triangle pairs back together into the quads they were modelled
    as (never across a UV seam or a material border: the UVs of the source are kept)"""
    bm = bmesh.new()
    bm.from_mesh(me)
    a = math.radians(angle)
    bmesh.ops.join_triangles(bm, faces=bm.faces, cmp_seam=False, cmp_sharp=False, cmp_uvs=True,
                             cmp_vcols=False, cmp_materials=True,
                             angle_face_threshold=a, angle_shape_threshold=a)
    bm.to_mesh(me)
    bm.free()


def material_is_alpha(mat):
    """material whose base colour image is linked to Alpha (core.materials does
    that only for images that really are transparent)"""
    if mat is None or not mat.use_nodes:
        return False
    b = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    return b is not None and b.inputs['Alpha'].is_linked


# ------------------------------------------------------- split + classify
class Comp:
    __slots__ = ('faces', 'mat', 'bones', 'top', 'area', 'lo', 'hi', 'pts', 'bvh', 'part', 'kind', 'host', 'near')


def analyse(obj, part_bones=None, relief_height=0.006, relief_cover=0.85, tiny_faces=30, bundle_min=4, shell_min_relief=8, relief_max_faces=400):
    """Connected components of the (welded) mesh, their part and their kind.
    part_bones: {part: [bone names]}; a component weighted only to these bones
    belongs to that part, the rest to 'character'.
    relief_height: fraction of the part's bbox diagonal."""
    me = obj.data
    mw = obj.matrix_world
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.transform(mw)
    bm.faces.ensure_lookup_table()
    dl = bm.verts.layers.deform.active
    gname = [g.name for g in obj.vertex_groups]
    cid = [-1] * len(bm.faces)
    comps = []
    for f in bm.faces:
        if cid[f.index] >= 0:
            continue
        k = len(comps)
        st = [f]
        cid[f.index] = k
        fs = []
        while st:
            g = st.pop()
            fs.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if cid[h.index] < 0:
                        cid[h.index] = k
                        st.append(h)
        c = Comp()
        c.faces = [g.index for g in fs]
        c.mat = fs[0].material_index
        bw = collections.Counter()
        vs = {v for g in fs for v in g.verts}
        if dl is not None:
            for v in vs:
                for gi, w in v[dl].items():
                    if w > 1e-4:
                        bw[gname[gi]] += w
        c.bones = set(bw)
        c.top = bw.most_common(1)[0][0] if bw else None
        co = np.array([v.co[:] for v in vs])
        c.lo, c.hi = co.min(0), co.max(0)
        c.pts = [(g.calc_center_median(), g.calc_area()) for g in fs]
        c.area = sum(a for _, a in c.pts)
        sub = bmesh.new()
        vmap = {}
        for g in fs:
            sub.faces.new([vmap.setdefault(v, sub.verts.new(v.co)) for v in g.verts])
        c.bvh = BVHTree.FromBMesh(sub)
        sub.free()
        c.part = 'character'
        for p, bs in (part_bones or {}).items():
            if c.bones and c.bones <= set(bs):
                c.part = p
        c.kind, c.host, c.near = 'keep', None, 0.0
        comps.append(c)
    bm.free()

    alpha = [material_is_alpha(m) for m in me.materials]
    parts = sorted({c.part for c in comps})
    for p in parts:
        cs = [c for c in comps if c.part == p]
        lo = np.min([c.lo for c in cs], 0)
        hi = np.max([c.hi for c in cs], 0)
        h = relief_height * float(np.linalg.norm(hi - lo))
        for c in cs:
            if alpha[c.mat]:
                c.kind = 'cards'

        def near(a, b, dist):
            if np.any(a.lo > b.hi + dist) or np.any(b.lo > a.hi + dist):
                return 0.0
            return sum(ar for q, ar in a.pts if b.bvh.find_nearest(q, dist)[0] is not None) / max(a.area, 1e-12)

        # bundles: clusters of tiny pieces close to each other
        tiny = [c for c in cs if c.kind == 'keep' and len(c.faces) <= tiny_faces]
        par = {id(c): c for c in tiny}

        def find(c):
            while par[id(c)] is not c:
                par[id(c)] = par[id(par[id(c)])]
                c = par[id(c)]
            return c
        for i, a in enumerate(tiny):
            for b in tiny[i + 1:]:
                if find(a) is not find(b) and max(near(a, b, 2 * h), near(b, a, 2 * h)) > 0.3:
                    par[id(find(a))] = find(b)
        groups = collections.defaultdict(list)
        for c in tiny:
            groups[id(find(c))].append(c)
        for g in groups.values():
            if len(g) >= bundle_min:
                for c in g:
                    c.kind = 'bundle'
                    c.host = g[0]
        # relief: small piece lying on kept, bigger pieces that carry its bones
        kept = []
        for c in sorted([c for c in cs if c.kind == 'keep'], key=lambda c: -c.area):
            best, host = 0.0, None
            for k in kept:
                if k.area < 4 * c.area or (c.top is not None and c.top not in k.bones):
                    continue
                n = near(c, k, h)
                if n > best:
                    best, host = n, k
            c.near = best
            # (a big ornament keeps its geometry: it has its own, much finer texture space, and
            # baked onto the coarse texels of its host it would turn into a blur - hat badge)
            if best >= relief_cover and len(c.faces) <= relief_max_faces:
                c.kind, c.host = 'relief', host
            else:
                kept.append(c)
        # a piece that carries many relief pieces (hair mass under its strands) is not reduced
        # by itself: host + relief are replaced by ONE envelope shell (see envelope_shell)
        cnt = collections.Counter(id(c.host) for c in cs if c.kind == 'relief')
        for c in cs:
            if c.kind == 'keep' and cnt.get(id(c), 0) >= shell_min_relief:
                c.kind = 'shell'
        # ... and takes the loose pieces of the same kind with it (single strands, long locks):
        # same material, small, and either rigged to the same bones or lying against the group.
        # Reduced on their own they end up as thin sticks
        for H in [c for c in cs if c.kind == 'shell']:
            group = [H] + [c for c in cs if c.kind == 'relief' and c.host is H]
            gb = set().union(*[g.bones for g in group])
            changed = True
            while changed:
                changed = False
                for c in cs:
                    if c.kind != 'keep' or c.mat != H.mat or c.area > 0.25 * H.area:
                        continue
                    touch = sum(ar for q, ar in c.pts if any(
                        not (np.any(np.array(q) > g.hi + 3 * h) or np.any(g.lo > np.array(q) + 3 * h)) and
                        g.bvh.find_nearest(q, 3 * h)[0] is not None for g in group)) / max(c.area, 1e-12)
                    if (c.bones and c.bones <= gb and touch >= 0.1) or touch >= 0.5:
                        c.kind, c.host = 'relief', H
                        group.append(c)
                        gb |= c.bones
                        changed = True
    return comps


def describe(comps, me):
    rows = []
    for i, c in enumerate(comps):
        rows.append('%3d %-9s %-7s tris %5d area %.4f mat %-12s top %-22s near %.2f' % (
            i, c.part, c.kind, len(c.faces), c.area, me.materials[c.mat].name, c.top, c.near))
    return rows


# ------------------------------------------------------------- geometry
def heal_tri_pairs(me, angle=25.0):
    """after the loop reduction: triangle pairs that form a flat, well shaped quad"""
    bm = bmesh.new()
    bm.from_mesh(me)
    tris = [f for f in bm.faces if len(f.verts) == 3]
    a = math.radians(angle)
    bmesh.ops.join_triangles(bm, faces=tris, cmp_seam=False, cmp_sharp=False, cmp_uvs=True,
                             cmp_vcols=False, cmp_materials=True,
                             angle_face_threshold=a, angle_shape_threshold=math.radians(40))
    bm.to_mesh(me)
    bm.free()


def reduce_loops(obj, target, name, symmetry='AUTO', **extra):
    """quad recovery + loop removal (core.character ring collapse).  The UV
    islands of the source survive (seams are constraints): they are repacked
    into the new atlas, which gives far better islands than any automatic unwrap."""
    join_quads(obj.data)
    n0 = len(obj.data.polygons)
    if target >= n0:
        r = _dup(obj, name)
        return r, {'faces_before': n0, 'faces_after': n0, 'seconds': 0}
    kw = dict(preserve_seams=True)
    kw.update(extra)
    r, info = api.reduce_character(obj, target_faces=target, symmetry=symmetry, name=name, **kw)
    heal_tri_pairs(r.data)
    return r, info


def bundle_shell(src, faces, name, target=None):
    """one merged shell around a cluster of tiny pieces: convex hull, planar
    regions dissolved, then triangle pairs joined to quads"""
    o = _dup(src, name)
    _keep_faces(o, set(faces))
    bm = bmesh.new()
    bm.from_mesh(o.data)
    res = bmesh.ops.convex_hull(bm, input=bm.verts, use_existing_faces=False)
    hull = set(res['geom'])
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f not in hull], context='FACES')
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    bm.normal_update()
    bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(12), verts=bm.verts, edges=bm.edges)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bmesh.ops.join_triangles(bm, faces=[f for f in bm.faces if len(f.verts) == 3], cmp_seam=False, cmp_sharp=False,
                             cmp_uvs=False, cmp_vcols=False, cmp_materials=False,
                             angle_face_threshold=math.radians(30), angle_shape_threshold=math.radians(60))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f in bm.faces:
        f.material_index = 0
    bm.to_mesh(o.data)
    bm.free()
    _project_uv(o, layer=src.data.uv_layers[0].name if src.data.uv_layers else 'UVMap')
    return o


SHELL = 'rt_shell'          # int face attribute: 1 = face of an envelope shell (baked from its own sources only)
NEW_UV = 'rt_new_uv'      # int face attribute: 1 = UVs made here (not the source's), density is set in atlas()


def _project_uv(obj, angle=66.0, layer='UVMap'):
    """UVs for a piece that has none from the source: smart projection, relaxed by an
    unwrap along the island borders; scaled to 1 UV unit per world unit; faces flagged NEW_UV"""
    me = obj.data
    while me.uv_layers:
        me.uv_layers.remove(me.uv_layers[0])
    me.uv_layers.new(name=layer)        # same name as the source layer: joining keeps ONE layer
    _activate(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.mark_seam(clear=True)
    bpy.ops.uv.smart_project(angle_limit=math.radians(angle), island_margin=0.01, area_weight=0.0)
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.uv.seams_from_islands(mark_seams=True, mark_sharp=False)
    try:
        bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.01)
        bpy.ops.uv.select_all(action='SELECT')
        bpy.ops.uv.average_islands_scale()
    except Exception as ex:
        print('WARN unwrap:', ex)
    bpy.ops.object.mode_set(mode='OBJECT')
    sc = max(abs(x) for x in obj.matrix_world.to_scale()) or 1.0
    uv = me.uv_layers[0].uv
    a3 = sum(p.area for p in me.polygons) * sc * sc
    a2 = 0.0
    for p in me.polygons:
        pts = [uv[l].vector for l in p.loop_indices]
        a2 += 0.5 * abs(sum(pts[i].x * pts[(i + 1) % len(pts)].y - pts[(i + 1) % len(pts)].x * pts[i].y for i in range(len(pts))))
    k = math.sqrt(a3 / a2) if a2 > 0 else 1.0
    for l in range(len(me.loops)):
        uv[l].vector = uv[l].vector * k
    at = me.attributes.get(NEW_UV) or me.attributes.new(NEW_UV, 'INT', 'FACE')
    at.data.foreach_set('value', [1] * len(me.polygons))


def _clean_border(obj, passes=3, smooth=6):
    """the open border left by removing the hidden faces follows the quads in steps: drop the
    faces that hang out (3+ border edges, or 2 opposite ones) and smooth the border line"""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for _ in range(passes):
        bad = []
        for f in bm.faces:
            be = [e.is_boundary for e in f.edges]
            n = sum(be)
            if n >= 3 or (n == 2 and len(be) == 4 and (be[0] and be[2] or be[1] and be[3])):
                bad.append(f)
        if not bad:
            break
        bmesh.ops.delete(bm, geom=bad, context='FACES')
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    for _ in range(smooth):
        new = {}
        for v in bm.verts:
            if not v.is_boundary:
                continue
            nb = [e.other_vert(v) for e in v.link_edges if e.is_boundary]
            if len(nb) == 2:
                new[v] = v.co * 0.5 + (nb[0].co + nb[1].co) * 0.25
        for v, co in new.items():
            v.co = co
    bm.to_mesh(obj.data)
    bm.free()


def _pull_tips(obj, tips, reach, min_len, cone=35.0, log=print):
    """The envelope loses the thin ends of the strands (they are thinner than its faces), its
    lower border is a soft scallop.  Border vertices are moved out to the strand ends that
    stick out beyond them: the vertex that reaches furthest along each stretch of border
    goes to the tip, its neighbours stay -> pointed locks at the places of the real ones,
    without new faces.  tips: world space points of the source that lie outside the envelope."""
    if not tips:
        return 0
    from mathutils import kdtree
    kd = kdtree.KDTree(len(tips))
    for i, q in enumerate(tips):
        kd.insert(q, i)
    kd.balance()
    mw = obj.matrix_world
    inv = mw.inverted()
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    cosc = math.cos(math.radians(cone))
    best = {}
    for v in bm.verts:
        if not v.is_boundary:
            continue
        inner = [e.other_vert(v) for e in v.link_edges if not e.is_boundary]
        if not inner:
            continue
        p = mw @ v.co
        o = p - sum((mw @ x.co for x in inner), Vector((0, 0, 0))) / len(inner)
        if o.length < 1e-9:
            continue
        o.normalize()
        far = None
        for q, _i, dist in kd.find_range(p, reach):
            d = q - p
            al = d.dot(o)
            if al > min_len and al > cosc * dist and (far is None or al > far[0]):
                far = (al, q)
        if far:
            best[v] = far
    moved = 0
    for v, (al, q) in best.items():
        nb = [e.other_vert(v) for e in v.link_edges if e.is_boundary]
        if all(best.get(x, (0.0,))[0] <= al for x in nb):
            v.co = inv @ q
            moved += 1
    bm.to_mesh(obj.data)
    bm.free()
    log('  shell %s: %d border vertices moved to strand tips' % (obj.name, moved))
    return moved


def envelope_shell(src, faces, name, target, bvh_all, T, mat_index=0, close=1.3, pull=0.7, voxel=0.35, opening=0.45, tips=True, log=print):
    """Many thin pieces on one host (strands on a hair mass): too many to keep, and removing
    loops turns the wavy tubes into straight sticks.  Like an artist: ONE solid shell around
    all of it, the strands go into the baked maps.
      solidify (the host is an open sheet) -> push out by `close`*T -> voxel union
      -> pull back in -> voxel remesh + smooth  = closed envelope, gaps < 2*close*T filled
      -> QuadriFlow quads -> faces that cannot be seen from outside (inside of the shell,
         against the head) removed -> repeated once or twice to land on `target` faces"""
    o = _dup(src, name)
    _keep_faces(o, set(faces))
    for m in list(o.modifiers):
        o.modifiers.remove(m)
    while o.data.materials:
        o.data.materials.pop()
    _activate(o)
    if o.data.has_custom_normals:
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
    sc = max(abs(x) for x in o.matrix_world.to_scale()) or 1.0
    d, vox = close * T / sc, voxel * T / sc

    def mod(tp, **kw):
        m = o.modifiers.new('m', tp)
        for k, v in kw.items():
            setattr(m, k, v)
        bpy.ops.object.modifier_apply(modifier=m.name)
    mod('SOLIDIFY', thickness=0.7 * T / sc, offset=-1.0)
    mod('DISPLACE', strength=d, mid_level=0.0, direction='NORMAL')
    mod('REMESH', mode='VOXEL', voxel_size=vox, use_smooth_shade=True)
    mod('DISPLACE', strength=-pull * d, mid_level=0.0, direction='NORMAL')      # a little less than pushed out: thin tips survive
    mod('REMESH', mode='VOXEL', voxel_size=vox, use_smooth_shade=True)
    if opening > 0:       # single stray strands that stick out would become little bars: shave them off
        mod('DISPLACE', strength=-opening * T / sc, mid_level=0.0, direction='NORMAL')
        mod('REMESH', mode='VOXEL', voxel_size=vox, use_smooth_shade=True)
        mod('DISPLACE', strength=opening * T / sc, mid_level=0.0, direction='NORMAL')
        mod('REMESH', mode='VOXEL', voxel_size=vox, use_smooth_shade=True)
    mod('SMOOTH', factor=0.5, iterations=6)
    env = o
    mw = env.matrix_world
    dirs = _hemi_dirs()
    n_ask = int(target * 1.8)
    best = None
    tried = []
    for attempt in range(5):
        r = _dup(env, name + '_q')
        _activate(r)
        res = bpy.ops.object.quadriflow_remesh(mode='FACES', target_faces=max(50, n_ask), use_preserve_sharp=False,
                                               use_preserve_boundary=False, use_mesh_symmetry=False)
        if 'FINISHED' not in res:
            bpy.data.objects.remove(r)
            raise RuntimeError('QuadriFlow failed on ' + name)
        bm = bmesh.new()
        bm.from_mesh(r.data)
        bm.normal_update()
        wb = bm.copy()
        wb.transform(mw)
        own = BVHTree.FromBMesh(wb)
        wb.free()
        n3 = mw.to_3x3().inverted().transposed()
        hidden = []
        for f in bm.faces:
            n = (n3 @ f.normal).normalized()
            c = mw @ f.calc_center_median()
            seen = False
            for q in [c] + [c.lerp(mw @ v.co, 0.6) for v in f.verts]:
                o_ = q + n * 2e-3
                for dv in dirs:
                    if dv.dot(n) < 0.15:
                        continue
                    if bvh_all.ray_cast(o_, dv, 10.0)[0] is None and own.ray_cast(o_, dv, 10.0)[0] is None:
                        seen = True
                        break
                if seen:
                    break
            if not seen:
                hidden.append(f)
        # only whole hidden regions go (the inside of the shell); a few hidden faces in the
        # middle of visible ones (under the hat brim) would be holes from another angle
        hs = set(hidden)
        done_h = set()
        hidden = []
        for f in hs:
            if f in done_h:
                continue
            st, isl = [f], [f]
            done_h.add(f)
            while st:
                g = st.pop()
                for e in g.edges:
                    for h in e.link_faces:
                        if h in hs and h not in done_h:
                            done_h.add(h)
                            st.append(h)
                            isl.append(h)
            if len(isl) >= 25:
                hidden.extend(isl)
        nh = len(hidden)
        bmesh.ops.delete(bm, geom=hidden, context='FACES')
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
        # crumbs left by the removal
        seen_f = set()
        for f in list(bm.faces):
            if f in seen_f:
                continue
            st, isl = [f], [f]
            seen_f.add(f)
            while st:
                g = st.pop()
                for e in g.edges:
                    for h in e.link_faces:
                        if h not in seen_f:
                            seen_f.add(h)
                            st.append(h)
                            isl.append(h)
            if len(isl) < 6:
                bmesh.ops.delete(bm, geom=isl, context='FACES')
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
        n = len(bm.faces)
        bm.to_mesh(r.data)
        bm.free()
        log('  shell %s: asked %d, %d hidden faces removed -> %d (target %d)' % (name, n_ask, nh, n, target))
        tried.append((n_ask, n))
        # QuadriFlow's face count comes in steps: keep the attempt closest to the target
        if best is None or abs(n - target) < abs(best[1] - target):
            if best is not None:
                bpy.data.objects.remove(best[0])
            best = (r, n)
        else:
            bpy.data.objects.remove(r)
        if n == 0 or abs(best[1] - target) <= 0.06 * target:
            break
        lo_ = max([t for t in tried if t[1] <= target], key=lambda t: t[1], default=None)
        hi_ = min([t for t in tried if t[1] > target], key=lambda t: t[1], default=None)
        if lo_ and hi_:
            n_ask = int(round(lo_[0] + (hi_[0] - lo_[0]) * (target - lo_[1]) / max(1, hi_[1] - lo_[1])))
        else:
            n_ask = int(n_ask * target / max(n, 1))
        if any(abs(n_ask - t[0]) < 8 for t in tried):
            break
    best = best[0]
    _clean_border(best)
    if tips:
        # points of the source outside the (smoothed) envelope = the ends it lost
        eb = bmesh.new()
        eb.from_mesh(env.data)
        eb.transform(mw)
        eb.normal_update()
        et = BVHTree.FromBMesh(eb)
        out_pts = []
        seen_v = set()
        for fi in faces:
            for vi in src.data.polygons[fi].vertices:
                if vi in seen_v:
                    continue
                seen_v.add(vi)
                q = src.matrix_world @ src.data.vertices[vi].co
                r = et.find_nearest(q)
                if r[0] is not None and r[3] > 0.25 * T and (q - r[0]).dot(r[1]) > 0:
                    out_pts.append(q)
        eb.free()
        _pull_tips(best, out_pts, reach=5 * T, min_len=0.8 * T, log=log)
    bpy.data.objects.remove(env)
    best.name = best.data.name = name
    _project_uv(best, layer=src.data.uv_layers[0].name if src.data.uv_layers else 'UVMap')
    best.data.materials.append(src.data.materials[mat_index])
    for p in best.data.polygons:
        p.material_index = 0
        p.use_smooth = True
    at = best.data.attributes.new(SHELL, 'INT', 'FACE')
    at.data.foreach_set('value', [1] * len(best.data.polygons))
    return best


# ------------------------------------------------- thin cloth: shell <-> sheet
_DIRS = None


def _hemi_dirs(n=40):
    global _DIRS
    if _DIRS is None or len(_DIRS) != n:
        rng = np.random.default_rng(7)
        d = rng.normal(size=(n, 3))
        _DIRS = [Vector(x) for x in d / np.linalg.norm(d, axis=1)[:, None]]
    return _DIRS


class Sheet:
    """what is needed to give a reduced sheet its inner side back"""
    __slots__ = ('bvh', 'inner', 'thick')


def sheetify(obj, bvh_all, T, min_fraction=0.4, min_visible=0.03, log=print):
    """Cloth modelled as a closed, paper thin double sided shell (coat, cape):
    remove the side that faces the body so that only ONE sheet is reduced; the
    inner side is rebuilt afterwards as a flipped copy (rethicken), like an
    artist does it.  Otherwise the two sides get different loops and poke
    through each other.  obj is modified in place.  Returns Sheet or None.
    bvh_all: BVH (world space) of the whole part, for the outside test.
    T: largest thickness (world units) that counts as 'thin'."""
    mw = obj.matrix_world
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(mw)
    bm.normal_update()
    bm.faces.ensure_lookup_table()
    own = BVHTree.FromBMesh(bm)
    N = len(bm.faces)
    opp = [-1] * N
    dist = [0.0] * N
    area = [f.calc_area() for f in bm.faces]
    cen = [f.calc_center_median() for f in bm.faces]
    for f in bm.faces:
        n = f.normal
        h = own.ray_cast(cen[f.index] - n * 1e-5, -n, T)
        if h[0] is not None and h[1].dot(n) < -0.3 and h[2] != f.index:
            opp[f.index] = h[2]
            dist[f.index] = h[3]
    thin_area = sum(area[i] for i in range(N) if opp[i] >= 0)
    tl = sorted(dist[i] for i in range(N) if opp[i] >= 0)
    if thin_area < min_fraction * sum(area) or not tl or tl[len(tl) // 2] > 0.5 * T:
        bm.free()
        return None
    # how much of the outside each face sees
    expo = [0.0] * N
    dirs = _hemi_dirs()
    for f in bm.faces:
        if opp[f.index] < 0:
            continue
        n = f.normal
        c = cen[f.index] + n * 1e-3
        k = tot = 0
        for d in dirs:
            if d.dot(n) < 0.15:
                continue
            tot += 1
            if bvh_all.ray_cast(c, d, 10.0)[0] is None:
                k += 1
        expo[f.index] = k / max(tot, 1)
    # smooth patches of thin faces
    patch = [-1] * N
    npatch = 0
    for f in bm.faces:
        if opp[f.index] < 0 or patch[f.index] >= 0:
            continue
        st = [f]
        patch[f.index] = npatch
        while st:
            g = st.pop()
            for e in g.edges:
                if len(e.link_faces) != 2:
                    continue
                h = e.link_faces[0] if e.link_faces[1] is g else e.link_faces[1]
                if opp[h.index] < 0 or patch[h.index] >= 0 or g.normal.dot(h.normal) < 0.5:
                    continue
                patch[h.index] = npatch
                st.append(h)
        npatch += 1
    pa = collections.defaultdict(float)
    pe = collections.defaultdict(float)
    pp = collections.defaultdict(collections.Counter)
    for i in range(N):
        p = patch[i]
        if p < 0:
            continue
        pa[p] += area[i]
        pe[p] += area[i] * expo[i]
        q = patch[opp[i]]
        if q >= 0:
            pp[p][q] += area[i]
    killp = set()
    for p in pa:
        if not pp[p]:
            continue
        q = pp[p].most_common(1)[0][0]
        if q != p and pe[p] / pa[p] < pe[q] / pa[q]:
            killp.add(p)
    kill = {i for i in range(N) if patch[i] in killp}
    # a face may only go when the face behind it stays
    for i in sorted(kill):
        j = opp[i]
        if j in kill and expo[i] >= expo[j]:
            kill.discard(i)
    for i in list(kill):
        if opp[i] in kill:
            kill.discard(i)
    # the inner side is rebuilt only where it can be seen from outside at all
    inner = [False] * N
    thick = [0.0] * N
    hidden = 0
    for i in kill:
        j = opp[i]
        if expo[i] <= min_visible:
            hidden += 1
            continue
        inner[j] = True
        thick[j] = max(thick[j], dist[i])
    for i in range(N):
        if i not in kill and opp[i] in kill and expo[opp[i]] > min_visible:
            inner[i] = True
            thick[i] = max(thick[i], dist[i])
    log('  sheet %s: %d of %d faces thin, inner side removed: %d faces (%d of them never visible)' % (
        obj.name, sum(1 for o_ in opp if o_ >= 0), N, len(kill), hidden))
    keepf = [f for f in bm.faces if f.index not in kill]
    sh = Sheet()
    sub = bmesh.new()
    vmap = {}
    for f in keepf:
        sub.faces.new([vmap.setdefault(v, sub.verts.new(v.co)) for v in f.verts])
    sh.bvh = BVHTree.FromBMesh(sub)
    sub.free()
    sh.inner = [inner[f.index] for f in keepf]
    sh.thick = [thick[f.index] for f in keepf]
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.index in kill], context='FACES')
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    bm.transform(mw.inverted())
    bm.to_mesh(obj.data)
    bm.free()
    return sh


def inner_flags(obj, sheet):
    """per polygon of obj: (has an inner side, thickness in world units).
    Sampled at the centre and towards every corner: a big reduced face must
    not lose its inner side because its centre happens to fall on one small
    source face without a partner."""
    mw = obj.matrix_world
    me = obj.data
    out = []
    for p in me.polygons:
        c = mw @ p.center
        pts = [c] + [c.lerp(mw @ me.vertices[v].co, 0.6) for v in p.vertices]
        hit = 0
        th = []
        for q in pts:
            i = sheet.bvh.find_nearest(q)[2]
            if i is not None and sheet.inner[i]:
                hit += 1
                th.append(sheet.thick[i])
        out.append((hit * 3 >= len(pts), float(np.median(th)) if th else 0.0))
    return out


def rethicken(obj, sheet, min_thick=0.003, log=print):
    """give the reduced sheet its inner side back: flipped copies of the faces
    that had one, pushed inwards by the cloth thickness.  The copies share the
    vertices of the open rim (hem), so the shell is closed there without extra
    faces; only where a face lies completely on the rim (spike tips) the rim is
    closed with a narrow strip."""
    flags = inner_flags(obj, sheet)
    mw = obj.matrix_world
    sc = max(abs(x) for x in mw.to_scale()) or 1.0
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.normal_update()
    S = [f for f in bm.faces if flags[f.index][0]]
    if not S:
        bm.free()
        return 0
    Sset = set(S)
    rim = set()
    for f in S:
        for e in f.edges:
            if len(e.link_faces) == 1:
                rim.update(e.verts)
    allrim = set()
    for f in S:
        if all(v in rim for v in f.verts):
            allrim.update(f.verts)
    vn = collections.defaultdict(lambda: Vector((0, 0, 0)))
    vt = collections.defaultdict(list)
    for f in S:
        for v in f.verts:
            vn[v] += f.normal * f.calc_area()
            vt[v].append(flags[f.index][1])
    dl = bm.verts.layers.deform.active
    uvls = list(bm.loops.layers.uv.values())
    inner = {}
    for v in vn:
        if v in rim and v not in allrim:
            inner[v] = v
            continue
        n = vn[v].normalized()
        # at least the cloth thickness, and enough for the size of the faces around: a warped
        # big quad and its copy must not cut through each other
        el = float(np.mean([e.calc_length() for e in v.link_edges])) if v.link_edges else 0.0
        t = max(min_thick / sc, float(np.median(vt[v])) / sc, 0.05 * el)
        w = bm.verts.new(v.co - n * t)
        if dl is not None:
            w[dl] = v[dl]
        inner[v] = w
    made = 0
    for f in S:
        fv = list(f.verts)
        vs = [inner[v] for v in [fv[0]] + fv[:0:-1]]      # reversed, same first vertex: same diagonal
        try:
            nf = bm.faces.new(vs)
        except ValueError:
            continue
        nf.material_index = f.material_index
        nf.smooth = f.smooth
        made += 1
        # same UV shape as the outer face, but its own island (shifted: no shared UV at the rim)
        fl = list(f.loops)
        src = [fl[0]] + fl[:0:-1]
        for uvl in uvls:
            for nl, sl in zip(nf.loops, src):
                nl[uvl].uv = sl[uvl].uv + Vector((INNER_SHIFT, 0.0))
    # close the rim where the inner side does not share the rim vertices
    for f in S:
        for l in f.loops:
            a, b = l.vert, l.link_loop_next.vert
            if a not in rim or b not in rim:
                continue
            e = l.edge
            if any(x is not f and x in Sset for x in e.link_faces):
                continue
            ia, ib = inner[a], inner[b]
            if ia is a and ib is b:
                continue
            vs = [b, a]
            if ia is not a:
                vs.append(ia)
            if ib is not b:
                vs.append(ib)
            try:
                nf = bm.faces.new(vs)
                nf.material_index = f.material_index
                made += 1
            except ValueError:
                pass
    bm.normal_update()
    bm.to_mesh(obj.data)
    bm.free()
    log('  rethicken %s: +%d inner faces' % (obj.name, made))
    return made


# ------------------------------------------------------------- build low
def _mirror_partner(c, cs, tol=0.02):
    for d in cs:
        if d is c or len(d.faces) != len(c.faces):
            continue
        lo = np.array([-d.hi[0], d.lo[1], d.lo[2]])
        hi = np.array([-d.lo[0], d.hi[1], d.hi[2]])
        if np.abs(lo - c.lo).max() < tol and np.abs(hi - c.hi).max() < tol:
            return d
    return None


def build_low(src, comps, part, target, log=print, card_ratio=0.45, density_gamma=0.3,
              relief_height=0.006, sheets=True, sheet_min_area=0.1, unit_min_ratio=0.25, bone_power=0.5, rigid_feature=0.006,
              shell_max_faces=950, **extra):
    """-> (low object without cards, cards object or None, info)

    Budget: every run unit (a component, or a left/right mirror pair) gets
    faces ~ area^g * source_faces^(1-g): g=0 keeps the density proportions of
    the source, g=1 gives the same edge length everywhere.  Units with many
    bones for their size (hands, hair) get more: * (1 + (bones - 1) / sqrt(area))^bone_power."""
    cs = [c for c in comps if c.part == part]
    info = {'part': part, 'kinds': dict(collections.Counter(c.kind for c in cs)), 'units': []}
    lo = np.min([c.lo for c in cs], 0)
    hi = np.max([c.hi for c in cs], 0)
    T = relief_height * float(np.linalg.norm(hi - lo))
    pieces = []
    # bundles
    groups = collections.defaultdict(list)
    for c in cs:
        if c.kind == 'bundle':
            groups[id(c.host)].append(c)
    for k, g in enumerate(groups.values()):
        pieces.append(bundle_shell(src, [f for c in g for f in c.faces], '%s_bundle%d' % (part, k)))
    used = sum(len(p.data.polygons) for p in pieces)
    # cards
    cards = None
    cf = {f for c in cs if c.kind == 'cards' for f in c.faces}
    if cf:
        co = _dup(src, part + '_cards_src')
        _keep_faces(co, cf)
        _strip_unused_materials(co)
        join_quads(co.data)
        n = len(co.data.polygons)
        cards, ci = api.reduce_character(co, target_faces=max(8, int(n * card_ratio)), name=part + '_cards')
        bpy.data.objects.remove(co)
        used += len(cards.data.polygons)
        info['cards'] = len(cards.data.polygons)
    # kept components -> run units
    keep = [c for c in cs if c.kind == 'keep']
    bvh_all = None
    if sheets or any(c.kind == 'shell' for c in cs):
        bm = bmesh.new()
        bm.from_mesh(src.data)
        bm.transform(src.matrix_world)
        pf = {f for c in cs if c.kind != 'cards' for f in c.faces}
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.index not in pf], context='FACES')
        bvh_all = BVHTree.FromBMesh(bm)
        bm.free()
    units = []
    done = set()
    part_area = sum(c.area for c in cs)
    for c in [c for c in cs if c.kind == 'shell']:
        fs = list(c.faces) + [f for x in cs if x.kind == 'relief' and x.host is c for f in x.faces]
        nb = max(1, len(c.bones))
        nf = len(fs) // 2
        units.append(dict(obj=None, sheet=None, faces=nf, inner=0, area=c.area, bones=nb, shell=(fs, c.mat),
                          w=(c.area ** density_gamma) * (nf ** (1 - density_gamma)) * (1.0 + (nb - 1) / math.sqrt(max(c.area, 1e-9))) ** bone_power))
    for c in sorted(keep, key=lambda c: -c.area):
        if id(c) in done:
            continue
        g = [c]
        m = _mirror_partner(c, keep)
        if m is not None and id(m) not in done:
            g.append(m)
        done.update(id(x) for x in g)
        o = _dup(src, '%s_u%d' % (part, len(units)))
        _keep_faces(o, {f for x in g for f in x.faces})
        area = sum(x.area for x in g)
        # only large cloth is turned into a sheet; small thin solids (fins, blades) stay closed
        sh = sheetify(o, bvh_all, T, log=log) if sheets and area >= sheet_min_area * part_area else None
        join_quads(o.data)
        nf = len(o.data.polygons)
        ninner = sum(1 for a, _ in inner_flags(o, sh) if a) if sh else 0
        nb = max(1, len(set().union(*[x.bones for x in g])))
        units.append(dict(obj=o, sheet=sh, faces=nf, inner=ninner, area=area, bones=nb,
                          w=(area ** density_gamma) * ((nf + ninner) ** (1 - density_gamma)) * (1.0 + (nb - 1) / math.sqrt(max(area, 1e-9))) ** bone_power))
    # distribute the budget (units cannot get more than they have)
    budget = max(16, target - used)
    free = list(units)
    left = budget
    for u in units:
        u['target'] = None
        u['w'] = max(u['w'], 0.0)
    while free:
        tw = sum(u['w'] for u in free)
        over = [u for u in free if left * u['w'] / tw >= u['faces'] + u['inner']]
        under = [u for u in free if left * u['w'] / tw < unit_min_ratio * (u['faces'] + u['inner'])]
        if not over and not under:
            for u in free:
                u['target'] = left * u['w'] / tw
            break
        for u in over:
            u['target'] = u['faces'] + u['inner']
        for u in under:                     # nothing is reduced below unit_min_ratio of its own faces
            u['target'] = unit_min_ratio * (u['faces'] + u['inner'])
        for u in over + under:
            left -= u['target']
            free.remove(u)
        left = max(left, 1.0)
    t0 = time.time()

    def total_of(u, r):
        return len(r.data.polygons) + (sum(1 for a, _t in inner_flags(r, u['sheet']) if a) if u['sheet'] is not None else 0)

    for k, u in enumerate(units):
        if u.get('shell'):
            fs, mi_ = u['shell']
            u['res'] = envelope_shell(src, fs, '%s_shell%d' % (part, k), int(round(min(u['target'], shell_max_faces))), bvh_all, T, mat_index=mi_, log=log)
            u['stuck'], u['sym'] = True, None
            continue
        o = u['obj']
        tgt = int(round(u['target'] * u['faces'] / (u['faces'] + u['inner'])))
        ex = dict(extra)
        if u['bones'] <= 2 and rigid_feature:
            # rigid prop (weapon): nothing bends, so the shape error is judged against one small
            # absolute size everywhere instead of the local thickness.  With the character rule
            # thin ornaments were protected and the big plates (crossbow limbs) paid for it
            ex.setdefault('max_feature', rigid_feature)
        u['extra'] = ex
        r, mi = reduce_loops(o, max(4, tgt), '%s_r%d' % (part, k), **ex)
        if u['sheet'] is not None:
            for _ in range(2):      # the inner side is counted on the result: correct the estimate
                tot = total_of(u, r)
                if tot <= 1.03 * u['target'] or mi.get('stuck'):
                    break
                r2, mi = reduce_loops(r, int(len(r.data.polygons) * u['target'] / tot), r.name + 'b', **ex)
                bpy.data.objects.remove(r)
                r = r2
        u['res'], u['stuck'], u['sym'] = r, bool(mi.get('stuck')), mi.get('symmetry_axis')
        bpy.data.objects.remove(o)
    # units that got stuck above their target: take the excess from the others
    tots = [total_of(u, u['res']) for u in units]
    excess = sum(tots) - budget
    flex = [i for i, u in enumerate(units) if not u['stuck'] and tots[i] >= 40]
    if excess > 0.01 * budget and flex:
        f = max(0.8, 1.0 - excess / sum(tots[i] for i in flex))
        for i in flex:
            u = units[i]
            r = u['res']
            r2, mi = reduce_loops(r, int(len(r.data.polygons) * f), r.name + 'c', **u['extra'])
            bpy.data.objects.remove(r)
            u['res'], u['stuck'] = r2, bool(mi.get('stuck'))
    for k, u in enumerate(units):
        r = u['res']
        if u['sheet'] is not None:
            rethicken(r, u['sheet'], log=log)
        info['units'].append({'faces_src': u['faces'], 'inner_src': u['inner'], 'target': round(u['target']), 'faces': len(r.data.polygons),
                              'stuck': u['stuck'], 'sym': u['sym'], 'sheet': u['sheet'] is not None})
        log('  %s unit %d: %s' % (part, k, info['units'][-1]))
        pieces.insert(k, r)
    info['seconds_reduce'] = round(time.time() - t0, 1)
    for p in pieces:
        for m in list(p.modifiers):
            p.modifiers.remove(m)
    _activate(pieces[0])
    for p in pieces[1:]:
        p.select_set(True)
    if len(pieces) > 1:
        bpy.ops.object.join()
    low = pieces[0]
    low.name = low.data.name = part + '_low'
    return low, cards, info


# --------------------------------------------------------------- quality
def topo_metrics(me, sliver_angle=15.0):
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.normal_update()
    q = sum(1 for f in bm.faces if len(f.verts) == 4)
    t = sum(1 for f in bm.faces if len(f.verts) == 3)
    n = len(bm.faces) - q - t
    mins = []
    sliver = 0
    for f in bm.faces:
        a = min(l.calc_angle() for l in f.loops)
        mins.append(math.degrees(a))
        if math.degrees(a) < sliver_angle:
            sliver += 1
    el = np.array([e.calc_length() for e in bm.edges])
    out = {'faces': len(bm.faces), 'quads': q, 'tris': t, 'ngons': n,
           'quad_pct': round(100.0 * q / max(1, len(bm.faces)), 1), 'tri_equiv': 2 * q + t + sum(len(f.verts) - 2 for f in bm.faces if len(f.verts) > 4),
           'nonmanifold_edges': sum(1 for e in bm.edges if len(e.link_faces) > 2),
           'boundary_edges': sum(1 for e in bm.edges if len(e.link_faces) == 1),
           'loose': sum(1 for e in bm.edges if not e.link_faces) + sum(1 for v in bm.verts if not v.link_edges),
           'inconsistent_edges': sum(1 for e in bm.edges if len(e.link_faces) == 2 and not e.is_contiguous),
           'min_angle_deg': round(min(mins), 1) if mins else 0, 'min_angle_p5_deg': round(float(np.percentile(mins, 5)), 1) if mins else 0,
           'slivers': sliver, 'edge_len_cv': round(float(el.std() / el.mean()), 2) if len(el) else 0}
    bm.free()
    return out


def lift_to_relief(low, bvh_relief, height, quantile=0.7, log=print):
    """The relief pieces were removed and get baked onto their host.  Where
    they cover the host densely (strands on the hair, a badge on the hat) the
    host surface is moved out to the top of the relief: the silhouette keeps
    the volume and the bake projects straight down instead of across a gap.
    Per vertex: the `quantile` of the relief height sampled over the faces
    around it (sparse relief -> little lift), then smoothed once."""
    mw = low.matrix_world
    inv = mw.inverted()
    n3 = mw.to_3x3().inverted().transposed()
    me = low.data
    samples = collections.defaultdict(list)
    for p in me.polygons:
        n = (n3 @ p.normal).normalized()
        cs = [mw @ me.vertices[v].co for v in p.vertices]
        c = mw @ p.center
        for q in [c] + [c.lerp(x, 0.6) for x in cs] + [c.lerp(x, 0.95) for x in cs]:
            o = q + n * 1e-5
            gone = 0.0
            top = 0.0
            for _ in range(8):
                h = bvh_relief.ray_cast(o, n, height - gone)
                if h[0] is None:
                    break
                gone += h[3] + 1e-5
                top = gone
                o = h[0] + n * 1e-5
            for v in p.vertices:
                samples[v].append(top)
    lift = np.zeros(len(me.vertices))
    for v, tops in samples.items():
        lift[v] = float(np.quantile(tops, quantile))
    nb = collections.defaultdict(set)
    for e in me.edges:
        a_, b_ = e.vertices
        nb[a_].add(b_)
        nb[b_].add(a_)
    sm = lift.copy()
    for v, ns in nb.items():
        sm[v] = 0.5 * lift[v] + 0.5 * float(np.mean([lift[x] for x in ns]))
    moved = 0
    for v in me.vertices:
        d = sm[v.index]
        if d < 0.001:
            continue
        n = (n3 @ v.normal).normalized()
        v.co = inv @ (mw @ v.co + n * d)
        moved += 1
    me.update()
    log('  lift to relief: %d vertices, max %.1f mm' % (moved, 1000 * float(sm.max()) if len(sm) else 0))
    return moved


# ------------------------------------------------------------ UV + bake
def _material_texture_size(mat):
    if mat is None or not mat.use_nodes:
        return 0
    b = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if b is None or not b.inputs['Base Color'].is_linked:
        return 0
    n = b.inputs['Base Color'].links[0].from_node
    return max(n.image.size[:]) if n.type == 'TEX_IMAGE' and n.image is not None else 0


def atlas(obj, margin=0.003, new_uv_density=1.0):
    """One UV atlas from the UV islands of the source: the islands of every
    material are scaled by the resolution of that material's texture (so the
    texel density the artist chose is kept) and packed together; overlapping
    (mirrored) islands get their own place, every texel is baked once.
    Then smooth shading with hard edges only on island borders."""
    me = obj.data
    while len(me.uv_layers) > 1:
        me.uv_layers.remove(me.uv_layers[-1])
    sizes = [_material_texture_size(m) for m in me.materials]
    big = max(sizes + [1])
    scale = [(sz / big) if sz else 0.25 for sz in sizes] or [1.0]
    uv = me.uv_layers[0]
    nl = len(me.loops)
    a = np.empty(nl * 2, dtype=np.float32)
    uv.uv.foreach_get('vector', a)
    a = a.reshape(nl, 2)
    for p in me.polygons:
        k = scale[p.material_index] if p.material_index < len(scale) else 1.0
        a[p.loop_start:p.loop_start + p.loop_total] *= k
    # pieces with UVs made here (bundle hull, envelope shell): same texel density as the rest
    at = me.attributes.get(NEW_UV)
    if at is not None:
        flag = np.zeros(len(me.polygons), dtype=np.int32)
        at.data.foreach_get('value', flag)
        sc = max(abs(x) for x in obj.matrix_world.to_scale()) or 1.0
        dens = [[0.0, 0.0], [0.0, 0.0]]
        for p in me.polygons:
            q = a[p.loop_start:p.loop_start + p.loop_total]
            x, y = q[:, 0], q[:, 1]
            a2 = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
            if a2 > 1e-12:
                k = 1 if flag[p.index] else 0
                dens[k][0] += a2
                dens[k][1] += p.area * sc * sc
        if dens[0][1] > 0 and dens[1][1] > 0 and dens[1][0] > 0:
            k = math.sqrt((dens[0][0] / dens[0][1]) / (dens[1][0] / dens[1][1])) * new_uv_density
            for p in me.polygons:
                if flag[p.index]:
                    a[p.loop_start:p.loop_start + p.loop_total] *= k
    uv.uv.foreach_set('vector', a.ravel())
    _activate(obj)
    if me.has_custom_normals:        # normals transferred from the source do not fit the new faces
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.mark_seam(clear=True)
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.uv.seams_from_islands(mark_seams=True, mark_sharp=False)
    try:
        bpy.ops.uv.pack_islands(rotate=True, margin=margin, merge_overlap=False, shape_method='CONCAVE')
    except TypeError:
        bpy.ops.uv.pack_islands(rotate=True, margin=margin)
    bpy.ops.object.mode_set(mode='OBJECT')
    bm = bmesh.new()
    bm.from_mesh(me)
    for f in bm.faces:
        f.smooth = True
    for e in bm.edges:
        e.smooth = True
        if len(e.link_faces) == 2:
            ang = e.calc_face_angle(0.0)
            # island border with a real angle, or a fold so sharp that no common normal /
            # cage direction exists for its two sides
            if (e.seam and ang > math.radians(40)) or ang > math.radians(FOLD_ANGLE):
                e.smooth = False
    bm.to_mesh(me)
    bm.free()


def _bvh_of(obj, faces=None):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    if faces is not None:
        bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.index not in faces], context='FACES')
    bm.normal_update()
    t = BVHTree.FromBMesh(bm)
    bm.free()
    return t


class HighSurf:
    """the kept high surface in world space with face adjacency (for walking on one layer)"""

    def __init__(self, obj, faces):
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.transform(obj.matrix_world)
        bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.index not in faces], context='FACES')
        bm.normal_update()
        bm.faces.ensure_lookup_table()
        self.bm = bm
        self.bvh = BVHTree.FromBMesh(bm)
        self.cen = [f.calc_center_median() for f in bm.faces]
        cosf = math.cos(math.radians(FOLD_ANGLE))
        self.adj = []
        for f in bm.faces:
            a = []
            for e in f.edges:
                for g in e.link_faces:
                    if g is not f and g.normal.dot(f.normal) > cosf:
                        a.append(g.index)
            self.adj.append(a)
        self.nrm = [f.normal.copy() for f in bm.faces]

    def own(self, pts, n, centre, radius):
        """high faces of the layer the points lie on: nearest faces that look the
        same way as n, grown over the surface (not around folds) within radius"""
        seeds = set()
        for q in pts:
            best = None
            for co, nn, i, d in self.bvh.find_nearest_range(q, max(0.004, 0.1 * radius)):
                if self.nrm[i].dot(n) > 0.2 and (best is None or d < best[1]):
                    best = (i, d)
            if best is None:
                r = self.bvh.find_nearest(q)
                if r[2] is not None and self.nrm[r[2]].dot(n) > -0.2:
                    best = (r[2], r[3])
            if best is not None:
                seeds.add(best[0])
        out = set(seeds)
        st = list(seeds)
        r2 = radius * radius
        while st:
            i = st.pop()
            for j in self.adj[i]:
                if j not in out and (self.cen[j] - centre).length_squared < r2:
                    out.add(j)
                    st.append(j)
        return out


def _surface_height(low, high):
    """per vertex of low: how far its OWN high surface lies above the low
    triangles around it (world units, >= 0).  Big reduced faces cut below the
    folds of the source, sometimes by more than the distance to the next cloth
    layer; a cage that starts under the own surface would bake the back of the
    layer behind it.  'Own' = the high faces reached by walking over the
    surface from the triangle's corners (which lie on it), see HighSurf.own.
    The triangles are the ones that get rasterised (a warped quad is two planes)."""
    mw = low.matrix_world
    n3 = mw.to_3x3().inverted().transposed()
    me = low.data
    need = np.zeros(len(me.vertices))
    me.calc_loop_triangles()
    for t in me.loop_triangles:
        n = (n3 @ t.normal).normalized()
        cs = [mw @ me.vertices[v].co for v in t.vertices]
        c = (cs[0] + cs[1] + cs[2]) / 3
        longest = max((cs[i] - cs[(i + 1) % 3]).length for i in range(3))
        own = high.own(cs, n, c, 1.2 * longest)
        if not own:
            continue
        reach = max(0.01, 0.35 * longest)
        pts = [c] + [c.lerp(q, 0.5) for q in cs] + [c.lerp(q, 0.92) for q in cs] + \
              [c.lerp(cs[i].lerp(cs[(i + 1) % 3], 0.5), 0.9) for i in range(3)]
        top = 0.0
        for q in pts:
            o = q + n * 1e-5
            gone = 0.0
            for _ in range(6):
                h = high.bvh.ray_cast(o, n, reach - gone)
                if h[0] is None:
                    break
                gone += h[3] + 1e-5
                if h[2] in own and h[1].dot(n) > 0.2:
                    top = max(top, gone)
                    break
                o = h[0] + n * 1e-5
        for v in t.vertices:
            if top > need[v]:
                need[v] = top
    return need


def make_cage(low, high, bvh_relief, e_base, e_relief, e_min=0.0004, margin=0.002, pair_tol=0.0025):
    """bake cage with a per vertex extrusion:
    - above the high surface wherever the reduced faces cut below it
    - far enough for the relief that was dropped (e_relief near it, e_base elsewhere)
    - but not further than half way to the underside of another layer above
      the surface (trousers under the coat, coat under the cape): otherwise
      the wrong layer is baked."""
    cage = _dup(low, low.name + '_cage')
    mw = low.matrix_world
    inv = mw.inverted()
    n3 = mw.to_3x3().inverted().transposed()
    need = _surface_height(low, high)
    bvh_block = high.bvh
    stats = collections.Counter()
    # extrusion direction: the one that leans least away from any face around the vertex
    # (the plain vertex normal can be almost tangent to a face on pleats and torn hems)
    fn = collections.defaultdict(list)
    for p in low.data.polygons:
        pn = (n3 @ p.normal).normalized()
        for vi in p.vertices:
            fn[vi].append(pn)
    for v in cage.data.vertices:
        p = mw @ v.co
        ns = fn.get(v.index)
        if not ns:
            continue
        n = sum(ns, Vector((0, 0, 0)))
        n = n.normalized() if n.length > 1e-9 else ns[0]
        for _ in range(12):
            worst = min(ns, key=lambda a: a.dot(n))
            if worst.dot(n) > 0.8:
                break
            n = (n + 0.4 * worst).normalized()
        lean = max(0.35, min(a.dot(n) for a in ns))
        E = e_base
        if bvh_relief is not None and bvh_relief.find_nearest(p, e_relief)[0] is not None:
            E = e_relief
            stats['relief'] += 1
        q = p + n * 1e-5
        gone = 0.0
        e = E
        for _ in range(8):
            h = bvh_block.ray_cast(q, n, E - gone)
            if h[0] is None:
                break
            gone += h[3]
            if h[1].dot(n) < 0.0 and gone > pair_tol:      # underside of another layer
                e = max(e_min, 0.5 * gone)
                stats['blocked'] += 1
                break
            q = h[0] + n * 1e-5
            gone += 1e-5
        if need[v.index] + margin > e:
            e = need[v.index] + margin
            stats['raised'] += 1
        v.co = inv @ (p + n * (e / lean))
    return cage, dict(stats)


def _bake_material(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    return m


def _new_image(name, size, data=False):
    img = bpy.data.images.new(name, size, size, alpha=True, is_data=data)
    if data:
        img.colorspace_settings.name = 'Non-Color'
    return img


def _push_pull(rgb, ok):
    """fill the texels that are not ok from the valid ones around them, at any distance
    (image pyramid: average of the valid texels going down, taken back going up)"""
    h, w = ok.shape
    if ok.all() or h < 2 or w < 2 or not ok.any():
        return rgb
    wgt = ok.astype(np.float32)
    c = rgb * wgt[..., None]
    h2, w2 = h // 2, w // 2
    cs = c[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, -1).sum((1, 3))
    ws = wgt[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).sum((1, 3))
    low = cs / np.maximum(ws, 1e-6)[..., None]
    low = _push_pull(low, ws > 0)
    up = np.repeat(np.repeat(low, 2, 0), 2, 1)
    out = rgb.copy()
    pad = np.zeros_like(rgb)
    pad[:h2 * 2, :w2 * 2] = up
    if h2 * 2 < h:
        pad[h2 * 2:] = pad[h2 * 2 - 1:h2 * 2]
    if w2 * 2 < w:
        pad[:, w2 * 2:] = pad[:, w2 * 2 - 1:w2 * 2]
    out[~ok] = pad[~ok]
    return out


def _uv_mask(obj, size):
    """texels covered by the UV faces of obj (plus one texel around them)"""
    me = obj.data
    m = np.zeros((size, size), dtype=bool)
    uv = np.empty(len(me.loops) * 2, dtype=np.float32)
    me.uv_layers[0].uv.foreach_get('vector', uv)
    uv = uv.reshape(-1, 2) * size
    me.calc_loop_triangles()
    for t in me.loop_triangles:
        p = uv[list(t.loops)]
        x0, y0 = np.floor(p.min(0)).astype(int) - 1
        x1, y1 = np.ceil(p.max(0)).astype(int) + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, size - 1), min(y1, size - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = (p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1])
        if abs(d) < 1e-9:
            continue
        l0 = ((p[1, 1] - p[2, 1]) * (xs - p[2, 0]) + (p[2, 0] - p[1, 0]) * (ys - p[2, 1])) / d
        l1 = ((p[2, 1] - p[0, 1]) * (xs - p[2, 0]) + (p[0, 0] - p[2, 0]) * (ys - p[2, 1])) / d
        e = 1.0 / max(1.0, math.sqrt(abs(d)))          # ~ one texel in barycentric units
        m[y0:y1 + 1, x0:x1 + 1] |= (l0 >= -e) & (l1 >= -e) & (l0 + l1 <= 1 + e)
    return m


def _fill_unbaked(img, key='BC', mask=None, regions=(), shade=()):
    """Texels no ray reached get the values of the baked texels around them.
    mask: baked texels (None: taken from alpha, which only the colour bake provides).
    Only texels every sample hit count as baked: on the edge of a strand, hit by half of
    the samples, the colour is mixed with black.  Normal map: texels whose normal points
    INTO the surface (a ray that went through a gap and hit the back of a layer) are
    rebuilt from their neighbours as well.
    regions: texel masks (one per bake group); the holes inside a region are filled from
    that region only (hair from hair, not from the coat that is next to it in the atlas)
    and blurred.  shade: per region, factor for the colour / AO of those holes: the gaps
    between the strands of an envelope shell are painted dark (depth), like on a hair
    texture; left at the strand colour the shell reads as a cloth hood.
    -> (missed fraction, mask)"""
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    ok = (px[..., 3] > 0.99) if mask is None else mask.copy()
    mask0 = ok.copy()
    missed = float((~ok).mean())
    if key == 'N':
        ok &= px[..., 2] > 0.6            # z = 2 * b - 1 > 0.2
    rgb = px[..., :3].copy()
    done = ok.copy()
    for ri, reg in enumerate(regions):
        hole = reg & ~ok
        if not hole.any() or not (reg & ok).any():
            continue
        f = _push_pull(rgb, reg & ok)
        for _ in range(12):                # soften the blocks of the pyramid
            g = f.copy()
            g[1:-1, 1:-1] = (f[1:-1, 1:-1] * 4 + f[:-2, 1:-1] + f[2:, 1:-1] + f[1:-1, :-2] + f[1:-1, 2:]) / 8
            g[reg & ok] = rgb[reg & ok]
            f = g
        rgb[hole] = f[hole] * (shade[ri] if key in ('BC', 'AO') and ri < len(shade) else 1.0)
        done |= hole
    px[..., :3] = _push_pull(rgb, done)
    if key == 'N':
        v = px[..., :3] * 2 - 1
        v /= np.maximum(np.linalg.norm(v, axis=2, keepdims=True), 1e-6)
        px[..., :3] = v * 0.5 + 0.5
    px[..., 3] = 1.0
    img.pixels.foreach_set(px.ravel())
    return missed, mask0


def bake_maps(low, highs, cage, out_prefix, size=2048, ao=True, max_ray=0.08, margin=3, samples=4, ao_samples=24, more=None, gap_shade=0.4, log=print):
    """Cycles selected-to-active: <out_prefix>_BC.PNG, _N.PNG (tangent space,
    includes the normal maps of the high materials), _AO.PNG.
    more: further (low, highs, cage) groups baked into the same images (their
    UVs share the atlas): faces that may only see their own sources.
    Returns {'BC': image, 'N': image, 'AO': image}"""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.use_denoising = False
    sc = max(abs(x) for x in low.matrix_world.to_scale()) or 1.0
    mat = _bake_material(os.path.basename(out_prefix))
    groups = [(low, highs, cage)] + list(more or [])
    for lo_, *_r in groups:
        lo_.data.materials.clear()
        lo_.data.materials.append(mat)
        for p in lo_.data.polygons:
            p.material_index = 0
    nt = mat.node_tree
    node = nt.nodes.new('ShaderNodeTexImage')
    nt.nodes.active = node
    hidden = {o.name: o.hide_render for o in scene.objects if o.type == 'MESH'}
    bk = scene.render.bake
    bk.use_selected_to_active = True
    bk.use_cage = True
    bk.max_ray_distance = max_ray
    bk.margin = margin
    bk.margin_type = 'EXTEND'
    bk.target = 'IMAGE_TEXTURES'
    out = {}
    mask = None
    regions = [_uv_mask(g[0], size) for g in groups] if len(groups) > 1 else []
    jobs = [('BC', 'DIFFUSE', False, samples), ('N', 'NORMAL', True, samples)]
    if ao:
        jobs.append(('AO', 'AO', True, ao_samples))
    for key, typ, data, smp in jobs:
        t0 = time.time()
        img = _new_image(os.path.basename(out_prefix) + '_' + key, size, data)
        px = np.zeros(size * size * 4, dtype=np.float32)
        img.pixels.foreach_set(px)
        node.image = img
        scene.cycles.samples = smp
        for gi, (lo_, hs_, cg_, *rest) in enumerate(groups):
            mr = rest[0] if rest else max_ray
            # only this group's sources exist for the rays (and for AO)
            for o in scene.objects:
                if o.type == 'MESH':
                    o.select_set(False)
                    o.hide_render = True
            for h in hs_:
                h.hide_set(False)
                h.hide_render = False
                h.select_set(True)
            lo_.hide_set(False)
            lo_.hide_render = False
            lo_.select_set(True)
            bpy.context.view_layer.objects.active = lo_
            bk.cage_object = cg_
            bk.use_clear = gi == 0
            bk.max_ray_distance = mr
            kw = dict(type=typ, use_selected_to_active=True, use_cage=True, cage_object=cg_.name,
                      max_ray_distance=mr, margin=margin, use_clear=gi == 0, target='IMAGE_TEXTURES')
            if typ == 'DIFFUSE':
                kw['pass_filter'] = {'COLOR'}
            if typ == 'NORMAL':
                kw['normal_space'] = 'TANGENT'
            bpy.ops.object.bake(**kw)
        missed, m0 = _fill_unbaked(img, key=key, mask=mask, regions=regions, shade=[1.0] + [gap_shade] * (len(groups) - 1))
        if key == 'BC':
            mask = m0
        img.filepath_raw = out_prefix + '_' + key + '.PNG'
        img.file_format = 'PNG'
        img.save()
        out[key] = img
        log('  bake %s: %.1fs, unbaked texels (empty atlas + rays without hit) %.1f%%' % (key, time.time() - t0, 100 * missed))
    for n, hr in hidden.items():
        bpy.data.objects[n].hide_render = hr
    # final material
    bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    node.image = out['BC']
    node.label = 'Base Color'
    nt.links.new(node.outputs['Color'], bsdf.inputs['Base Color'])
    nn = nt.nodes.new('ShaderNodeTexImage')
    nn.image = out['N']
    nn.label = 'Normal'
    nm = nt.nodes.new('ShaderNodeNormalMap')
    nt.links.new(nn.outputs['Color'], nm.inputs['Color'])
    nt.links.new(nm.outputs['Normal'], bsdf.inputs['Normal'])
    bsdf.inputs['Roughness'].default_value = 0.6
    nt.nodes.active = node
    return out


# ------------------------------------------------------------------ skin
def transfer_skin(low, high, limit=4):
    """weights of the high model -> low (nearest face interpolated), `limit`
    influences, normalised, same armature modifier"""
    for g in list(low.vertex_groups):
        low.vertex_groups.remove(g)
    for g in high.vertex_groups:
        low.vertex_groups.new(name=g.name)
    mod = low.modifiers.new('rt_skin', 'DATA_TRANSFER')
    mod.object = high
    mod.use_object_transform = False
    mod.use_vert_data = True
    mod.data_types_verts = {'VGROUP_WEIGHTS'}
    mod.vert_mapping = 'POLYINTERP_NEAREST'
    mod.layers_vgroup_select_src = 'ALL'
    mod.layers_vgroup_select_dst = 'NAME'
    with bpy.context.temp_override(object=low, active_object=low, selected_objects=[low]):
        bpy.ops.object.modifier_apply(modifier=mod.name)
    finish_skin(low, high, limit)


def finish_skin(low, high, limit=4):
    api.limit_skin_weights(low, limit)
    groups = low.vertex_groups
    for v in low.data.vertices:
        gs = [(g.group, g.weight) for g in v.groups if g.weight > 0]
        tot = sum(w for _, w in gs)
        if tot > 0 and abs(tot - 1.0) > 1e-6:
            for gi, w in gs:
                groups[gi].add([v.index], w / tot, 'REPLACE')
    for m in list(low.modifiers):
        if m.type == 'ARMATURE':
            low.modifiers.remove(m)
    for m in high.modifiers:
        if m.type == 'ARMATURE':
            nm = low.modifiers.new(m.name, 'ARMATURE')
            nm.object = m.object
    low.parent = high.parent
    low.matrix_parent_inverse = high.matrix_parent_inverse.copy()
    low.matrix_world = high.matrix_world.copy()


def skin_metrics(obj):
    n = len(obj.data.vertices)
    cnt = [sum(1 for g in v.groups if g.weight > 0) for v in obj.data.vertices]
    tot = [sum(g.weight for g in v.groups) for v in obj.data.vertices]
    return {'skinned_vertex_pct': round(100.0 * sum(1 for c in cnt if c > 0) / max(n, 1), 2), 'max_influences': max(cnt, default=0),
            'weight_sum_min': round(min(tot, default=0), 4), 'weight_sum_max': round(max(tot, default=0), 4)}


# -------------------------------------------------------------- pipeline
def retopo_part(src_raw, src, comps, part, target, out_prefix, size=2048, ao=True, relief_height=0.006, lift=True, log=print, **extra):
    """src_raw: untouched imported object (bake source, skin source)
    src: welded copy that `comps` (analyse) refers to.
    Writes <out_prefix>_BC/_N/_AO.PNG.  Returns (low object, high object, report)"""
    t0 = time.time()
    cs = [c for c in comps if c.part == part]
    w2r = source_faces(src.data)
    low, cards, info = build_low(src, comps, part, target, log=log, relief_height=relief_height, **extra)
    t1 = time.time()
    # high model of this part (unwelded: original shading), without the alpha cards
    hi = _dup(src_raw, part + '_high')
    _keep_faces(hi, {int(w2r[f]) for c in cs if c.kind != 'cards' for f in c.faces})
    hi_all = _dup(src_raw, part + '_high_all')
    _keep_faces(hi_all, {int(w2r[f]) for c in cs for f in c.faces})
    lo = np.min([c.lo for c in cs], 0)
    hi_ = np.max([c.hi for c in cs], 0)
    T = relief_height * float(np.linalg.norm(hi_ - lo))
    block = HighSurf(src, {f for c in cs if c.kind == 'keep' for f in c.faces})
    rf = {f for c in cs if c.kind in ('relief', 'bundle', 'shell') for f in c.faces}
    relief = _bvh_of(src, rf) if rf else None
    ro = {f for c in cs if c.kind == 'relief' and c.host.kind != 'shell' for f in c.faces}
    if ro and lift:
        lift_to_relief(low, _bvh_of(src, ro), 2.5 * T, log=log)
    atlas(low)
    # bake on a copy that is split along the folded-over hard edges (the cloth rim, where outer
    # and inner side share vertices): the two sides need opposite cage directions there.
    # Ordinary hard edges keep one averaged cage direction, so relief lying across them is covered
    blow = _dup(low, low.name + '_bake')
    bm = bmesh.new()
    bm.from_mesh(blow.data)
    bmesh.ops.split_edges(bm, edges=[e for e in bm.edges if not e.smooth and len(e.link_faces) == 2
                                     and e.calc_face_angle(0.0) > math.radians(FOLD_ANGLE - 0.5)])
    bm.to_mesh(blow.data)
    bm.free()
    cage, cst = make_cage(blow, block, relief, 0.6 * T, 2.5 * T)
    log('  cage: %s of %d verts' % (cst, len(blow.data.vertices)))
    # the envelope shell may only see what it replaces: through the gaps between the strands
    # its rays would otherwise reach the face, the mask or the coat (white patches in the hair).
    # Gaps stay unbaked and are filled from the strands next to them.
    more = []
    tmp = [cage, blow]
    sf = {f for c in cs if c.kind == 'shell' for f in c.faces} | \
         {f for c in cs if c.kind == 'relief' and c.host.kind == 'shell' for f in c.faces}
    at = blow.data.attributes.get(SHELL)
    if sf and at is not None:
        flag = np.zeros(len(blow.data.polygons), dtype=np.int32)
        at.data.foreach_get('value', flag)
        keep_s = {i for i in range(len(flag)) if flag[i]}
        if keep_s:
            hs = _dup(src_raw, part + '_high_shell')
            _keep_faces(hs, {int(w2r[f]) for f in sf})
            bs, cg = _dup(blow, blow.name + '_s'), _dup(cage, cage.name + '_s')
            keep_m = set(range(len(flag))) - keep_s
            _keep_faces(bs, keep_s)
            _keep_faces(cg, keep_s)
            _keep_faces(blow, keep_m)
            _keep_faces(cage, keep_m)
            more.append((bs, [hs], cg))
            tmp += [hs, bs, cg]
    imgs = bake_maps(blow, [hi], cage, out_prefix, size=size, ao=ao, max_ray=6 * T, more=more, log=log)
    low.data.materials.clear()
    low.data.materials.append(blow.data.materials[0])
    for p in low.data.polygons:
        p.material_index = 0
    if os.environ.get('RT_KEEP_CAGE'):         # debugging: keep the bake meshes and cages in the scene
        for o in tmp:
            o.hide_render = True
    else:
        for o in tmp:
            bpy.data.objects.remove(o)
    block.bm.free()
    t2 = time.time()
    transfer_skin(low, hi_all)
    if cards is not None:
        for m in list(cards.modifiers):
            cards.modifiers.remove(m)
        # cards keep their own material / UVs: second material slot
        uvn = low.data.uv_layers[0].name
        if cards.data.uv_layers:
            cards.data.uv_layers[0].name = uvn
        _activate(low)
        cards.select_set(True)
        bpy.ops.object.join()
        finish_skin(low, hi_all)
    rep = {'part': part, 'build': info, 'cage': cst,
           'seconds': {'geometry': round(t1 - t0, 1), 'uv_bake': round(t2 - t1, 1), 'total': round(time.time() - t0, 1)}}
    return low, hi_all, rep
