"""Static props (triangle soups, rocks, ...): attribute aware triangle decimation.

Uses Blender's own edge-collapse decimator (quadric error, interpolates all
UV layers) and then verifies / repairs the result.  Character meshes must not
go through here."""
import time
import bpy
import bmesh
import numpy as np


def _bad_uv_points(src_me, res_me, stretch_limit=1.0):
    """World-less (local) positions of result faces whose UV orientation is
    flipped against the original, or whose texel density changed by more than
    2**stretch_limit, in any UV layer."""
    from . import metrics
    A = metrics.TriMesh(src_me)
    B = metrics.TriMesh(res_me)
    diag = float(np.linalg.norm(A.pos.max(0) - A.pos.min(0)))
    p = B.pos[B.tri]
    cents = p.mean(1)
    nrm = B.poly_normal[B.tri_poly]
    tc, _d = A.closest_compatible(cents, nrm, diag)
    bad = np.zeros(len(B.tri), dtype=bool)
    nflip = 0
    for k in range(min(len(A.tri_uv), len(B.tri_uv))):
        def sarea(M, t):
            u = M.tri_uv[k][t]
            return 0.5 * ((u[:, 1, 0] - u[:, 0, 0]) * (u[:, 2, 1] - u[:, 0, 1]) - (u[:, 2, 0] - u[:, 0, 0]) * (u[:, 1, 1] - u[:, 0, 1]))
        ua = sarea(A, tc)
        ub = sarea(B, np.arange(len(B.tri)))
        flip = (ua * ub < 0) & (np.abs(ua) > 1e-12)
        nflip += int(flip.sum())
        da = np.abs(ua) / np.maximum(A.tri_area[tc], 1e-20)
        db = np.abs(ub) / np.maximum(B.tri_area, 1e-20)
        ok = (da > 1e-12) & (db > 1e-12)
        st = np.zeros(len(ub))
        st[ok] = np.abs(np.log2(np.sqrt(db[ok] / da[ok])))
        bad |= flip | (st > stretch_limit)
    pts = p[bad].reshape(-1, 3)
    A.free()
    B.free()
    return pts, nflip, int(bad.sum())


def _repair_uv(src_me, res_me, stretch_limit=1.0, passes=3):
    """Check every UV layer of the decimated mesh against the original: faces
    whose UV orientation flipped or whose texel density changed more than 2x
    get the UVs of their interior (non-seam) vertices re-sampled from the
    original surface.  Returns a small report."""
    from . import metrics
    A = metrics.TriMesh(src_me)
    diag = float(np.linalg.norm(A.pos.max(0) - A.pos.min(0)))
    report = {'bad_faces_before': 0, 'bad_faces_after': 0, 'vertices_resampled': 0}
    bm = bmesh.new()
    bm.from_mesh(res_me)
    bm.faces.ensure_lookup_table()
    layers = list(bm.loops.layers.uv.values())
    cents = np.array([f.calc_center_median()[:] for f in bm.faces]).reshape(-1, 3)
    nrm = np.array([f.normal[:] for f in bm.faces]).reshape(-1, 3)
    tc, _d = A.closest_compatible(cents, nrm, diag)
    farea = np.array([f.calc_area() for f in bm.faces])

    def face_uv_area(f, layer):
        s = 0.0
        ls = f.loops
        for i in range(len(ls)):
            p, q = ls[i][layer].uv, ls[(i + 1) % len(ls)][layer].uv
            s += p.x * q.y - q.x * p.y
        return 0.5 * s
    for it in range(passes + 1):
        total_bad = 0
        for k, layer in enumerate(layers):
            if k >= len(A.tri_uv):
                continue
            u = A.tri_uv[k][tc]
            ua = 0.5 * ((u[:, 1, 0] - u[:, 0, 0]) * (u[:, 2, 1] - u[:, 0, 1]) - (u[:, 2, 0] - u[:, 0, 0]) * (u[:, 1, 1] - u[:, 0, 1]))
            da = np.abs(ua) / np.maximum(A.tri_area[tc], 1e-20)
            bad = []
            for f in bm.faces:
                ub = face_uv_area(f, layer)
                i = f.index
                if abs(ua[i]) > 1e-12 and ua[i] * ub < 0:
                    bad.append(f)
                    continue
                db = abs(ub) / max(farea[i], 1e-20)
                if da[i] > 1e-12 and db > 1e-12 and abs(np.log2(np.sqrt(db / da[i]))) > stretch_limit:
                    bad.append(f)
            total_bad += len(bad)
            if it == passes:
                continue
            verts = {v for f in bad for v in f.verts}
            for v in verts:
                uvs = [l[layer].uv.copy() for l in v.link_loops]
                if not uvs or max((x - uvs[0]).length for x in uvs) > 1e-6:
                    continue                      # seam vertex: leave it
                ti, _ = A.closest_compatible(np.array([v.co[:]]), np.array([v.normal[:]]), diag)
                q, _t, bc, _dd = A.closest(np.array([v.co[:]]))
                tri = A.tri[ti[0]]
                bcc = metrics.bary(np.array([v.co[:]]), A.pos[tri][None])[0]
                new_uv = (A.tri_uv[k][ti[0]] * bcc[:, None]).sum(0)
                scale = max((l[layer].uv - l.link_loop_next[layer].uv).length for l in v.link_loops)
                if np.linalg.norm(new_uv - np.array(uvs[0][:])) > 4.0 * max(scale, 1e-9):
                    continue                      # the closest original point is on another island
                for l in v.link_loops:
                    l[layer].uv = new_uv
                report['vertices_resampled'] += 1
        if it == 0:
            report['bad_faces_before'] = total_bad
        report['bad_faces_after'] = total_bad
        if total_bad == 0:
            break
    bm.to_mesh(res_me)
    bm.free()
    A.free()
    return report


def reduce_prop(obj, target_faces=None, ratio=0.5, symmetry='NONE', preserve_seams=True,
                preserve_sharp=True, method='STRICT', do_fit=True, progress=None, name=None, weld=True,
                relax_seams='STUCK', material_targets=None, **_unused):
    """Triangle friendly reduction for static props.

    method 'STRICT' (default): our own edge-collapse decimator.  UV seams of
    every UV layer, borders and hard edges are constraints, a collapse that
    would flip a face or a UV triangle is rejected, UVs are interpolated wedge
    by wedge.  A few seconds for some thousand triangles.
    method 'FAST': Blender's Decimate modifier followed by a numeric UV check
    (it is quicker and geometrically a little tighter, but it lets a few UV
    triangles flip next to seams)."""
    if method == 'AUTO':
        # our own decimator keeps every seam exact but is not made for very strong
        # reductions (it gets stuck or folds thin parts); below a quarter of the faces
        # Blender's quadric decimator is clearly closer to the original (see HANDOFF)
        n = len(obj.data.polygons)
        r = (target_faces / max(1, n)) if target_faces else ratio
        method = 'FAST' if r < 0.25 else 'STRICT'
    if method == 'FAST':
        return _reduce_prop_blender(obj, target_faces, ratio, symmetry, preserve_seams, progress, name,
                                    weld=weld, material_targets=material_targets)
    from .rmesh import RMesh
    from . import character, fit as fitmod, api
    t0 = time.time()
    me = obj.data
    mesh = RMesh.from_blender(me, use_sharp=preserve_sharp, weld=weld)
    user_progress = progress
    if progress:
        def progress(f, msg=''):
            user_progress(0.05 + 0.78 * f, msg)
    params = character.Params(target_faces=target_faces, ratio=ratio, symmetry='NONE', auto_joints=False,
                              preserve_seams=preserve_seams, preserve_sharp=preserve_sharp,
                              seam_layers='ALL', cost_mode='qem', progress=progress,
                              relax_seams=relax_seams,
                              group_targets={int(k): int(v) for k, v in material_targets.items()} if material_targets else None)
    if user_progress:
        user_progress(0.02, 'analysing mesh')
    red = character.Reducer(mesh, params)
    red.run_edges()
    if do_fit:
        if user_progress:
            user_progress(0.85, 'fitting to original surface')
        fitmod.fit(red, n_samples=16000, iterations=3)
    # faces whose UV orientation differs from their own source face (must be 0)
    flips = 0
    for fid, vs in enumerate(mesh.faces):
        if vs is None:
            continue
        a0 = red._uv_areas(red.orig_fuv[fid])
        a1 = red._uv_areas(mesh.fuv[fid])
        for k in range(mesh.n_layers):
            if abs(a0[k]) > 1e-12 and a0[k] * a1[k] <= 0:
                flips += 1
    new_me, vmap = mesh.to_blender(name or (me.name + '_reduced'), me, bpy)
    new = bpy.data.objects.new(name or (obj.name + '_reduced'), new_me)
    api._link_like(new, obj)
    api._transfer_extras(obj, new, mesh.flat_shaded)
    q, t, o = mesh.count()
    info = {'mode': 'prop', 'method': 'STRICT', 'faces_before': len(me.polygons), 'faces_after': q + t + o,
            'quads': q, 'tris': t, 'ngons': o,
            'target': target_faces or int(round(len(me.polygons) * ratio)),
            'stuck': bool(red.stuck), 'uv_faces_flipped_vs_source': flips,
            'seconds': round(time.time() - t0, 2)}
    if user_progress:
        user_progress(1.0, 'done')
    return new, info


def weld_mesh(me, rel_dist=1e-7):
    """merge vertices that sit at the same position (in place).  Returns the
    number of vertices removed.  See RMesh.from_blender(weld=True): meshes
    exported split along UV seams / smoothing groups must be closed before any
    edge collapse or the surface tears open along those seams."""
    n0 = len(me.vertices)
    if n0 == 0:
        return 0
    co = np.empty(n0 * 3, dtype=np.float32)
    me.vertices.foreach_get('co', co)
    co = co.reshape(n0, 3)
    diag = float(np.linalg.norm(co.max(0) - co.min(0)))
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=max(diag * rel_dist, 1e-12))
    bm.to_mesh(me)
    bm.free()
    return n0 - len(me.vertices)


def _decimate_mesh(me, want, symmetry='NONE'):
    """Blender Decimate (collapse) of mesh datablock `me` to `want` polygons -> new mesh"""
    tmp = bpy.data.objects.new('mr_tmp_decimate', me)
    bpy.context.scene.collection.objects.link(tmp)
    try:
        n = len(me.polygons)
        r = min(1.0, want / max(1, n))
        out = None
        for _ in range(5):
            for m in list(tmp.modifiers):
                tmp.modifiers.remove(m)
            mod = tmp.modifiers.new('mr_decimate', 'DECIMATE')
            mod.decimate_type = 'COLLAPSE'
            mod.ratio = r
            mod.use_collapse_triangulate = False
            if symmetry in ('X', 'Y', 'Z'):
                mod.use_symmetry = True
                mod.symmetry_axis = symmetry
            dg = bpy.context.evaluated_depsgraph_get()
            ev = tmp.evaluated_get(dg)
            if out is not None:
                bpy.data.meshes.remove(out)
            out = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
            got = len(out.polygons)
            if abs(got - want) <= max(1, int(0.002 * want)) or got == 0:
                break
            r = min(1.0, r * want / max(1, got))
        return out
    finally:
        bpy.data.objects.remove(tmp)


def _reduce_prop_blender(obj, target_faces=None, ratio=0.5, symmetry='NONE', preserve_seams=True,
                         progress=None, name=None, weld=False, material_targets=None):
    t0 = time.time()
    src = obj.data
    me = src.copy()
    me.name = name or (src.name + '_reduced')
    welded = weld_mesh(me) if weld else 0
    new = bpy.data.objects.new(name or (obj.name + '_reduced'), me)
    for c in (list(obj.users_collection) or [bpy.context.scene.collection]):
        c.objects.link(new)
    new.parent = obj.parent
    new.matrix_parent_inverse = obj.matrix_parent_inverse.copy()
    new.matrix_world = obj.matrix_world.copy()
    for g in obj.vertex_groups:
        if g.name not in new.vertex_groups:
            new.vertex_groups.new(name=g.name)
    n_poly = len(src.polygons)
    n_tri = sum(p.loop_total - 2 for p in src.polygons)
    if target_faces:
        # the target counts polygons of the result (= triangles for a triangle mesh)
        r = min(1.0, max(0.0, target_faces / max(1, n_poly)))
    else:
        r = ratio
    if progress:
        progress(0.1, 'decimating')


    def run(rr):
        for m in list(new.modifiers):
            new.modifiers.remove(m)
        mod = new.modifiers.new('mr_decimate', 'DECIMATE')
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = rr

        mod.use_collapse_triangulate = False
        if symmetry in ('X', 'Y', 'Z'):
            mod.use_symmetry = True
            mod.symmetry_axis = symmetry
        dg = bpy.context.evaluated_depsgraph_get()
        ev = new.evaluated_get(dg)
        out = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
        return out
    want = int(round(n_poly * r))

    def run_to_target(r):
        out = run(r)
        for _ in range(3):
            got = len(out.polygons)
            if abs(got - want) <= max(2, int(0.002 * n_poly)):
                break
            bpy.data.meshes.remove(out)
            r = min(1.0, r * want / max(1, got))
            out = run(r)
        return out
    if material_targets:
        # every material gets its own budget: decimate the faces of each material
        # separately (materials never share an edge after such a split) and
        # put the pieces back into one mesh
        mt = {int(k): int(v) for k, v in material_targets.items()}
        mi = np.empty(len(me.polygons), dtype=np.int32)
        me.polygons.foreach_get('material_index', mi)
        acc = bmesh.new()
        for k in sorted(set(mi.tolist())):
            part = me.copy()
            bm = bmesh.new()
            bm.from_mesh(part)
            bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.material_index != k], context='FACES')
            bm.to_mesh(part)
            bm.free()
            nk = len(part.polygons)
            tk = mt.get(k, int(round(nk * r)))
            if tk < nk:
                dec = _decimate_mesh(part, max(1, tk), symmetry)
                bpy.data.meshes.remove(part)
                part = dec
            acc.from_mesh(part)
            bpy.data.meshes.remove(part)
        out = me.copy()
        acc.to_mesh(out)
        acc.free()
        want = len(out.polygons)
    else:
        out = run_to_target(r)
    uv_report = {}
    if preserve_seams and len(src.uv_layers):
        uv_report = _repair_uv(src, out)
    for m in list(new.modifiers):
        new.modifiers.remove(m)
    out.name = me.name
    new.data = out
    if me.users == 0:
        bpy.data.meshes.remove(me)
    vg = new.vertex_groups.get('mr_keep')
    if vg:
        new.vertex_groups.remove(vg)
    # custom normals: re-project from the original
    if src.has_custom_normals:
        mod = new.modifiers.new('mr_transfer', 'DATA_TRANSFER')
        mod.object = obj
        mod.use_object_transform = False
        mod.use_loop_data = True
        mod.data_types_loops = {'CUSTOM_NORMAL'}
        mod.loop_mapping = 'POLYINTERP_NEAREST'
        with bpy.context.temp_override(object=new, active_object=new, selected_objects=[new]):
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except Exception as ex:
                print('WARN normal transfer failed:', ex)
                new.modifiers.remove(mod)
    for m in obj.modifiers:
        if m.type == 'ARMATURE':
            nm = new.modifiers.new(m.name, 'ARMATURE')
            nm.object = m.object
    if len(obj.vertex_groups):
        from . import api
        api.limit_skin_weights(new, max(4, api.max_influences(obj)))
    q = sum(1 for p in new.data.polygons if p.loop_total == 4)
    t = sum(1 for p in new.data.polygons if p.loop_total == 3)
    info = {'mode': 'prop', 'method': 'FAST', 'faces_before': n_poly, 'faces_after': len(new.data.polygons), 'quads': q, 'tris': t,
            'ngons': len(new.data.polygons) - q - t, 'target': want, 'stuck': False, 'uv_repair': uv_report, 'welded_vertices': welded,
            'seconds': round(time.time() - t0, 2)}
    if progress:
        progress(1.0, 'done')
    return new, info
