"""dev/retopo_verify.py OUTDIR [SRC.FBX]  -> re-import every *_retopo.FBX of OUTDIR and check that
armature, skin weights, UVs and textures survived the export (prints one JSON line per file)."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bpy
import numpy as np
from core import io_util, retopo

outdir = sys.argv[1]
src_bones = None
if len(sys.argv) > 2:
    io_util.reset_scene()
    io_util.import_file(sys.argv[2])
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    src_bones = {b.name: (arm.matrix_world @ b.head_local).to_tuple(4) for b in arm.data.bones}
for path in sorted(glob.glob(os.path.join(outdir, '*_retopo.FBX')) + glob.glob(os.path.join(outdir, '*_skel.FBX'))):      # core.retopo / core.skelretopo
    io_util.reset_scene()
    objs = io_util.import_file(path)
    o = objs[0]
    arm = next((x for x in bpy.data.objects if x.type == 'ARMATURE'), None)
    rep = {'file': os.path.basename(path), 'meshes': len(objs), 'faces': len(o.data.polygons),
           'quads': sum(1 for p in o.data.polygons if len(p.vertices) == 4), 'tris': sum(1 for p in o.data.polygons if len(p.vertices) == 3),
           'armature': arm.name if arm else None, 'bones': len(arm.data.bones) if arm else 0,
           'armature_modifier': any(m.type == 'ARMATURE' and m.object is arm for m in o.modifiers),
           'vertex_groups': len(o.vertex_groups), 'uv_layers': len(o.data.uv_layers)}
    rep.update(retopo.skin_metrics(o))
    if arm:
        names = {b.name for b in arm.data.bones}
        used = {o.vertex_groups[g.group].name for v in o.data.vertices for g in v.groups if g.weight > 0}
        rep['weighted_groups_without_bone'] = sorted(used - names)
        if src_bones:
            d = [np.linalg.norm(np.array((arm.matrix_world @ b.head_local)[:]) - np.array(src_bones[b.name])) for b in arm.data.bones if b.name in src_bones]
            rep['bones_matching_source'] = len(d)
            rep['bone_head_max_offset_m'] = round(float(max(d)), 6) if d else None
    tex = {}
    for s in o.material_slots:
        m = s.material
        imgs = {}
        if m and m.use_nodes:
            b = next((n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
            for sock in ('Base Color', 'Normal', 'Alpha'):
                if b is None or not b.inputs[sock].is_linked:
                    continue
                n = b.inputs[sock].links[0].from_node
                if n.type == 'NORMAL_MAP' and n.inputs['Color'].is_linked:
                    n = n.inputs['Color'].links[0].from_node
                if n.type == 'TEX_IMAGE' and n.image:
                    p = bpy.path.abspath(n.image.filepath)
                    imgs[sock] = '%s (%dx%d, %s)' % (os.path.basename(p), n.image.size[0], n.image.size[1], 'found' if os.path.isfile(p) else 'MISSING')
        tex[m.name if m else '-'] = imgs
    rep['materials'] = tex
    # pose test: rotate a bone, the mesh must follow
    if arm:
        arm.data.pose_position = 'POSE'          # import_file leaves every armature in its rest position
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        before = np.array([v.co[:] for v in o.evaluated_get(dg).data.vertices])
        used_b = [arm.pose.bones[n] for n in sorted(used & names)]
        pb = arm.pose.bones.get('Bip001-Spine1') if 'Bip001-Spine1' in used else None
        pb = pb or (used_b[0] if used_b else arm.pose.bones[0])          # a bone this mesh is weighted to
        pb.rotation_mode = 'XYZ'
        pb.rotation_euler = (0.4, 0.2, 0.3)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        after = np.array([v.co[:] for v in o.evaluated_get(dg).data.vertices])
        rep['pose_test_bone'] = pb.name
        rep['pose_test_moved_vertex_pct'] = round(100 * float((np.linalg.norm(after - before, axis=1) > 1e-4).mean()), 1)
    # mesh hygiene of the exported file (round 5): nothing degenerate / loose / non manifold, every closed
    # component with its normals outward (positive signed volume), no component wound inconsistently
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.transform(o.matrix_world)
    diag = float(np.linalg.norm(np.ptp(np.array([v.co[:] for v in bm.verts]), axis=0))) if bm.verts else 1.0
    seen, inside_out, closed_n, comps_n = set(), 0, 0, 0
    for f0 in bm.faces:
        if f0 in seen:
            continue
        comps_n += 1
        st, fs = [f0], []
        seen.add(f0)
        while st:
            g = st.pop()
            fs.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h not in seen:
                        seen.add(h)
                        st.append(h)
        if all(len(e.link_faces) == 2 for g in fs for e in g.edges):
            closed_n += 1
            vol = 0.0
            for g in fs:
                vv = [l.vert.co for l in g.loops]
                for k in range(1, len(vv) - 1):
                    vol += vv[0].dot(vv[k].cross(vv[k + 1]))
            inside_out += vol < 0
    rep['hygiene'] = {'components': comps_n, 'closed_components': closed_n, 'inside_out_components': int(inside_out),
                      'degenerate_faces': sum(1 for f in bm.faces if f.calc_area() < 1e-10 * diag * diag),
                      'zero_length_edges': sum(1 for e in bm.edges if e.calc_length() < 1e-7 * diag),
                      'loose_verts': sum(1 for v in bm.verts if not v.link_faces), 'loose_edges': sum(1 for e in bm.edges if not e.link_faces),
                      'nonmanifold_edges': sum(1 for e in bm.edges if len(e.link_faces) > 2),
                      'inconsistent_winding_edges': sum(1 for e in bm.edges if len(e.link_faces) == 2 and not e.is_contiguous),
                      'ngons': sum(1 for f in bm.faces if len(f.verts) > 4)}
    bm.free()
    hy = rep['hygiene']
    ok = (rep['meshes'] == 1 and rep['uv_layers'] >= 1 and (arm is None or (rep['armature_modifier'] and rep['skinned_vertex_pct'] == 100.0 and rep['max_influences'] <= 4
          and not rep['weighted_groups_without_bone'] and abs(rep['weight_sum_min'] - 1) < 1e-3 and abs(rep['weight_sum_max'] - 1) < 1e-3))
          and all('MISSING' not in t for im in tex.values() for t in im.values())
          and not any(hy[k] for k in ('inside_out_components', 'degenerate_faces', 'zero_length_edges', 'loose_verts', 'loose_edges', 'nonmanifold_edges', 'inconsistent_winding_edges', 'ngons')))
    rep['pass'] = bool(ok)
    print('VERIFY', 'PASS' if ok else 'FAIL', json.dumps(rep, ensure_ascii=False))
