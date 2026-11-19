"""High level entry points shared by the add-on, the runner and the CLI."""
import time
import numpy as np
import bpy

from .rmesh import RMesh
from . import character, fit as fitmod


def _link_like(new, src):
    cols = list(src.users_collection) or [bpy.context.scene.collection]
    for c in cols:
        c.objects.link(new)
    new.matrix_world = src.matrix_world.copy()
    new.parent = src.parent
    new.matrix_parent_inverse = src.matrix_parent_inverse.copy()
    new.matrix_world = src.matrix_world.copy()


class rest_pose:
    """Context manager: every armature in its rest position.  The data transfer modifier
    reads the EVALUATED source; in a file that carries an animation the source stands in
    the pose of the current frame while the new mesh is in the bind pose, so weights and
    normals would be taken from the wrong places (43% of the normals pointed into the
    surface on a model exported with its walk cycle)."""

    def __enter__(self):
        self.saved = {a.name: a.pose_position for a in bpy.data.armatures}
        for a in bpy.data.armatures:
            a.pose_position = 'REST'
        bpy.context.view_layer.update()
        return self

    def __exit__(self, *exc):
        for n, p in self.saved.items():
            a = bpy.data.armatures.get(n)
            if a is not None:
                a.pose_position = p
        bpy.context.view_layer.update()
        return False


def _transfer_extras(src, new, flat):
    """vertex groups (skin weights), armature modifiers, custom normals."""
    me = new.data
    need_vg = len(src.vertex_groups) > 0
    need_nrm = (not flat) and src.data.has_custom_normals
    if need_vg:
        for g in src.vertex_groups:
            new.vertex_groups.new(name=g.name)
    if need_vg or need_nrm:
        mod = new.modifiers.new('mr_transfer', 'DATA_TRANSFER')
        mod.object = src
        mod.use_object_transform = False
        if need_vg:
            mod.use_vert_data = True
            mod.data_types_verts = {'VGROUP_WEIGHTS'}
            mod.vert_mapping = 'POLYINTERP_NEAREST'
            mod.layers_vgroup_select_src = 'ALL'
            mod.layers_vgroup_select_dst = 'NAME'
        if need_nrm:
            mod.use_loop_data = True
            mod.data_types_loops = {'CUSTOM_NORMAL'}
            mod.loop_mapping = 'POLYINTERP_NEAREST'
        ctx = bpy.context
        with rest_pose(), ctx.temp_override(object=new, active_object=new, selected_objects=[new]):
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except Exception as ex:           # keep going, extras are optional
                print('WARN transfer failed:', ex)
                new.modifiers.remove(mod)
    for m in src.modifiers:
        if m.type == 'ARMATURE':
            nm = new.modifiers.new(m.name, 'ARMATURE')
            nm.object = m.object
            nm.use_vertex_groups = m.use_vertex_groups


def max_influences(obj):
    return max((sum(1 for g in v.groups if g.weight > 0) for v in obj.data.vertices), default=0)


def limit_skin_weights(obj, limit=4):
    """keep the `limit` strongest vertex groups per vertex and renormalise
    (interpolated weights can use more bones than the source did; game engines
    cut at 4 by default).  Returns the number of vertices changed."""
    changed = 0
    groups = obj.vertex_groups
    for v in obj.data.vertices:
        gs = [(g.weight, g.group) for g in v.groups if g.weight > 0]
        if len(gs) <= limit:
            continue
        gs.sort(reverse=True)
        keep, drop = gs[:limit], gs[limit:]
        tot = sum(w for w, _ in gs)
        ksum = sum(w for w, _ in keep) or 1.0
        for _w, gi in drop:
            groups[gi].remove([v.index])
        for w, gi in keep:
            groups[gi].add([v.index], w * tot / ksum, 'REPLACE')
        changed += 1
    return changed


def reduce_character(obj, target_faces=None, ratio=0.5, symmetry='AUTO', protect_group=None,
                     preserve_seams=True, preserve_sharp=True, auto_joints=True,
                     protect_strength=4.0, joint_strength=5.0, do_fit=True, progress=None, name=None,
                     texture=None, finish_edges=False, **extra):
    """Quad/loop preserving reduction of a mesh object. Returns (new_obj, info).
    finish_edges: when the loops are used up above the target (info['stuck']), go on with
    single edge collapses (same seam / flip safeguards).  That reaches lower counts on
    game-ready low poly meshes but turns the quads it touches into triangles, which is
    what the artists rejected: off by default, the result reports 'stuck' instead."""
    t0 = time.time()
    me = obj.data
    mesh = RMesh.from_blender(me, use_sharp=preserve_sharp)
    # EXPERIMENTAL (off by default, it did not improve the measured texture error):
    # texture content aware cost. 'AUTO' = image of the object's material, or a file path
    from . import texture as texmod
    tex = None
    if texture == 'AUTO':
        img = texmod.find_image(obj)
        tex = texmod.load(img) if img is not None else None
    elif texture:
        tex = texmod.load(texture)
    protect = None
    if protect_group and protect_group in obj.vertex_groups:
        gi = obj.vertex_groups[protect_group].index
        protect = np.zeros(len(me.vertices))
        for v in me.vertices:
            for g in v.groups:
                if g.group == gi:
                    protect[v.index] = g.weight
    user_progress = progress
    if progress:
        def progress(f, msg=''):
            user_progress(0.05 + 0.78 * f, msg)
    params = character.Params(target_faces=target_faces, ratio=ratio, symmetry=symmetry,
                              preserve_seams=preserve_seams, preserve_sharp=preserve_sharp,
                              protect=protect, protect_strength=protect_strength,
                              auto_joints=auto_joints, joint_strength=joint_strength,
                              fit=do_fit, progress=progress, texture=tex, **extra)
    if user_progress:
        user_progress(0.02, 'analysing mesh')
    red = character.Reducer(mesh, params)
    red.run()
    loops_end = mesh.alive_faces
    if finish_edges and red.stuck and mesh.alive_faces > 1.05 * (params.target_faces or 0):
        red.run_edges()
    if user_progress:
        user_progress(0.85, 'fitting to original surface')
    if do_fit:
        fitmod.fit(red)
    new_me, vmap = mesh.to_blender(name or (me.name + '_reduced'), me, bpy)
    new = bpy.data.objects.new(name or (obj.name + '_reduced'), new_me)
    _link_like(new, obj)
    _transfer_extras(obj, new, mesh.flat_shaded)
    q, t, o = mesh.count()
    nflip, uvflip = red.self_check()
    info = {'mode': 'character', 'faces_flipped_vs_source': nflip, 'uv_faces_flipped_vs_source': uvflip, 'faces_before': len(me.polygons), 'faces_after': q + t + o,
            'quads': q, 'tris': t, 'ngons': o, 'target': params.target_faces or int(round(len(me.polygons) * ratio)),
            'stuck': bool(red.stuck), 'faces_after_loops': loops_end, 'triangle_budget_relaxed': getattr(red, 'n_tri_relax', 0),
            'texture_aware': tex is not None, 'symmetry_axis': None if red.sym_axis is None else 'XYZ'[red.sym_axis],
            'operations': len(red.log), 'seconds': round(time.time() - t0, 2)}
    if user_progress:
        user_progress(1.0, 'done')
    return new, info
