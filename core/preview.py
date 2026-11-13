"""Headless preview rendering (Workbench) : textured + wireframe, several views,
original vs result side by side.  Only bpy + numpy are used (no PIL)."""
import bpy, os, math, tempfile
import numpy as np
from mathutils import Vector, Matrix

VIEWS = {
    # name: direction the camera looks FROM (world), Blender: -Y is front
    'front': Vector((0, -1, 0)), 'back': Vector((0, 1, 0)),
    'left': Vector((1, 0, 0)), 'right': Vector((-1, 0, 0)),
    'front34': Vector((0.7, -1, 0.35)), 'back34': Vector((-0.7, 1, 0.35)),
    'top': Vector((0, -0.001, 1)),
}


def _world_bbox(obj):
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = Vector((min(p[i] for p in pts) for i in range(3)))
    mx = Vector((max(p[i] for p in pts) for i in range(3)))
    return mn, mx


def _ensure_tex_material(obj, texture_path):
    """Give obj a material whose active image-texture node is texture_path
    (workbench TEXTURE colour mode uses the active image node)."""
    if not texture_path or not os.path.exists(texture_path):
        return
    img = bpy.data.images.load(texture_path, check_existing=True)
    mat = bpy.data.materials.new('mr_preview_mat')
    mat.use_nodes = True
    nt = mat.node_tree
    node = nt.nodes.new('ShaderNodeTexImage')
    node.image = img
    nt.nodes.active = node
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf:
        nt.links.new(node.outputs['Color'], bsdf.inputs['Base Color'])
    return mat


def _make_wire(obj, thickness_world, color=(0, 0, 0, 1)):
    me = obj.data.copy()
    w = bpy.data.objects.new(obj.name + '_wire', me)
    w.matrix_world = obj.matrix_world.copy()
    bpy.context.scene.collection.objects.link(w)
    me.materials.clear()
    mat = bpy.data.materials.new('mr_wire')
    mat.diffuse_color = color
    try:        # Cycles needs a node material: plain black emission
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes):
            if n.type != 'OUTPUT_MATERIAL':
                nt.nodes.remove(n)
        outn = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
        em = nt.nodes.new('ShaderNodeEmission')
        em.inputs[0].default_value = color
        em.inputs[1].default_value = 0.0 if sum(color[:3]) == 0 else 1.0
        nt.links.new(em.outputs[0], outn.inputs[0])
    except Exception:
        pass
    me.materials.append(mat)
    for p in me.polygons:
        p.material_index = 0
    mod = w.modifiers.new('w', 'WIREFRAME')
    sc = max(abs(s) for s in obj.matrix_world.to_scale()) or 1.0
    mod.thickness = thickness_world / sc
    mod.use_replace = True
    mod.use_even_offset = False
    mod.offset = 0.0
    return w


def _read_png(path):
    img = bpy.data.images.load(path)
    w, h = img.size
    a = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(a)
    bpy.data.images.remove(img)
    return a.reshape(h, w, 4)[::-1]  # top row first


def save_png(arr, path):
    h, w = arr.shape[:2]
    img = bpy.data.images.new('mr_out', w, h, alpha=True)
    img.pixels.foreach_set(np.ascontiguousarray(arr[::-1]).astype(np.float32).ravel())
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)


def render_views(objs, out_png, views=('front', 'back', 'left'), res=700,
                 texture=None, wire=True, textured=True, zoom_box=None, ref=None,
                 bg=(0.32, 0.33, 0.36), engine='WORKBENCH'):
    """objs: list of objects. Produces a montage: one row per view, one column
    per object.  zoom_box=(center Vector, half_size) for close-ups, else framed
    on `ref` (default objs[0]).  Returns out_png."""
    scene = bpy.context.scene
    cycles = engine.upper() == 'CYCLES'
    lights = []
    if cycles:
        # CPU fallback for machines without a usable GPU / OpenGL
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'
        scene.cycles.samples = 12
        scene.cycles.use_denoising = False
        scene.cycles.max_bounces = 1
    else:
        scene.render.engine = 'BLENDER_WORKBENCH'
        sh = scene.display.shading
        sh.light = 'STUDIO'
        sh.color_type = 'TEXTURE' if textured else 'MATERIAL'
        sh.show_cavity = False
        sh.show_shadows = False
        sh.show_backface_culling = False
        scene.display.render_aa = '8'
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'Standard'
    if scene.world is None:
        scene.world = bpy.data.worlds.new('w')
    scene.world.color = bg
    if cycles:
        scene.world.use_nodes = True
        bgn = next((n for n in scene.world.node_tree.nodes if n.type == 'BACKGROUND'), None)
        if bgn:
            bgn.inputs[0].default_value = (bg[0], bg[1], bg[2], 1.0)
            bgn.inputs[1].default_value = 1.6
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    ref = ref or objs[0]
    mn, mx = _world_bbox(ref)
    center = (mn + mx) / 2
    size = (mx - mn)
    diag = size.length
    half = None
    if zoom_box is not None:
        center, half = Vector(zoom_box[0]), zoom_box[1]

    cam_d = bpy.data.cameras.new('mr_cam')
    cam_d.type = 'ORTHO'
    cam = bpy.data.objects.new('mr_cam', cam_d)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam_d.clip_start = 0.001
    cam_d.clip_end = diag * 20 + 10

    tmpmats = []
    wires = []
    saved_mats = {}
    for o in objs:
        if texture and textured:
            mat = _ensure_tex_material(o, texture)
            if mat:
                saved_mats[o.name] = [s.material for s in o.material_slots]
                if o.material_slots:
                    for s in o.material_slots:
                        s.material = mat
                else:
                    o.data.materials.append(mat)
                tmpmats.append(mat)
        wires.append(_make_wire(o, diag * (0.0012 if half is None else 0.0012 * max(0.25, 2 * half / max(size))) ) if wire else None)

    all_objs = [o for o in scene.objects if o.type == 'MESH']
    hidden = {o.name: o.hide_render for o in all_objs}
    rows = []
    tmpdir = tempfile.mkdtemp()
    try:
        for vname in views:
            d = VIEWS[vname].normalized()
            cam.location = center + d * (diag * 3 + 1)
            cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
            if cycles:
                for l in lights:
                    bpy.data.objects.remove(l)
                lights.clear()
                ld = bpy.data.lights.new('mr_sun', 'SUN')
                ld.energy = 2.5
                lo = bpy.data.objects.new('mr_sun', ld)
                scene.collection.objects.link(lo)
                lo.rotation_euler = (-(d + Vector((0.3, 0.2, 0.6))).normalized()).to_track_quat('-Z', 'Y').to_euler()
                lights.append(lo)
            # extent in camera plane
            if half is None:
                rot = cam.rotation_euler.to_matrix()
                rx, ry = rot.col[0], rot.col[1]
                cs = [Vector((x, y, z)) for x in (mn.x, mx.x) for y in (mn.y, mx.y) for z in (mn.z, mx.z)]
                ex = max(abs((c - center).dot(rx)) for c in cs)
                ey = max(abs((c - center).dot(ry)) for c in cs)
                aspect = max(0.45, min(1.6, ex / ey))
                rx_px, ry_px = int(res * aspect), res
                cam_d.ortho_scale = 2.1 * max(ex, ey * rx_px / ry_px) if rx_px >= ry_px else 2.1 * max(ey, ex * ry_px / rx_px)
            else:
                rx_px, ry_px = res, res
                cam_d.ortho_scale = 2 * half
            scene.render.resolution_x, scene.render.resolution_y = rx_px, ry_px
            scene.render.resolution_percentage = 100
            cols = []
            for o, w in zip(objs, wires):
                for a in all_objs:
                    a.hide_render = True
                for ww in wires:
                    if ww:
                        ww.hide_render = True
                o.hide_render = False
                if w:
                    w.hide_render = False
                p = os.path.join(tmpdir, 'v.png')
                scene.render.filepath = p
                bpy.ops.render.render(write_still=True)
                cols.append(_read_png(p))
            sep = np.ones((cols[0].shape[0], 6, 4), np.float32)
            row = cols[0]
            for c in cols[1:]:
                row = np.concatenate([row, sep, c], axis=1)
            rows.append(row)
        W = max(r.shape[1] for r in rows)
        out = []
        for r in rows:
            if r.shape[1] < W:
                pad = np.ones((r.shape[0], W - r.shape[1], 4), np.float32)
                pad[..., :3] = bg
                r = np.concatenate([r, pad], axis=1)
            out.append(r)
            out.append(np.ones((6, W, 4), np.float32))
        save_png(np.concatenate(out[:-1], axis=0), out_png)
    finally:
        for o in all_objs:
            if o.name in hidden:
                o.hide_render = hidden[o.name]
        for w in wires:
            if w:
                me = w.data
                bpy.data.objects.remove(w)
                bpy.data.meshes.remove(me)
        for o in objs:
            if o.name in saved_mats:
                for s, m in zip(o.material_slots, saved_mats[o.name]):
                    s.material = m
        for l in lights:
            bpy.data.objects.remove(l)
        bpy.data.objects.remove(cam)
        bpy.data.cameras.remove(cam_d)
    return out_png
