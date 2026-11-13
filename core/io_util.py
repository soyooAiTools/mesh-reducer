"""Import / export helpers (FBX / OBJ / GLB)."""
import bpy, os


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_file(path):
    """Import a model file, return list of new mesh objects."""
    before = set(bpy.data.objects)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == '.obj':
        bpy.ops.wm.obj_import(filepath=path)
    elif ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        raise ValueError('unsupported file type: ' + ext)
    return [o for o in bpy.data.objects if o not in before and o.type == 'MESH']


def export_file(path, objs):
    ext = os.path.splitext(path)[1].lower()
    for o in bpy.context.scene.objects:
        o.select_set(False)
    sel = set()
    for o in objs:
        o.hide_set(False) if o.name in bpy.context.view_layer.objects else None
        o.select_set(True)
        sel.add(o)
        p = o.parent
        while p is not None:          # keep armatures / parents
            p.select_set(True)
            p = p.parent
        for m in o.modifiers:
            if m.type == 'ARMATURE' and m.object:
                m.object.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if ext == '.fbx':
        bpy.ops.export_scene.fbx(filepath=path, use_selection=True,
                                 mesh_smooth_type='FACE', use_mesh_modifiers=False,
                                 add_leaf_bones=False, bake_anim=False,
                                 use_tspace=False, path_mode='AUTO')
    elif ext == '.obj':
        bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True,
                              export_triangulated_mesh=False)
    elif ext in ('.glb', '.gltf'):
        bpy.ops.export_scene.gltf(filepath=path, use_selection=True)
    else:
        raise ValueError('unsupported file type: ' + ext)
    return path
