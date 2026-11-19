"""Import / export helpers (FBX / OBJ / GLB)."""
import bpy, os


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def fbx_null_skin_bones(path):
    """uuids of FBX Model nodes of type Null (not LimbNode) that must be read
    as bones: 3ds Max exports Biped / Bone objects as plain Null nodes, the
    skin Clusters link to them, and Blender's importer only reads clusters of
    LimbNode models -> the mesh would arrive without armature and without
    vertex groups.  Returned are the cluster linked Null nodes, every Null
    between them and their top-most Null ancestor below the scene's first
    level, and all Null descendants (nubs, footsteps), so the whole skeleton
    tree keeps its names and paths.  Empty set when the file needs no help
    (no skin, proper LimbNodes, ASCII FBX, ...)."""
    try:
        import addon_utils
        addon_utils.enable('io_scene_fbx', default_set=False)
        from io_scene_fbx import parse_fbx
        root, _ver = parse_fbx.parse(path)
    except Exception:
        return set()
    objs = next((e for e in root.elems if e.id == b'Objects'), None)
    conns = next((e for e in root.elems if e.id == b'Connections'), None)
    if objs is None or conns is None:
        return set()
    models = {}
    clusters = set()
    for e in objs.elems:
        if e.id == b'Model':
            models[e.props[0]] = e.props[2]
        elif e.id == b'Deformer' and e.props[2] == b'Cluster':
            clusters.add(e.props[0])
    if not clusters:
        return set()
    parent = {}
    linked = set()
    for c in conns.elems:
        if c.props[0] != b'OO':
            continue
        src, dst = c.props[1], c.props[2]
        if src in models and dst in clusters:
            linked.add(src)
        elif src in models and dst in models:
            parent[src] = dst
    null_linked = {u for u in linked if models[u] in (b'Null', b'Root')}
    if not null_linked:
        return set()
    kids = {}
    for c, p in parent.items():
        kids.setdefault(p, []).append(c)
    bones = set()
    tops = set()
    for u in null_linked:
        chain = [u]
        while chain[-1] in parent and models[parent[chain[-1]]] in (b'Null', b'Root'):
            chain.append(parent[chain[-1]])
        # the top-most Null stays an object: the importer turns it into the
        # armature object, so "Top/Bip001/..." paths survive an export.
        # (a skeleton root sitting directly at scene level stays a bone and
        # the importer inserts an 'Armature' object above it)
        if len(chain) > 1 and chain[-1] not in linked:
            chain.pop()
        bones.update(chain)
        tops.add(chain[-1])
    stack = list(tops)
    while stack:
        u = stack.pop()
        for k in kids.get(u, ()):
            if models[k] in (b'Null', b'Root', b'LimbNode', b'Limb'):
                bones.add(k)
                stack.append(k)
    return {u for u in bones if models[u] not in (b'LimbNode', b'Limb')}


class _null_bones_as_bones:
    """Context manager around bpy.ops.import_scene.fbx, see fbx_null_skin_bones."""

    def __init__(self, path):
        self.uuids = fbx_null_skin_bones(path)
        self.mod = None

    def __enter__(self):
        try:
            from io_scene_fbx import import_fbx
        except Exception:
            return self
        uuids = self.uuids
        base = import_fbx.FbxImportHelperNode

        class Node(base):
            __slots__ = ()

            def __init__(self, fbx_elem, bl_data, fbx_transform_data, is_bone):
                if fbx_elem is not None and not is_bone and fbx_elem.id == b'Model' and fbx_elem.props[0] in uuids:
                    is_bone = True
                super().__init__(fbx_elem, bl_data, fbx_transform_data, is_bone)

            def find_fake_bones(self, in_armature=False):
                # Importer bug (KeyError in link_hierarchy, "mesh.armature_setup[self]"): a Null /
                # Root node between bones (Max: "root" LimbNode -> "Bip001" Root -> bones) is turned
                # into a bone only here, AFTER the bones were told which armature they belong
                # to.  It and every bone below it keep armature None, their clusters register
                # the mesh under None and the armature cannot find its mesh.  Tell them again.
                super().find_fake_bones(in_armature)
                if self.is_armature:
                    self.find_armature_bones(self)
        Node.__name__ = base.__name__
        self.mod, self.base = import_fbx, base
        import_fbx.FbxImportHelperNode = Node
        return self

    def __exit__(self, *exc):
        if self.mod is not None:
            self.mod.FbxImportHelperNode = self.base
        return False


_BIN_CACHE = {}


def binary_fbx(path):
    """path of a binary FBX with the content of `path`: the file itself, or a
    converted temporary copy when it is an ASCII FBX (see core.fbx_ascii)"""
    from . import fbx_ascii
    key = (os.path.abspath(path), os.path.getmtime(path))
    if key in _BIN_CACHE and os.path.isfile(_BIN_CACHE[key]):
        return _BIN_CACHE[key]
    out = path
    if fbx_ascii.is_ascii(path):
        import tempfile
        out = os.path.join(tempfile.mkdtemp(prefix='mr_fbx_'), os.path.basename(path))
        fbx_ascii.to_binary(path, out)
    _BIN_CACHE[key] = out
    return out


def _relink_images(folder, before):
    """images the importer could not find (the converted temporary file lives elsewhere): look next to the real file"""
    for img in bpy.data.images:
        if img in before or img.packed_file is not None or not img.filepath:
            continue
        if os.path.isfile(bpy.path.abspath(img.filepath)):
            continue
        cand = os.path.join(folder, os.path.basename(img.filepath.replace('\\', '/')))
        if os.path.isfile(cand):
            img.filepath = cand


def import_file(path):
    """Import a model file, return list of new mesh objects."""
    before = set(bpy.data.objects)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.fbx':
        real = binary_fbx(path)
        imgs = set(bpy.data.images)
        with _null_bones_as_bones(real):
            bpy.ops.import_scene.fbx(filepath=real)
        if real != path:
            _relink_images(os.path.dirname(os.path.abspath(path)), imgs)
    elif ext == '.obj':
        bpy.ops.wm.obj_import(filepath=path)
    elif ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        raise ValueError('unsupported file type: ' + ext)
    new = [o for o in bpy.data.objects if o not in before]
    # work in the bind pose: a file exported with an animation arrives posed at its first
    # frame, and everything that reads evaluated meshes (data transfer of weights / normals,
    # Decimate, bakes, previews) would mix the posed source with the unposed result
    for o in new:
        if o.type == 'ARMATURE':
            o.data.pose_position = 'REST'
    return [o for o in new if o.type == 'MESH']


def fbx_axes(path):
    """(axis_forward, axis_up) in Blender exporter terms of a binary FBX file, or None"""
    try:
        import addon_utils
        addon_utils.enable('io_scene_fbx', default_set=False)
        from io_scene_fbx import parse_fbx
        root, _ver = parse_fbx.parse(binary_fbx(path))
        gs = next(e for e in root.elems if e.id == b'GlobalSettings')
        pr = {p.props[0]: p.props[4] for p in next(e for e in gs.elems if e.id == b'Properties70').elems if len(p.props) > 4}
        up = ('-' if pr[b'UpAxisSign'] < 0 else '') + 'XYZ'[pr[b'UpAxis']]
        # FBX "front" points at the viewer, Blender's axis_forward away from it
        fw = ('-' if pr[b'FrontAxisSign'] > 0 else '') + 'XYZ'[pr[b'FrontAxis']]
        return fw, up
    except Exception:
        return None


def _copy_images_next_to(path, objs):
    """copy every image file used by the materials of objs into the folder of
    `path` and point the image datablocks at the copies (so the exporter can
    write plain relative file names)."""
    import shutil
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    done = set()
    for o in objs:
        for s in o.material_slots:
            m = s.material
            if m is None or not m.use_nodes or m.node_tree is None:
                continue
            for n in m.node_tree.nodes:
                img = getattr(n, 'image', None) if n.type == 'TEX_IMAGE' else None
                if img is None or img.name in done or img.packed_file is not None:
                    continue
                done.add(img.name)
                src = bpy.path.abspath(img.filepath)
                if not src or not os.path.isfile(src):
                    continue
                dst = os.path.join(folder, os.path.basename(src))
                if not (os.path.exists(dst) and os.path.samefile(src, dst)):
                    shutil.copy2(src, dst)
                img.filepath = dst


def export_file(path, objs, copy_textures=False, axes=None):
    """copy_textures: FBX only, copy the material images next to the file and
    reference them by relative path (what Unity / Max look for).
    axes: (axis_forward, axis_up) for FBX, default Blender's (-Z, Y).  Pass
    fbx_axes(source) for skinned meshes so the skeleton keeps the world
    orientation of the source file."""
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
        if copy_textures:
            _copy_images_next_to(path, objs)
        kw = {}
        if axes:
            kw.update(axis_forward=axes[0], axis_up=axes[1])
        bpy.ops.export_scene.fbx(filepath=path, use_selection=True,
                                 mesh_smooth_type='FACE', use_mesh_modifiers=False,
                                 add_leaf_bones=False, bake_anim=False,
                                 use_tspace=False, path_mode='RELATIVE' if copy_textures else 'AUTO',
                                 embed_textures=False, **kw)
    elif ext == '.obj':
        bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True,
                              export_triangulated_mesh=False)
    elif ext in ('.glb', '.gltf'):
        bpy.ops.export_scene.gltf(filepath=path, use_selection=True)
    else:
        raise ValueError('unsupported file type: ' + ext)
    return path
