"""Link texture files to materials by name.

3ds Max multi/sub-object materials usually arrive in Blender with no (or the
wrong) images, while the texture files sit next to the FBX and are named after
the sub-materials:  <material>_BC.png (base colour), <material>_N.png (normal).
link_textures() wires them into the Principled BSDF of each material so that
the FBX exporter writes them out again (DiffuseColor / NormalMap / alpha)."""
import os
import re
import bpy
import numpy as np

IMG_EXT = ('.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tif', '.tiff', '.dds', '.psd')
# suffix after "<material>_"  ->  role
SUFFIX = {
    'bc': 'base', 'basecolor': 'base', 'base_color': 'base', 'albedo': 'base', 'diffuse': 'base',
    'd': 'base', 'col': 'base', 'color': 'base',
    'n': 'normal', 'nor': 'normal', 'nrm': 'normal', 'normal': 'normal',
}


def material_key(name):
    """material name without Blender's '.001' duplicate suffix, lower case"""
    return re.sub(r'\.\d{3}$', '', name).lower()


def match_textures(folder, material_names):
    """-> {material name: {'base': path, 'normal': path}}.

    A file belongs to the material with the LONGEST name that its file name
    starts with (so '范海辛' does not take '范海辛_武器_BC.png'), and only when
    what follows the name is a known suffix (_BC, _N, ...)."""
    keys = {}
    for n in material_names:
        keys.setdefault(material_key(n), []).append(n)
    out = {n: {} for n in material_names}
    try:
        files = sorted(os.listdir(folder))
    except OSError:
        return out
    for f in files:
        stem, ext = os.path.splitext(f)
        if ext.lower() not in IMG_EXT:
            continue
        low = stem.lower()
        owner = None
        for k in keys:
            if low.startswith(k) and len(low) > len(k) and low[len(k)] in '_-. ':
                if owner is None or len(k) > len(owner):
                    owner = k
        if owner is None:
            continue
        role = SUFFIX.get(low[len(owner) + 1:])
        if role is None:
            continue
        for n in keys[owner]:
            out[n].setdefault(role, os.path.join(folder, f))
    return out


def image_has_alpha(img, threshold=0.99, min_fraction=0.001):
    """True when a real part of the image is transparent (not just an alpha
    channel that is 1 everywhere, which is what most BC maps have)."""
    if img is None or img.channels < 4 or img.size[0] == 0:
        return False
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    a = px[3::4]
    return float((a < threshold).mean()) > min_fraction


def _principled(mat):
    if not mat.use_nodes:
        mat.use_nodes = True
    nt = mat.node_tree
    out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
    if out is not None and out.inputs['Surface'].is_linked:
        n = out.inputs['Surface'].links[0].from_node
        if n.type == 'BSDF_PRINCIPLED':
            return n
    b = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if b is None:
        b = nt.nodes.new('ShaderNodeBsdfPrincipled')
        if out is None:
            out = nt.nodes.new('ShaderNodeOutputMaterial')
        nt.links.new(b.outputs['BSDF'], out.inputs['Surface'])
    return b


def _unlink(nt, socket):
    for l in list(socket.links):
        nt.links.remove(l)


def _tex_node(nt, img, label, loc, reuse=None):
    node = reuse if reuse is not None and reuse.type == 'TEX_IMAGE' else nt.nodes.new('ShaderNodeTexImage')
    node.image = img
    node.label = label
    if reuse is None:
        node.location = loc
    return node


def link_material(mat, base=None, normal=None, alpha='AUTO'):
    """Wire the image files into the Principled BSDF of `mat`.
    alpha: 'AUTO' (only when the base colour image really is transparent),
    True or False.  Returns a small report dict."""
    rep = {'material': mat.name, 'base': None, 'normal': None, 'alpha': False}
    bsdf = _principled(mat)
    nt = mat.node_tree
    x, y = bsdf.location
    if base:
        img = bpy.data.images.load(base, check_existing=True)
        img.colorspace_settings.name = 'sRGB'
        sock = bsdf.inputs['Base Color']
        old = sock.links[0].from_node if sock.is_linked else None
        node = _tex_node(nt, img, 'Base Color', (x - 360, y + 120), reuse=old)
        _unlink(nt, sock)
        nt.links.new(node.outputs['Color'], sock)
        nt.nodes.active = node            # Workbench texture mode shows the active image node
        use_alpha = image_has_alpha(img) if alpha == 'AUTO' else bool(alpha)
        asock = bsdf.inputs['Alpha']
        _unlink(nt, asock)
        asock.default_value = 1.0
        if use_alpha:
            nt.links.new(node.outputs['Alpha'], asock)
            for attr, val in (('surface_render_method', 'DITHERED'), ('blend_method', 'CLIP')):
                try:
                    setattr(mat, attr, val)
                except Exception:
                    pass
            try:
                mat.alpha_threshold = 0.5
            except Exception:
                pass
            mat.use_backface_culling = False
        rep['base'] = os.path.basename(base)
        rep['alpha'] = use_alpha
    if normal:
        img = bpy.data.images.load(normal, check_existing=True)
        img.colorspace_settings.name = 'Non-Color'
        sock = bsdf.inputs['Normal']
        nm = sock.links[0].from_node if sock.is_linked else None
        if nm is None or nm.type != 'NORMAL_MAP':
            _unlink(nt, sock)
            nm = nt.nodes.new('ShaderNodeNormalMap')
            nm.location = (x - 200, y - 420)
            nt.links.new(nm.outputs['Normal'], sock)
        csock = nm.inputs['Color']
        old = csock.links[0].from_node if csock.is_linked else None
        node = _tex_node(nt, img, 'Normal', (nm.location[0] - 320, nm.location[1]), reuse=old)
        _unlink(nt, csock)
        nt.links.new(node.outputs['Color'], csock)
        rep['normal'] = os.path.basename(normal)
    return rep


def link_textures(objs, folder, alpha='AUTO'):
    """For every material of `objs`: find <material>_BC / <material>_N images
    in `folder` and link them.  Returns a list of reports (one per material)."""
    mats = []
    for o in objs:
        for s in o.material_slots:
            if s.material is not None and s.material not in mats:
                mats.append(s.material)
    found = match_textures(folder, [m.name for m in mats])
    reports = []
    for m in mats:
        f = found.get(m.name, {})
        if not f:
            reports.append({'material': m.name, 'base': None, 'normal': None, 'alpha': False})
            continue
        reports.append(link_material(m, f.get('base'), f.get('normal'), alpha=alpha))
    return reports


def base_image_path(mat):
    """file path of the image linked to Base Color (or '')"""
    if mat is None or not mat.use_nodes:
        return ''
    b = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if b is None or not b.inputs['Base Color'].is_linked:
        return ''
    n = b.inputs['Base Color'].links[0].from_node
    if n.type != 'TEX_IMAGE' or n.image is None:
        return ''
    return bpy.path.abspath(n.image.filepath)
