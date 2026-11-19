"""dev/validate_one.py SRC OUTDIR (--ratio R | --tris N) [--mode auto|character|rejoin|prop|retopo] [--res 700] [--size 1024]

One model, one target: reduce, export <OUTDIR>/result.fbx, measure, render
<OUTDIR>/wire.png and tex.png (per view: left = result, right = source) and
write <OUTDIR>/result.json.  Used by dev/validate_set.py.

  auto     core.auto.reduce_model: per object character / rejoin / prop (UVs and textures kept)
  retopo   core.retopo: merge objects, detect held props, new atlas + baked BC / N / AO per part
The ratio is that of the triangle count (quads count as two)."""
import sys, os, json, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bpy
import numpy as np
import bmesh
from mathutils import Matrix
from core import io_util, preview, metrics, retopo, auto, materials


def arg(name, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def joined(objs, name):
    """world space copy of all objs in one mesh, no modifiers (rest pose), for metrics and previews"""
    cs = []
    for o in objs:
        c = auto._dup(o, 'mr_j')
        for m in list(c.modifiers):
            c.modifiers.remove(m)
        cs.append(c)
    uv = next((c.data.uv_layers[0].name for c in cs if c.data.uv_layers), None)
    for c in cs:
        if c.data.uv_layers and uv:
            c.data.uv_layers[0].name = uv
    for o in bpy.context.scene.objects:
        o.select_set(False)
    for c in cs:
        c.hide_set(False)
        c.select_set(True)
    bpy.context.view_layer.objects.active = cs[0]
    if len(cs) > 1:
        bpy.ops.object.join()
    j = cs[0]
    bpy.context.view_layer.update()
    mw = j.matrix_world.copy()
    j.parent = None
    j.data.transform(mw)
    j.matrix_world = Matrix.Identity(4)
    bpy.context.view_layer.update()
    j.name = j.data.name = name
    return j


def grey(o):
    m = bpy.data.materials.new('v_grey')
    m.diffuse_color = (0.78, 0.78, 0.78, 1)
    c = auto._dup(o, o.name + '_grey')
    c.data.materials.clear()
    c.data.materials.append(m)
    for p in c.data.polygons:
        p.material_index = 0
    bpy.context.view_layer.update()
    return c


def checker(path, n=16, size=1024):
    """UV test texture for models whose textures are not available: coloured cells (hue by column,
    brightness by row) with dark lines; shows stretched, sheared or shifted UVs at a glance"""
    import colorsys
    a = np.zeros((size, size, 4), np.float32)
    c = size // n
    for y in range(n):
        for x in range(n):
            r, g, b = colorsys.hsv_to_rgb(x / n, 0.55 if (x + y) % 2 else 0.25, 0.55 + 0.4 * y / n)
            a[y * c:(y + 1) * c, x * c:(x + 1) * c, :3] = (r, g, b)
    for k in range(n):
        a[k * c:k * c + 2, :, :3] = 0.05
        a[:, k * c:k * c + 2, :3] = 0.05
    a[..., 3] = 1
    preview.save_png(a, path)
    return path


def image_proxies(g1, g0, R, O, tex, tmp, res=360):
    """numbers for what the eye sees first, from small renders of result and source with the
    same camera (4 views): silhouette_diff_pct = area where only one of them is / area of
    both (worst view): missing parts, thinned hair, lost volume;  shade_diff = mean
    brightness difference (0-255) of the grey shaded renders inside the common silhouette:
    faceting, broken normals;  tex_diff = the same for the textured (or UV checker)
    renders: sliding / smeared texture."""
    sil, shade, texd = [], [], []
    for v in ('front', 'left', 'top', 'front34'):
        def pair(a, b, **kw):
            preview.render_views([a, b], tmp, views=(v,), res=res, wire=False, ref=O, bg=(0.0, 1.0, 0.0), **kw)
            im = preview._read_png(tmp)
            w = (im.shape[1] - 6) // 2
            return im[:, :w, :3], im[:, w + 6:w + 6 + w, :3]
        a, b = pair(g1, g0, textured=False)
        ma = ~((a[..., 1] > 0.8) & (a[..., 0] < 0.3) & (a[..., 2] < 0.3))
        mb = ~((b[..., 1] > 0.8) & (b[..., 0] < 0.3) & (b[..., 2] < 0.3))
        both, any_ = ma & mb, ma | mb
        if any_.sum() == 0:
            continue
        sil.append(100.0 * float((ma ^ mb).sum()) / float(any_.sum()))
        if both.sum():
            shade.append(255.0 * float(np.abs(a.mean(2) - b.mean(2))[both].mean()))
            ta, tb = pair(R, O, textured=True, texture=tex)
            texd.append(255.0 * float(np.abs(ta - tb).mean(2)[both].mean()))
    return {'silhouette_diff_pct': round(max(sil), 2) if sil else None, 'silhouette_diff_mean_pct': round(float(np.mean(sil)), 2) if sil else None,
            'shade_diff': round(float(np.mean(shade)), 2) if shade else None, 'tex_diff': round(float(np.mean(texd)), 2) if texd else None}


def skin_of(objs, src_objs):
    n = sk = 0
    mx = 0
    for o in objs:
        for v in o.data.vertices:
            k = sum(1 for g in v.groups if g.weight > 0)
            n += 1
            sk += k > 0
            mx = max(mx, k)
    ns = sum(len(o.data.vertices) for o in src_objs)
    ss = sum(1 for o in src_objs for v in o.data.vertices if any(g.weight > 0 for g in v.groups))
    # vertices without weights in results whose source object is completely skinned
    lost = 0
    if len(objs) == len(src_objs):
        for o, so in zip(objs, src_objs):
            if len(so.data.vertices) and all(any(g.weight > 0 for g in v.groups) for v in so.data.vertices):
                lost += sum(1 for v in o.data.vertices if not any(g.weight > 0 for g in v.groups))
    return {'unskinned_new_vertices': lost,'skinned_vertex_pct': round(100.0 * sk / max(n, 1), 2), 'max_influences': mx,
            'source_skinned_vertex_pct': round(100.0 * ss / max(ns, 1), 2),
            'source_max_influences': max((sum(1 for g in v.groups if g.weight > 0) for o in src_objs for v in o.data.vertices), default=0)}


def main():
    src_path, outdir = [a for a in sys.argv[1:] if not a.startswith('--')][:2]
    ratio = arg('--ratio', None, float)
    tris = arg('--tris', None, int)
    mode = arg('--mode', 'auto')
    res = arg('--res', 700, int)
    size = arg('--size', 1024, int)
    os.makedirs(outdir, exist_ok=True)
    T0 = time.time()
    out = {'source': src_path, 'mode_asked': mode, 'ok': False}
    io_util.reset_scene()
    objs = [o for o in io_util.import_file(src_path) if len(o.data.polygons)]
    if not objs:
        raise RuntimeError('no mesh in file')
    te = sum(p.loop_total - 2 for o in objs for p in o.data.polygons)
    if te < arg('--min-source-tris', 0, int):
        json.dump({'ok': True, 'skipped': True, 'source_tris': te}, open(os.path.join(outdir, 'result.json'), 'w'))
        return
    out['source_objects'] = len(objs)
    out['source_faces'] = sum(len(o.data.polygons) for o in objs)
    out['source_tris'] = te
    wco = [np.array([(o.matrix_world @ v.co)[:] for v in o.data.vertices]) for o in objs]
    lo = np.min([c.min(0) for c in wco], 0)
    hi = np.max([c.max(0) for c in wco], 0)
    out['size'] = [round(float(x), 4) for x in hi - lo]
    out['bones'] = sum(len(a.data.bones) for a in bpy.data.objects if a.type == 'ARMATURE')
    skinned = any(m.type == 'ARMATURE' for o in objs for m in o.modifiers)
    axes = io_util.fbx_axes(src_path) if skinned and src_path.lower().endswith('.fbx') else None
    log = lambda t: print('LOG', t, flush=True)
    if mode == 'retopo':
        lods = auto.lod_duplicates(objs)
        objs = [o for o in objs if o not in lods]
        te = sum(p.loop_total - 2 for o in objs for p in o.data.polygons)
        if ratio is None:
            ratio = min(1.0, tris / te)
        raw = objs[0] if len(objs) == 1 else auto.merge_objects(objs, 'model')
        pr = auto.probe([raw])
        src = retopo._dup(raw, 'welded')
        retopo.weld(src, manifold='--plain-weld' not in sys.argv)
        comps = retopo.analyse(src, 'AUTO', log=log)
        raw.hide_render = src.hide_render = True
        parts = sorted({c.part for c in comps}, key=lambda p: (p != 'character', p))
        out['probe'] = pr
        out['parts'] = {}
        new, orig = [], []
        for part in parts:
            ptris = sum(len(c.faces) for c in comps if c.part == part)       # welded triangles / faces of the part
            pte = sum(src.data.polygons[f].loop_total - 2 for c in comps if c.part == part for f in c.faces)
            target = max(16, int(round(ratio * pte * pr['objects'][0]['faces_rejoined'] / max(pr['tri_equiv'], 1))))
            low, high, rep = retopo.retopo_part(raw, src, comps, part, target, os.path.join(outdir, part), size=size, log=log)
            low.name = low.data.name = part
            out['parts'][part] = {'target_faces': target, 'kinds': rep['build']['kinds'], 'seconds': rep['seconds'], 'cage': rep['cage'],
                                  'units_stuck': sum(1 for u in rep['build']['units'] if u['stuck']), 'units': len(rep['build']['units'])}
            new.append(low)
            orig.append(high)
            low.hide_render = high.hide_render = True
        out['modes'] = {'retopo': te}
        infos = []
    else:
        if ratio is None:
            ratio = min(1.0, tris / te)
        res_, pr = auto.reduce_model(objs, ratio, mode=mode, log=log, **({'finish_edges': True} if '--finish-edges' in sys.argv else {}))
        out['probe'] = pr
        new = [r[0] for r in res_]
        orig = [r[1] for r in res_]
        infos = [r[2] for r in res_]
        modes = {}
        for n_, o_, i_ in res_:
            modes[i_['mode']] = modes.get(i_['mode'], 0) + sum(p.loop_total - 2 for p in o_.data.polygons)
        out['modes'] = modes
        for n_, o_ in zip(new, orig):
            nm = o_.name
            o_.name = nm + '__orig'
            n_.name = nm
    out['ratio'] = ratio
    out['objects'] = infos
    out['seconds_reduce'] = round(time.time() - T0, 1)
    io_util.export_file(os.path.join(outdir, 'result.fbx'), new, copy_textures=(mode == 'retopo'), axes=axes)
    # ---- metrics on the whole model
    R = joined(new, 'result_all')
    O = joined(orig, 'source_all')
    m = metrics.evaluate(O.data, R.data, n_samples=30000)
    row = metrics.summary_row(m)
    topo = retopo.topo_metrics(R.data)
    topo0 = retopo.topo_metrics(O.data)
    out['metrics'] = {
        'faces': row['faces'], 'quads': row['quads'], 'tris': row['tris'], 'ngons': row['ngons'], 'quad_pct': row['quad_pct'],
        'tri_equiv': row['unity_tris'], 'tri_ratio': round(row['unity_tris'] / max(1, sum(p.loop_total - 2 for p in O.data.polygons)), 4),
        'nonmanifold_edges': topo['nonmanifold_edges'], 'nonmanifold_edges_source': topo0['nonmanifold_edges'],
        'boundary_edges': topo['boundary_edges'], 'boundary_edges_source': topo0['boundary_edges'],
        'flipped': row['flipped'], 'slivers': topo['slivers'], 'slivers_source': topo0['slivers'],
        'sliver_pct': round(100.0 * topo['slivers'] / max(1, topo['faces']), 2), 'sliver_pct_source': round(100.0 * topo0['slivers'] / max(1, topo0['faces']), 2),
        'hausdorff_pct': row['hausdorff_pct'], 'mean_pct': row['mean_pct'], 'p99_pct': m['geometry']['p99_dist_pct'],
        'normal_dev_deg': row['normal_dev_deg'], 'uv_islands': row['uv_islands'], 'uv_flipped': row['uv_flipped'],
        'quad_rings': row['quad_rings'], 'source_quad_pct': m['original']['quad_pct'],
    }
    # UV islands of the source are counted on a welded copy: files split along smoothing groups
    # have several vertices per UV point, which the island count would take for separate islands
    Ow = auto._dup(O, 'source_welded')
    retopo.weld(Ow)
    bm = bmesh.new()
    bm.from_mesh(Ow.data)
    if bm.loops.layers.uv:
        isl = metrics.uv_islands(bm, bm.loops.layers.uv[0])[0]
        out['metrics']['uv_islands'] = '%s/%s' % (row['uv_islands'].split('/')[0], isl)
        out['metrics']['uv_islands_source_unwelded'] = row['uv_islands'].split('/')[1]
    out['metrics']['nonmanifold_edges_source_welded'] = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    bm.free()
    bpy.data.objects.remove(Ow)
    out['metrics'].update(skin_of(new, orig))
    out['stuck'] = any(i.get('stuck') for i in infos)
    out['ok'] = True
    out['seconds'] = round(time.time() - T0, 1)
    json.dump(out, open(os.path.join(outdir, 'result.json'), 'w'), ensure_ascii=False, indent=1)
    # ---- previews: left result, right source
    for o in bpy.context.scene.objects:
        if o.type == 'MESH':
            o.hide_render = True
    try:
        g1, g0 = grey(R), grey(O)
        ext = np.array(out['size'])
        views = ('front', 'left') if ext[2] >= 0.6 * max(ext[0], ext[1]) else ('front34', 'top')
        preview.render_views([g1, g0], os.path.join(outdir, 'wire.png'), views=views, res=res, wire=True, textured=False, ref=O)
        tex = None
        if mode != 'retopo' and not preview._has_usable_texture([O]):
            tex = checker(os.path.join(outdir, 'uv_checker.png'))
            out['preview_texture'] = 'uv checker (source textures not available)'
        preview.render_views([R, O], os.path.join(outdir, 'tex.png'), views=views, res=res, wire=False, textured=True, ref=O, texture=tex)
        out['views'] = list(views)
        out['metrics'].update(image_proxies(g1, g0, R, O, tex, os.path.join(outdir, '_tmp.png')))
        os.remove(os.path.join(outdir, '_tmp.png'))
    except Exception as ex:
        traceback.print_exc()
        out['render_error'] = str(ex)[-300:]
    out['seconds'] = round(time.time() - T0, 1)
    json.dump(out, open(os.path.join(outdir, 'result.json'), 'w'), ensure_ascii=False, indent=1)
    print('RESULT', json.dumps(out['metrics'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        traceback.print_exc()
        outdir = [a for a in sys.argv[1:] if not a.startswith('--')][1]
        os.makedirs(outdir, exist_ok=True)
        json.dump({'ok': False, 'error': '%s: %s' % (type(ex).__name__, str(ex)[-400:])}, open(os.path.join(outdir, 'result.json'), 'w'), ensure_ascii=False, indent=1)
        sys.exit(1)
