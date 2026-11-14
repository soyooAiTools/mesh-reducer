# -*- coding: utf-8 -*-
"""Headless entry point of MeshReducer.

Runs inside Blender:
    blender -b --factory-startup --python runner.py -- --in X.fbx --out Y.fbx --faces 1500 --mode character
or with the bpy python module:
    python3 runner.py --in X.fbx --out Y.fbx --ratio 0.4 --mode character

stdout protocol (one item per line, UTF-8):
    PROGRESS <0..1> <message>
    LOG <text>
    RESULT <json>          (last line on success)
    ERROR <message>        (on failure, exit code 1)
"""
import sys
import os
import json
import time
import argparse
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def emit(kind, text):
    sys.stdout.write('%s %s\n' % (kind, text))
    sys.stdout.flush()


def progress(frac, msg=''):
    emit('PROGRESS', '%.3f %s' % (max(0.0, min(1.0, frac)), msg))


def parse_args(argv):
    ap = argparse.ArgumentParser(prog='runner.py', description='MeshReducer headless runner')
    ap.add_argument('--in', dest='inp', help='input model (.fbx/.obj/.glb/.gltf)')
    ap.add_argument('--out', dest='out', help='output model (format from the extension)')
    ap.add_argument('--faces', type=int, default=0, help='target polygon count (all meshes of the file together)')
    ap.add_argument('--ratio', type=float, default=0.0, help='target ratio 0..1 (used when --faces is not given)')
    ap.add_argument('--mode', default='character', choices=['character', 'prop', 'auto'])
    ap.add_argument('--symmetry', default='auto', help='auto | x | y | z | none')
    ap.add_argument('--protect-group', default='', help='vertex group whose weights protect loops')
    ap.add_argument('--no-seams', action='store_true', help='do not treat UV seams as constraints')
    ap.add_argument('--no-sharp', action='store_true', help='do not treat hard edges as constraints')
    ap.add_argument('--no-joints', action='store_true', help='no automatic joint loop protection')
    ap.add_argument('--joint-strength', type=float, default=5.0)
    ap.add_argument('--prop-method', default='strict', choices=['strict', 'fast', 'auto'],
                    help='auto: strict, but fast (Blender decimate) below 25%% of the faces')
    ap.add_argument('--link-textures', action='store_true',
                    help='link <material>_BC / <material>_N images next to the input to the materials and '
                         'copy them next to the output (FBX, relative paths)')
    ap.add_argument('--material-targets', default='',
                    help='prop mode, JSON {material slot index: faces}: own face budget per material')
    ap.add_argument('--report', default='', help='write the result JSON here')
    ap.add_argument('--renders', default='', help='folder for before/after preview PNGs')
    ap.add_argument('--texture', default='', help='texture used for previews and the texture error metric')
    ap.add_argument('--regions', default='', help='JSON with joint regions for the loop check')
    ap.add_argument('--no-eval', action='store_true', help='skip the metric evaluation')
    ap.add_argument('--engine', default='auto', help='preview engine: auto | workbench | cycles')
    ap.add_argument('--render-only', action='store_true', help='only render previews: --orig A --result B --renders DIR')
    ap.add_argument('--orig', default='')
    ap.add_argument('--result', default='')
    ap.add_argument('--res', type=int, default=640)
    return ap.parse_args(argv)


IMG_EXT = ('.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tif', '.tiff')
TEX_HINTS = ('diffuse', 'basecolor', 'base_color', 'albedo', '_bc', '_d.', '_col', 'color')


def find_texture(model_path, objs):
    """best guess of the base colour texture: material image that exists on
    disk, else an image file next to the model"""
    import bpy
    for o in objs:
        for slot in o.material_slots:
            mat = slot.material
            if not mat or not mat.use_nodes:
                continue
            for n in mat.node_tree.nodes:
                if n.type == 'TEX_IMAGE' and n.image:
                    p = bpy.path.abspath(n.image.filepath)
                    if p and os.path.isfile(p):
                        return p
    folder = os.path.dirname(os.path.abspath(model_path))
    cands = []
    stem = os.path.splitext(os.path.basename(model_path))[0].lower()
    try:
        names = []
        for root, _dirs, files in os.walk(folder):
            if root[len(folder):].count(os.sep) > 1:
                continue
            names += [os.path.join(root, f) for f in files]
    except OSError:
        names = []
    for p in names:
        f = os.path.basename(p)
        low = f.lower()
        if not low.endswith(IMG_EXT):
            continue
        score = 0
        if any(h in low for h in TEX_HINTS):
            score += 1
        tokens = [t for t in stem.replace('-', '_').split('_') if len(t) >= 3 and t not in ('high', 'low', 'mesh')]
        if any(t in low for t in tokens):
            score += 2
        if any(h in low for h in ('_n.', 'normal', 'mask', 'rough', 'metal', '_ao', 'spec', 'preview', 'compare')):
            score -= 3
        cands.append((score, p))
    cands.sort(reverse=True)
    # needs a name match with the model (or be the only plausible image)
    if cands and (cands[0][0] >= 2 or (cands[0][0] > 0 and len([c for c in cands if c[0] > 0]) == 1)):
        return cands[0][1]
    return ''


def do_renders(pairs, outdir, texture, engine, res, stem, use_materials=False):
    """pairs: [(result_obj, orig_obj)] -> list of written png paths"""
    from core import preview
    os.makedirs(outdir, exist_ok=True)
    engines = ['WORKBENCH', 'CYCLES'] if engine in ('auto', '') else [engine.upper()]
    out = []
    # the biggest object pair is shown
    r, o = max(pairs, key=lambda p: len(p[1].data.polygons))
    sets = [('front_back', ('front', 'back')), ('side_34', ('left', 'front34'))]
    last_err = None
    for eng in engines:
        try:
            out = []
            for name, views in sets:
                p = os.path.join(outdir, '%s_%s.png' % (stem, name))
                preview.render_views([r, o], p, views=views, res=res, texture=texture or None, engine=eng,
                                     textured=bool(texture) or use_materials)
                if not os.path.isfile(p) or os.path.getsize(p) < 1000:
                    raise RuntimeError('render produced no image')
                out.append(p)
            return out, eng, None
        except Exception as ex:          # try the next engine
            last_err = '%s: %s' % (eng, ex)
            emit('LOG', 'preview render failed with ' + last_err)
    return [], None, last_err


def main(argv=None):
    if argv is None:
        argv = sys.argv
        argv = argv[argv.index('--') + 1:] if '--' in argv else argv[1:]
    args = parse_args(argv)
    t0 = time.time()
    try:
        import bpy
        from core import io_util, api, prop, metrics
    except Exception as ex:
        emit('ERROR', 'cannot import Blender python API: %s' % ex)
        return 1
    try:
        if args.render_only:
            io_util.reset_scene()
            oo = io_util.import_file(args.orig)
            rr = io_util.import_file(args.result)
            if not oo or not rr:
                raise RuntimeError('no mesh in input')
            oo.sort(key=lambda x: -len(x.data.polygons))
            rr.sort(key=lambda x: -len(x.data.polygons))
            tex = args.texture or find_texture(args.orig, oo)
            stem = os.path.splitext(os.path.basename(args.result))[0]
            imgs, eng, err = do_renders([(rr[0], oo[0])], args.renders, tex, args.engine.lower(), args.res, stem)
            res = {'ok': bool(imgs), 'renders': imgs, 'engine': eng, 'error': err}
            if args.report:
                with open(args.report, 'w', encoding='utf-8') as f:
                    json.dump(res, f, ensure_ascii=False, indent=1)
            emit('RESULT', json.dumps(res, ensure_ascii=False))
            return 0 if imgs else 2

        if not args.inp or not args.out:
            raise RuntimeError('--in and --out are required')
        if not os.path.isfile(args.inp):
            raise RuntimeError('input file not found: %s' % args.inp)
        progress(0.01, 'loading')
        io_util.reset_scene()
        objs = io_util.import_file(args.inp)
        objs = [o for o in objs if len(o.data.polygons) > 0]
        if not objs:
            raise RuntimeError('the file contains no mesh')
        if args.link_textures:
            from core import materials
            for rep_ in materials.link_textures(objs, os.path.dirname(os.path.abspath(args.inp))):
                emit('LOG', 'textures: %s' % json.dumps(rep_, ensure_ascii=False))
        skinned = any(m.type == 'ARMATURE' for o in objs for m in o.modifiers)
        mat_targets = json.loads(args.material_targets) if args.material_targets else None
        total = sum(len(o.data.polygons) for o in objs)
        if args.faces > 0:
            ratio = min(1.0, args.faces / total)
        elif args.ratio > 0:
            ratio = min(1.0, args.ratio)
        else:
            ratio = 0.5
        sym = args.symmetry.upper()
        tex = args.texture or find_texture(args.inp, objs)
        if tex:
            emit('LOG', 'texture: %s' % tex)
        results = []
        pairs = []
        done = 0
        for o in objs:
            n = len(o.data.polygons)
            share = n / total
            base = 0.05 + 0.75 * done / total

            def cb(f, msg='', base=base, share=share):
                progress(base + 0.75 * share * f, '%s: %s' % (o.name, msg))
            tgt = max(4, int(round(n * ratio))) if len(objs) > 1 else (args.faces if args.faces > 0 else max(4, int(round(n * ratio))))
            mode = args.mode
            if mode == 'auto':
                quads = sum(1 for p in o.data.polygons if p.loop_total == 4)
                mode = 'character' if quads > 0.3 * n else 'prop'
            if tgt >= n:
                emit('LOG', '%s: target %d >= %d faces, copied unchanged' % (o.name, tgt, n))
                new = o.copy()
                new.data = o.data.copy()
                for c in (list(o.users_collection) or [bpy.context.scene.collection]):
                    c.objects.link(new)
                info = {'mode': mode, 'faces_before': n, 'faces_after': n, 'target': tgt, 'unchanged': True}
            elif mode == 'character':
                new, info = api.reduce_character(
                    o, target_faces=tgt, symmetry=sym, protect_group=args.protect_group or None,
                    preserve_seams=not args.no_seams, preserve_sharp=not args.no_sharp,
                    auto_joints=not args.no_joints, joint_strength=args.joint_strength, progress=cb)
            else:
                new, info = prop.reduce_prop(
                    o, target_faces=tgt, symmetry=sym, preserve_seams=not args.no_seams,
                    preserve_sharp=not args.no_sharp, method=args.prop_method.upper(), progress=cb,
                    material_targets=mat_targets if len(objs) == 1 else None)
            info['object'] = o.name
            new_name = o.name
            o.name = o.name + '__orig'
            new.name = new_name
            results.append(info)
            pairs.append((new, o))
            done += n
        progress(0.82, 'exporting')
        # skinned meshes keep the axes of the source file, so the skeleton's rest matrices stay as they were
        axes = io_util.fbx_axes(args.inp) if skinned and args.inp.lower().endswith('.fbx') else None
        io_util.export_file(args.out, [p[0] for p in pairs], copy_textures=args.link_textures, axes=axes)
        res = {'ok': True, 'input': os.path.abspath(args.inp), 'output': os.path.abspath(args.out),
               'faces_before': total, 'faces_after': sum(r['faces_after'] for r in results),
               'objects': results, 'renders': [], 'blender': bpy.app.version_string}
        res['texture'] = tex
        if not args.no_eval:
            progress(0.86, 'measuring result')
            regions = None
            if args.regions and os.path.isfile(args.regions):
                with open(args.regions, 'r', encoding='utf-8') as f:
                    regions = json.load(f)
            ms = []
            for new, o in pairs:
                mtex = tex or None
                if args.link_textures:
                    from core import materials
                    mtex = {i: materials.base_image_path(sl.material) for i, sl in enumerate(o.material_slots)}
                m = metrics.evaluate(o.data, new.data, texture=mtex, regions=regions)
                m['object'] = new.name
                ms.append(m)
            res['metrics'] = ms
            big = max(range(len(pairs)), key=lambda i: len(pairs[i][1].data.polygons))
            res['summary'] = metrics.summary_row(ms[big])
        res['seconds'] = round(time.time() - t0, 2)
        if args.report:
            os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
            with open(args.report, 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
        if args.renders:
            progress(0.93, 'rendering previews')
            stem = os.path.splitext(os.path.basename(args.out))[0]
            imgs, eng, err = do_renders(pairs, args.renders, '' if args.link_textures else tex,
                                        args.engine.lower(), args.res, stem, use_materials=args.link_textures)
            res['renders'] = imgs
            res['render_engine'] = eng
            if err and not imgs:
                res['render_error'] = err
            res['seconds'] = round(time.time() - t0, 2)
            if args.report:
                with open(args.report, 'w', encoding='utf-8') as f:
                    json.dump(res, f, ensure_ascii=False, indent=1)
        progress(1.0, 'done')
        emit('RESULT', json.dumps(res, ensure_ascii=False))
        return 0
    except Exception as ex:
        traceback.print_exc()
        emit('ERROR', '%s: %s' % (type(ex).__name__, ex))
        return 1


if __name__ == '__main__':
    code = main()
    sys.stdout.flush()
    # under "blender --python" sys.exit sets Blender's exit code
    sys.exit(code)
