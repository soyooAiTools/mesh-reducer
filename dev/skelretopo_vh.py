"""dev/skelretopo_vh.py SRC.FBX OUTDIR ['character:1400+900,axe:430,crossbow:340'] [--size 2048] [--no-ao] [--no-renders] [--no-alpha]
(character:BODY+CLOTH triangles, or character:TOTAL: the cloth sheets then get 39% when the model has cloth)

Driver of core/skelretopo.py (skeleton driven retopology + bake).  Held props are split off
automatically (retopo.detect_parts; the job names other than 'character' are given to the detected
parts in ascending X of their centre, PART_BONES overrides).  Numbers are TRIANGLE budgets.
Writes per part <stem>_<part>_skel.FBX + <stem>_<part>_BC/_N/_AO.PNG, previews/ (left = result,
right = source), report.json."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
from core import io_util, materials, preview, metrics, retopo, auto, skelretopo

argv = [a for a in sys.argv[1:] if not a.startswith('--')]
src_path, outdir = argv[0], argv[1]
spec = argv[2] if len(argv) > 2 else 'character:1400+900,axe:430,crossbow:340'
jobs, cloth = [], {}
for p in spec.split(','):
    n, b = p.split(':')
    if '+' in b:                      # body + cloth
        cloth[n] = int(b.split('+')[1])
        b = b.split('+')[0]
    jobs.append((n, int(b)))
size = int(sys.argv[sys.argv.index('--size') + 1]) if '--size' in sys.argv else 2048
ao = '--no-ao' not in sys.argv
renders = '--no-renders' not in sys.argv
PART_BONES = json.loads(os.environ['PART_BONES']) if 'PART_BONES' in os.environ else 'AUTO'
stem = os.path.splitext(os.path.basename(src_path))[0]
os.makedirs(os.path.join(outdir, 'previews'), exist_ok=True)
T0 = time.time()
io_util.reset_scene()
objs = [o for o in io_util.import_file(src_path) if len(o.data.polygons)]
materials.link_textures(objs, os.path.dirname(os.path.abspath(src_path)))
raw = objs[0] if len(objs) == 1 else auto.merge_objects(objs, stem)
axes = io_util.fbx_axes(src_path)
arm = next((m.object for m in raw.modifiers if m.type == 'ARMATURE' and m.object), None)
report = {'source': os.path.basename(src_path), 'parts': {}}
if arm is None:
    report['error'] = 'no armature: skeleton driven retopology needs a skinned model'
    json.dump(report, open(os.path.join(outdir, 'report.json'), 'w'), ensure_ascii=False, indent=1)
    print('FAILED', report['error'])
    sys.exit(0)
src = retopo._dup(raw, 'welded')
retopo.weld(src)
comps = retopo.analyse(src, PART_BONES, labels=[p for p, _ in jobs if p != 'character'], log=lambda *a: None)
have = sorted({c.part for c in comps})
print('PARTS', have)
if len(jobs) == 1 and len(have) > 1:        # only the character was asked for: props get a default budget (never more than a third of what they have)
    ptris = {p: sum(len(src.data.polygons[f].vertices) - 2 for c in comps if c.part == p for f in c.faces) for p in have}
    jobs += [(p, max(24, min(300, ptris[p] // 3))) for p in have if p != 'character']
jobs = [j for j in jobs if j[0] in have]
part_bones = set().union(*([c.bones for c in comps if c.part != 'character'] or [set()]))
for o in objs + [raw, src]:
    o.hide_render = True
done = []
for part, target in jobs:
    print('PART', part, target, flush=True)
    try:
        low, high, rep = skelretopo.retopo_part(raw, src, comps, part, target, arm, os.path.join(outdir, '%s_%s' % (stem, part)),
                                                exclude_bones=part_bones, cloth_budget=cloth.get(part), size=size, ao=ao, alpha_sheets='--no-alpha' not in sys.argv)
    except ValueError as ex:
        report['parts'][part] = {'error': str(ex)}
        print('FAILED', part, ex, flush=True)
        continue
    low.name = low.data.name = '%s_%s' % (stem, part)
    rep['topology'] = retopo.topo_metrics(low.data)
    rep['skin'] = retopo.skin_metrics(low)
    m = metrics.evaluate(high.data, low.data)
    g = m['geometry']
    rep['geometry'] = {'hausdorff_pct': g['hausdorff_pct'], 'hausdorff_high_to_low_pct': g['hausdorff_orig_to_result_pct'],
                       'hausdorff_low_to_high_pct': g['hausdorff_result_to_orig_pct'], 'mean_pct': g['mean_dist_pct'], 'p99_pct': g['p99_dist_pct'],
                       'flipped_faces': m['normals']['flipped_faces']}
    rep['source_tris'] = sum(len(p.vertices) - 2 for p in high.data.polygons)
    fbx = os.path.join(outdir, '%s_%s_skel.FBX' % (stem, part))
    io_util.export_file(fbx, [low], copy_textures=True, axes=axes)
    rep['fbx'] = os.path.basename(fbx)
    report['parts'][part] = rep
    print('REPORT', part, json.dumps({k: rep[k] for k in ('topology', 'skin', 'geometry', 'seconds')}, ensure_ascii=False), flush=True)
    done.append((part, low, high, rep))
    low.hide_render = high.hide_render = True
report['seconds_total'] = round(time.time() - T0, 1)
json.dump(report, open(os.path.join(outdir, 'report.json'), 'w'), ensure_ascii=False, indent=1, default=float)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(outdir, 'previews', 'skel_debug.blend'))
if renders and done:
    import skelretopo_views
    skelretopo_views.render_all(done, os.path.join(outdir, 'previews'), stem, report, arm)
    json.dump(report, open(os.path.join(outdir, 'report.json'), 'w'), ensure_ascii=False, indent=1, default=float)
print('DONE', report['seconds_total'], 's (+renders %.0fs)' % (time.time() - T0 - report['seconds_total']))
