"""dev/retopo_vh.py SRC.FBX OUTDIR 'character:6000,axe:1000,crossbow:1000' [--size 2048] [--no-ao] [--no-renders]

Prototype driver of core/retopo.py ("拓扑烘焙": new low topology + baked maps).
Held props (weapons) are split off automatically (retopo.detect_parts); the job
names other than 'character' are given to the detected parts in their order
(ascending X of their centre).  PART_BONES='{"name": ["bone", ...]}' overrides
the detection.  Several mesh objects are merged into one first.  Writes per part  <stem>_<part>_retopo.FBX  +  <stem>_<part>_BC/_N/_AO.PNG,
previews/ (left = result, right = source) and report.json."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bpy
import numpy as np
from core import io_util, materials, preview, metrics, retopo, auto

argv = [a for a in sys.argv[1:] if not a.startswith('--')]
src_path, outdir = argv[0], argv[1]
jobs = [(p.split(':')[0], int(p.split(':')[1])) for p in argv[2].split(',')]
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
src = retopo._dup(raw, 'welded')
retopo.weld(src, manifold=bool(os.environ.get('RT_MANIFOLD')))
comps = retopo.analyse(src, PART_BONES, labels=[p for p, _ in jobs if p != 'character'])
have = {c.part for c in comps}
for p, _ in jobs:
    if p not in have:
        print('WARN no part "%s" in this model (parts: %s), job skipped' % (p, sorted(have)))
jobs = [j for j in jobs if j[0] in have]
open(os.path.join(outdir, 'components.txt'), 'w').write('\n'.join(retopo.describe(comps, src.data)))
raw.hide_render = src.hide_render = True
report = {'source': os.path.basename(src_path), 'parts': {}}
done = []
for part, target in jobs:
    print('PART', part, target, flush=True)
    low, high, rep = retopo.retopo_part(raw, src, comps, part, target, os.path.join(outdir, '%s_%s' % (stem, part)), size=size, ao=ao)
    low.name = low.data.name = '%s_%s' % (stem, part)
    rep['topology'] = retopo.topo_metrics(low.data)
    rep['skin'] = retopo.skin_metrics(low)
    m = metrics.evaluate(high.data, low.data)
    g = m['geometry']
    rep['geometry'] = {'hausdorff_pct': g['hausdorff_pct'], 'hausdorff_high_to_low_pct': g['hausdorff_orig_to_result_pct'],
                       'hausdorff_low_to_high_pct': g['hausdorff_result_to_orig_pct'], 'mean_pct': g['mean_dist_pct'], 'p99_pct': g['p99_dist_pct'],
                       'normal_dev_deg': m['normals']['mean_deviation_deg'], 'flipped_faces': m['normals']['flipped_faces']}
    rep['source_tris'] = len(high.data.polygons)
    fbx = os.path.join(outdir, '%s_%s_retopo.FBX' % (stem, part))
    io_util.export_file(fbx, [low], copy_textures=True, axes=axes)
    rep['fbx'] = os.path.basename(fbx)
    report['parts'][part] = rep
    print('REPORT', part, json.dumps({k: rep[k] for k in ('topology', 'skin', 'geometry', 'seconds')}, ensure_ascii=False), flush=True)
    done.append((part, low, high))
    low.hide_render = high.hide_render = True
report['seconds_total'] = round(time.time() - T0, 1)
json.dump(report, open(os.path.join(outdir, 'report.json'), 'w'), ensure_ascii=False, indent=1)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(outdir, 'previews', 'retopo_debug.blend'))
if renders:
    import retopo_views
    retopo_views.render_all(done, os.path.join(outdir, 'previews'), stem)
print('DONE', report['seconds_total'], 's (+renders %.0fs)' % (time.time() - T0 - report['seconds_total']))
