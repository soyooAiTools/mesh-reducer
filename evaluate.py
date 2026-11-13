# -*- coding: utf-8 -*-
"""Compare a reduced model with its original and print / save the metrics.

    python3 evaluate.py original.fbx result.fbx [--texture tex.png] [--regions regions.json]
                        [--out metrics.json] [--renders DIR] [--closeups closeups.json]
(or: blender -b --factory-startup --python evaluate.py -- original.fbx result.fbx ...)

Metrics: face/quad/tri/n-gon counts, poles, two-sided Hausdorff + mean distance
(% of bbox diagonal, also per part), normal deviation, flipped faces,
non-manifold / degenerate checks, per UV layer: islands, flipped UV faces,
stretch and texture error, closed quad rings (overall and at joint regions).
"""
import sys
import os
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main(argv=None):
    if argv is None:
        argv = sys.argv
        argv = argv[argv.index('--') + 1:] if '--' in argv else argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('original')
    ap.add_argument('result')
    ap.add_argument('--texture', default='')
    ap.add_argument('--regions', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--renders', default='')
    ap.add_argument('--closeups', default='', help='JSON {name: {"center": [x,y,z] (world), "half": h, "view": "front"}}')
    ap.add_argument('--engine', default='WORKBENCH')
    ap.add_argument('--res', type=int, default=800)
    a = ap.parse_args(argv)
    import bpy
    from core import io_util, metrics, preview
    io_util.reset_scene()
    oo = sorted(io_util.import_file(a.original), key=lambda o: -len(o.data.polygons))
    rr = sorted(io_util.import_file(a.result), key=lambda o: -len(o.data.polygons))
    regions = json.load(open(a.regions, encoding='utf-8')) if a.regions else None
    o, r = oo[0], rr[0]
    m = metrics.evaluate(o.data, r.data, texture=a.texture or None, regions=regions)
    m['original_file'] = os.path.abspath(a.original)
    m['result_file'] = os.path.abspath(a.result)
    m['summary'] = metrics.summary_row(m)
    m['object_checks'] = {
        'scale_matches': [round(x, 6) for x in r.scale] == [round(x, 6) for x in o.scale],
        'uv_layer_names': [l.name for l in r.data.uv_layers],
        'materials': [s.material.name if s.material else None for s in r.material_slots],
        'dimensions_original': [round(x, 4) for x in o.dimensions],
        'dimensions_result': [round(x, 4) for x in r.dimensions],
    }
    imgs = []
    if a.renders:
        os.makedirs(a.renders, exist_ok=True)
        stem = os.path.splitext(os.path.basename(a.result))[0]
        for name, views in (('front_back', ('front', 'back')), ('side_34', ('left', 'front34'))):
            p = os.path.join(a.renders, '%s_%s.png' % (stem, name))
            preview.render_views([r, o], p, views=views, res=a.res, texture=a.texture or None,
                                 textured=bool(a.texture), engine=a.engine)
            imgs.append(p)
        if a.closeups:
            cu = json.load(open(a.closeups, encoding='utf-8'))
            for name, c in cu.items():
                p = os.path.join(a.renders, '%s_closeup_%s.png' % (stem, name))
                preview.render_views([r, o], p, views=(c.get('view', 'front'),), res=max(480, a.res * 3 // 4),
                                     texture=a.texture or None, textured=bool(a.texture),
                                     zoom_box=(c['center'], c['half']), engine=a.engine)
                imgs.append(p)
    m['renders'] = imgs
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(m, f, ensure_ascii=False, indent=1)
    print('SUMMARY ' + json.dumps(m['summary'], ensure_ascii=False))
    print('GEOMETRY ' + json.dumps(m['geometry']))
    print('NORMALS ' + json.dumps(m['normals']))
    print('TOPOLOGY ' + json.dumps(m['result']))
    for k, v in m['uv'].items():
        print('UV %s %s' % (k, json.dumps(v)))
    print('LOOPS ' + json.dumps(m['loops']))
    print('CHECKS ' + json.dumps(m['object_checks'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
