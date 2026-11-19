"""dev/skelretopo_check.py OUTDIR [--mark]

Mesh hygiene of a dev/skelretopo_vh.py result (reads OUTDIR/previews/skel_debug.blend and report.json, writes
the numbers into report.json under parts.<part>.checks and prints them).  Per part and per bake group:
  flipped       faces whose normal points the wrong way: closed components that are inside out (negative signed
                volume) or wound inconsistently, and every face of the closed BODY cage that fails the ray parity
                test (a ray from the face along its normal must leave the component: round 4's fingers were wound
                consistently but grew through the hand, inside out); open sheets against the source (no source
                surface within 1.5% of the height faces the same way)
  self_crossing faces of the other closed components (thin shells: cape, hair, brim) that fail the ray parity
                test: the two sides of the shell touch / cross there (reported, no failure)
  double_sided  pairs of faces of OPEN components that lie on each other with opposite normals (centres closer
                than 0.25% of the height): the "two unconnected layers" the artists reject for hard surfaces
  degenerate    zero area faces / zero length edges, duplicate faces
  loose         vertices without face, edges without face
  nonmanifold   edges with more than 2 faces
  inside_body   fraction of the vertices of a sheet group (hair, cape ...) that lie inside the closed body cage
  crossing      fraction of its edges that pass through the body cage
Exit code 1 when a part has flipped / degenerate / loose / non manifold elements."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree
from core import skelretopo, preview


def components(bm):
    cid = {}
    comps = []
    for f in bm.faces:
        if f in cid:
            continue
        k = len(comps)
        st = [f]
        cid[f] = k
        fs = []
        while st:
            g = st.pop()
            fs.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h not in cid:
                        cid[h] = k
                        st.append(h)
        comps.append(fs)
    return comps, cid


def parity_outside(bvh, p, d, eps, skip=-1, owner=None):
    """number of surfaces a ray from p along d crosses (the polygon `skip` does not count)"""
    n = 0
    o = Vector(p)
    d = Vector(d)
    for _ in range(200):
        hit = bvh.ray_cast(o, d)
        if hit[0] is None:
            break
        if (owner[hit[2]] if owner is not None else hit[2]) != skip:
            n += 1
        o = hit[0] + d * eps
    return n


def check(low, high, H, names):
    bm = bmesh.new()
    bm.from_mesh(low.data)
    bm.transform(low.matrix_world)
    bm.normal_update()
    bm.faces.ensure_lookup_table()
    lay = bm.faces.layers.int.get('sr_group')
    comps, cid = components(bm)
    closed = [all(len(e.link_faces) == 2 for f in fs for e in f.edges) for fs in comps]
    hs = skelretopo.High(high, n_samples=250000)
    kd = KDTree(len(hs.pts))
    for i, p in enumerate(hs.pts):
        kd.insert(Vector(p), i)
    kd.balance()
    eps = 1e-5 * H
    flipped, crossing = set(), set()
    kbig = max((k for k in range(len(comps)) if closed[k]), key=lambda k: len(comps[k]), default=-1)
    jit = Vector((0.0137, 0.0071, -0.0113))
    for kc, (fs, cl) in enumerate(zip(comps, closed)):
        if cl:
            vol = 0.0
            for f in fs:
                vv = [l.vert.co for l in f.loops]
                for i in range(1, len(vv) - 1):
                    vol += vv[0].dot(vv[i].cross(vv[i + 1]))
            if vol < 0 or any(not e.is_contiguous for f in fs for e in f.edges):
                flipped.update(f.index for f in fs)
                continue
            b, owner = _fan_bvh(fs)
            for k, f in enumerate(fs):
                if f.calc_area() < 1e-9 * H * H:
                    continue
                c = f.calc_center_median()
                n = (f.normal + jit).normalized()
                if parity_outside(b, c + f.normal * eps, n, eps, k, owner) % 2 == 1:
                    (flipped if kc == kbig else crossing).add(f.index)
        else:
            for f in fs:
                c = f.calc_center_median()
                near = kd.find_range(c, 0.015 * H)
                if not near:
                    near = kd.find_n(c, 8)
                if max(float(f.normal.dot(Vector(hs.pt_n[i]))) for _co, i, _d in near) < 0.0:
                    flipped.add(f.index)
    deg = [f.index for f in bm.faces if f.calc_area() < 1e-9 * H * H or any(e.calc_length() < 1e-6 * H for e in f.edges)]
    seen, dup = {}, []
    for f in bm.faces:
        k = tuple(sorted(v.index for v in f.verts))
        if k in seen:
            dup.append(f.index)
        seen[k] = f.index
    fk = KDTree(len(bm.faces))
    for f in bm.faces:
        fk.insert(f.calc_center_median(), f.index)
    fk.balance()
    pairs = []
    for f in bm.faces:
        if closed[cid[f]]:
            continue
        for _co, i, _d in fk.find_range(f.calc_center_median(), 0.006 * H):
            g = bm.faces[i]
            if i > f.index and not closed[cid[g]] and g.normal.dot(f.normal) < -0.95:
                pairs.append((f.index, i))
                break
    # the body cage: the biggest closed component
    body = None
    ci = [k for k in range(len(comps)) if closed[k]]
    if ci:
        kb = max(ci, key=lambda k: len(comps[k]))
        body = BVHTree.FromPolygons([v.co[:] for f in comps[kb] for v in f.verts], _poly_index(comps[kb]))
    res = {'faces': len(bm.faces), 'tris': sum(len(f.verts) - 2 for f in bm.faces), 'components': len(comps), 'closed_components': int(sum(closed)),
           'flipped': len(flipped), 'self_crossing': len(crossing), 'double_sided_pairs': len(pairs), 'degenerate': len(deg), 'duplicate_faces': len(dup),
           'loose_verts': sum(1 for v in bm.verts if not v.link_faces), 'loose_edges': sum(1 for e in bm.edges if not e.link_faces),
           'nonmanifold_edges': sum(1 for e in bm.edges if len(e.link_faces) > 2), 'open_edges': sum(1 for e in bm.edges if len(e.link_faces) == 1), 'groups': {}}
    other_bvh = {}
    if lay is not None and names:
        pf = {a for a, _b in pairs} | {b for _a, b in pairs}
        for gi, n in enumerate(names):
            fs = [f for f in bm.faces if f[lay] == gi]
            if not fs:
                continue
            g = {'tris': sum(len(f.verts) - 2 for f in fs), 'closed': bool(all(closed[cid[f]] for f in fs)), 'flipped': sum(f.index in flipped for f in fs), 'self_crossing': sum(f.index in crossing for f in fs),
                 'double_sided_faces': sum(f.index in pf for f in fs)}
            if body is not None and n != 'body':
                vs = {v for f in fs for v in f.verts}
                ins = sum(parity_outside(body, v.co, (0.577 + 0.01, 0.577, 0.577 - 0.01), eps) % 2 == 1 for v in vs)
                es = {e for f in fs for e in f.edges}
                cr = 0
                for e in es:
                    a, b_ = e.verts[0].co, e.verts[1].co
                    d = b_ - a
                    if d.length > 1e-9 and body.ray_cast(a, d.normalized(), d.length)[0] is not None:
                        cr += 1
                g['inside_body_vert_pct'] = round(100.0 * ins / max(len(vs), 1), 1)
                g['edges_crossing_body_pct'] = round(100.0 * cr / max(len(es), 1), 1)
                g['z_median_by_H'] = round(float(np.median([v.co[2] for v in vs]) - hs.lo[2]) / H, 2)
                # against the other shells (a quiver in the coat): edges that pass through a closed component of another group
                own_c = {cid[f] for f in fs}
                cr2 = 0
                for kc2 in range(len(comps)):
                    if kc2 in own_c or kc2 == kb or not closed[kc2]:
                        continue
                    if kc2 not in other_bvh:
                        other_bvh[kc2] = _fan_bvh(comps[kc2])[0]
                for e in es:
                    a, b_ = e.verts[0].co, e.verts[1].co
                    d = b_ - a
                    if d.length > 1e-9 and any(other_bvh[k2].ray_cast(a, d.normalized(), d.length)[0] is not None for k2 in other_bvh if k2 not in own_c):
                        cr2 += 1
                g['edges_crossing_other_shells_pct'] = round(100.0 * cr2 / max(len(es), 1), 1)
            res['groups'][n] = g
    # hair must be part of the head mesh (round 6): no cloth / hair shell of its own above the shoulders
    res['hair_shells'] = sum(1 for n, g in res['groups'].items() if n.startswith('extra') and g.get('z_median_by_H', 0) > 0.76)
    global LAST_CROSSING
    LAST_CROSSING = sorted(crossing)
    marks = (sorted(flipped), sorted({a for a, _b in pairs} | {b for _a, b in pairs}), deg)
    bm.free()
    return res, marks


def _poly_index(fs):
    out, k = [], 0
    for f in fs:
        n = len(f.verts)
        out.append(tuple(range(k, k + n)))
        k += n
    return out


def _fan_bvh(fs):
    """BVH of the faces split as a fan from their first vertex (what an exporter / engine does with a quad; the
    two sides of a thin shell are built so that this split matches) -> bvh, face index of every triangle"""
    co, tri, owner = [], [], []
    for k, f in enumerate(fs):
        b = len(co)
        co += [v.co[:] for v in f.verts]
        for i in range(1, len(f.verts) - 1):
            tri.append((b, b + i, b + i + 1))
            owner.append(k)
    return BVHTree.FromPolygons(co, tri), owner


def main():
    outdir = sys.argv[1]
    rep_path = os.path.join(outdir, 'report.json')
    report = json.load(open(rep_path))
    bpy.ops.wm.open_mainfile(filepath=os.path.join(outdir, 'previews', 'skel_debug.blend'))
    stem = os.path.splitext(report['source'])[0]
    highs = [o for o in bpy.data.objects if o.type == 'MESH' and o.name.startswith('high_')]
    bad = False
    main_h = None
    for part, rep in report['parts'].items():
        low = bpy.data.objects.get('%s_%s' % (stem, part))
        if low is None:
            continue
        lc = np.array(preview._world_bbox(low)).mean(0)
        high = min(highs, key=lambda h: float(np.linalg.norm(np.array(preview._world_bbox(h)).mean(0) - lc)))
        H = rep.get('build', {}).get('height') or float(np.linalg.norm(np.ptp(np.array(preview._world_bbox(high)), axis=0)))
        names = rep.get('groups')
        if names is None:                      # reports of the earlier rounds: the groups in the order of the builder
            b = rep.get('build', {})
            names = sorted(set(['body'] + [e['name'] for e in b.get('extras', [])] + (['sheets'] if b.get('tris', {}).get('sheets') or b.get('sheet_faces') else [])
                               + (['rods'] if b.get('tris', {}).get('rods') else [])))
        res, marks = check(low, high, H, names)
        rep['checks'] = res
        ok = not (res['flipped'] or res['degenerate'] or res['duplicate_faces'] or res['loose_verts'] or res['loose_edges'] or res['nonmanifold_edges'])
        bad |= not ok
        print('CHECK', part, 'OK' if ok else 'FAILED', json.dumps(res, ensure_ascii=False), flush=True)
    json.dump(report, open(rep_path, 'w'), ensure_ascii=False, indent=1, default=float)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
