"""Objective comparison of a reduced mesh with its original.

Everything is measured in the object's local space and reported relative to
the bounding-box diagonal of the original.  Only bpy/bmesh/mathutils/numpy.
"""
import math
import numpy as np
import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree


class TriMesh:
    """triangulated copy of a Blender mesh with per-corner UVs"""

    def __init__(self, me):
        bm = bmesh.new()
        bm.from_mesh(me)
        self.n_poly = len(bm.faces)
        self.poly_sizes = [len(f.verts) for f in bm.faces]
        self.pos = np.array([v.co[:] for v in bm.verts], dtype=np.float64).reshape(-1, 3)
        uvl = list(bm.loops.layers.uv.values())
        self.uv_names = [l.name for l in bm.loops.layers.uv.values()]
        tris = bm.calc_loop_triangles()
        self.tri = np.array([[l.vert.index for l in t] for t in tris], dtype=np.int64).reshape(-1, 3)
        self.tri_poly = np.array([t[0].face.index for t in tris], dtype=np.int64)
        self.tri_uv = [np.array([[l[u].uv[:] for l in t] for t in tris], dtype=np.float64).reshape(-1, 3, 2) for u in uvl]
        self.poly_normal = np.array([f.normal[:] for f in bm.faces], dtype=np.float64).reshape(-1, 3)
        self.poly_mat = np.array([f.material_index for f in bm.faces], dtype=np.int64)
        self.bm = bm
        p = self.pos[self.tri]
        self.tri_area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
        self.bvh = BVHTree.FromPolygons([tuple(x) for x in self.pos], [tuple(int(i) for i in t) for t in self.tri])
        # connected components per polygon
        comp = [-1] * len(bm.faces)
        nc = 0
        for f in bm.faces:
            if comp[f.index] >= 0:
                continue
            st = [f]
            comp[f.index] = nc
            while st:
                g = st.pop()
                for v in g.verts:
                    for h in v.link_faces:
                        if comp[h.index] < 0:
                            comp[h.index] = nc
                            st.append(h)
            nc += 1
        self.poly_comp = np.array(comp, dtype=np.int64)
        self.n_comp = nc

    def sample(self, n, rng):
        """area-uniform samples -> (points, tri index, bary)"""
        a = self.tri_area
        cdf = np.cumsum(a)
        if cdf[-1] <= 0:
            idx = rng.integers(0, len(a), n)
        else:
            idx = np.minimum(np.searchsorted(cdf, rng.random(n) * cdf[-1]), len(a) - 1)
        r1 = np.sqrt(rng.random(n))
        r2 = rng.random(n)
        b = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], axis=1)
        p = self.pos[self.tri[idx]]
        pts = (p * b[:, :, None]).sum(1)
        return pts, idx, b

    def closest(self, pts):
        """-> closest points, tri index, bary, distance"""
        n = len(pts)
        q = np.zeros((n, 3))
        ti = np.zeros(n, dtype=np.int64)
        d = np.zeros(n)
        fn = self.bvh.find_nearest
        for i in range(n):
            co, nrm, idx, dist = fn(pts[i])
            q[i] = co
            ti[i] = idx
            d[i] = dist
        return q, ti, bary(q, self.pos[self.tri[ti]]), d

    def closest_compatible(self, pts, normals, diag):
        """closest original triangle whose normal agrees best (thin two-sided
        parts: do not match the back side of a neighbouring shell)"""
        n = len(pts)
        ti = np.zeros(n, dtype=np.int64)
        dots = np.zeros(n)
        for i in range(n):
            co, nrm, idx, dist = self.bvh.find_nearest(pts[i])
            best = idx
            bd = float(np.dot(self.poly_normal[self.tri_poly[idx]], normals[i]))
            if bd < 0.5:
                for co2, nrm2, idx2, dist2 in self.bvh.find_nearest_range(pts[i], dist * 1.5 + 0.004 * diag):
                    d2 = float(np.dot(self.poly_normal[self.tri_poly[idx2]], normals[i]))
                    if d2 > bd:
                        bd = d2
                        best = idx2
            ti[i] = best
            dots[i] = bd
        return ti, dots

    def free(self):
        self.bm.free()


def bary(q, tri):
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1, v2 = b - a, c - a, q - a
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1)
    d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den[np.abs(den) < 1e-30] = 1e-30
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    out = np.stack([1 - v - w, v, w], axis=1)
    out = np.clip(out, 0, 1)
    return out / out.sum(1, keepdims=True)


def topology_stats(bm):
    q = sum(1 for f in bm.faces if len(f.verts) == 4)
    t = sum(1 for f in bm.faces if len(f.verts) == 3)
    n = len(bm.faces) - q - t
    tri_after = sum(len(f.verts) - 2 for f in bm.faces)
    nonman = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    boundary = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    wire = sum(1 for e in bm.edges if len(e.link_faces) == 0)
    loose = sum(1 for v in bm.verts if not v.link_faces)
    areas = np.array([f.calc_area() for f in bm.faces]) if len(bm.faces) else np.zeros(1)
    degenerate = int((areas < 1e-8 * max(areas.mean(), 1e-30)).sum())
    poles = 0
    interior = 0
    for v in bm.verts:
        if not v.link_faces or v.is_boundary:
            continue
        interior += 1
        if len(v.link_edges) != 4:
            poles += 1
    # non-manifold vertices (bow ties)
    bow = 0
    for v in bm.verts:
        if v.link_faces and not v.is_manifold and not any(len(e.link_faces) > 2 for e in v.link_edges) and not v.is_boundary:
            bow += 1
    nf = max(1, len(bm.faces))
    return {
        'verts': len(bm.verts), 'faces': len(bm.faces), 'quads': q, 'tris': t, 'ngons': n,
        'quad_pct': round(100.0 * q / nf, 2), 'tri_pct': round(100.0 * t / nf, 2),
        'tris_after_triangulation': tri_after,
        'poles': poles, 'pole_pct_of_interior_verts': round(100.0 * poles / max(1, interior), 2),
        'non_manifold_edges': nonman, 'non_manifold_verts': bow, 'boundary_edges': boundary,
        'wire_edges': wire, 'loose_verts': loose, 'degenerate_faces': degenerate,
    }


def uv_islands(bm, layer):
    """number of UV islands, number of faces with negative / zero UV area"""
    par = list(range(len(bm.faces)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    seams = 0
    for e in bm.edges:
        if len(e.link_loops) != 2:
            continue
        l1, l2 = e.link_loops
        if l1.vert == l2.vert:
            same = (l1[layer].uv - l2[layer].uv).length < 1e-6 and \
                   (l1.link_loop_next[layer].uv - l2.link_loop_next[layer].uv).length < 1e-6
        else:
            same = (l1[layer].uv - l2.link_loop_next[layer].uv).length < 1e-6 and \
                   (l1.link_loop_next[layer].uv - l2[layer].uv).length < 1e-6
        if same:
            par[find(l1.face.index)] = find(l2.face.index)
        else:
            seams += 1
    neg = zero = 0
    for f in bm.faces:
        s = 0.0
        ls = f.loops
        n = len(ls)
        for i in range(n):
            a = ls[i][layer].uv
            b = ls[(i + 1) % n][layer].uv
            s += a.x * b.y - b.x * a.y
        if abs(s) < 1e-12:
            zero += 1
        elif s < 0:
            neg += 1
    return len({find(i) for i in range(len(bm.faces))}), neg, zero, seams


def closed_loops(bm):
    """closed edge loops (through regular valence-4 vertices of quads).
    -> list of vertex index lists"""
    done = set()
    loops = []

    def nxt(e, v):
        # continue across a regular vertex: the edge sharing no face with e
        if len(v.link_edges) != 4 or v.is_boundary:
            return None
        fs = set(e.link_faces)
        for e2 in v.link_edges:
            if e2 is not e and not (fs & set(e2.link_faces)):
                return e2
        return None
    for e0 in bm.edges:
        if e0.index in done or len(e0.link_faces) != 2:
            continue
        path = [e0]
        v = e0.verts[1]
        e = e0
        closed = False
        seen = {e0.index}
        while True:
            e2 = nxt(e, v)
            if e2 is None:
                break
            if e2 is e0:
                closed = True
                break
            if e2.index in seen:
                break
            seen.add(e2.index)
            path.append(e2)
            v = e2.other_vert(v)
            e = e2
        if closed:
            done |= seen
            vs = []
            v = e0.verts[0]
            for ed in path:
                vs.append(v.index)
                v = ed.other_vert(v)
            loops.append(vs)
        else:
            done.add(e0.index)
    return loops


def closed_bands(bm):
    """closed strips of quads (a ring of faces around a limb) -> list of face lists"""
    seen = set()
    bands = []
    for f0 in bm.faces:
        if len(f0.verts) != 4:
            continue
        for l0 in f0.loops[:2]:
            key = (f0.index, l0.edge.index)
            if key in seen:
                continue
            faces = []
            l = l0
            ok = False
            while True:
                f = l.face
                if len(f.verts) != 4:
                    break
                seen.add((f.index, l.edge.index))
                opp = l.link_loop_next.link_loop_next
                seen.add((f.index, opp.edge.index))
                faces.append(f.index)
                if len(opp.edge.link_loops) != 2:
                    break
                l = opp.link_loop_radial_next
                if l.face is f0:
                    ok = l.edge is l0.edge
                    break
                if len(faces) > len(bm.faces):
                    break
            if ok and len(faces) >= 3 and len(set(faces)) == len(faces):
                bands.append(faces)
    return bands


def load_image(path):
    import bpy
    img = bpy.data.images.load(path, check_existing=True)
    w, h = img.size
    a = np.empty(w * h * img.channels, dtype=np.float32)
    img.pixels.foreach_get(a)
    return a.reshape(h, w, img.channels)[:, :, :3]


def sample_tex(img, uv):
    h, w = img.shape[:2]
    x = (uv[:, 0] % 1.0) * w - 0.5
    y = (uv[:, 1] % 1.0) * h - 0.5
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    x0 %= w
    y0 %= h
    return (img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy) + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy


def evaluate(orig_me, res_me, texture=None, regions=None, n_samples=40000, seed=1):
    """-> dict of metrics. regions: {name: {'center': [x,y,z], 'radius': r}}
    in the mesh's local coordinates (joint zones for the loop check)."""
    rng = np.random.default_rng(seed)
    A = TriMesh(orig_me)
    B = TriMesh(res_me)
    diag = float(np.linalg.norm(A.pos.max(0) - A.pos.min(0)))
    out = {'bbox_diag': diag}
    out['original'] = topology_stats(A.bm)
    out['result'] = topology_stats(B.bm)
    out['face_ratio'] = round(B.n_poly / max(1, A.n_poly), 4)

    # ---- geometry: two sided distances
    pa, ta, ba = A.sample(n_samples, rng)
    pa = np.concatenate([pa, A.pos])
    pb, tb, bb = B.sample(n_samples, rng)
    qa, tqa, bqa, da = B.closest(pa)          # original -> result
    qb, tqb, bqb, db = A.closest(pb)          # result -> original
    g = {
        'hausdorff_orig_to_result_pct': round(100 * float(da.max()) / diag, 4),
        'hausdorff_result_to_orig_pct': round(100 * float(db.max()) / diag, 4),
        'hausdorff_pct': round(100 * float(max(da.max(), db.max())) / diag, 4),
        'mean_dist_pct': round(100 * float(0.5 * (da.mean() + db.mean())) / diag, 4),
        'rms_dist_pct': round(100 * float(math.sqrt(0.5 * ((da ** 2).mean() + (db ** 2).mean()))) / diag, 4),
        'p99_dist_pct': round(100 * float(max(np.percentile(da, 99), np.percentile(db, 99))) / diag, 4),
    }
    out['geometry'] = g
    # per original part (connected component)
    comp_s = A.poly_comp[A.tri_poly[ta]]
    parts = []
    for c in range(A.n_comp):
        msk = comp_s == c
        if msk.sum() < 5:
            continue
        pts = pa[:n_samples][msk]
        d = da[:n_samples][msk]
        ext = pts.max(0) - pts.min(0)
        parts.append({'part': int(c), 'center': [round(float(x), 2) for x in pts.mean(0)],
                      'size': [round(float(x), 2) for x in ext],
                      'max_pct': round(100 * float(d.max()) / diag, 4),
                      'mean_pct': round(100 * float(d.mean()) / diag, 4),
                      'max_rel_to_part_pct': round(100 * float(d.max()) / max(1e-9, float(np.linalg.norm(ext))), 3)})
    parts.sort(key=lambda r: -r['max_rel_to_part_pct'])
    out['parts_worst'] = parts[:8]

    # ---- normals (against the best matching original face near by)
    sub = slice(0, min(len(pb), 12000))
    nb = B.poly_normal[B.tri_poly[tb[sub]]]
    _ti, dots = A.closest_compatible(pb[sub], nb, diag)
    ang = np.degrees(np.arccos(np.clip(dots, -1, 1)))
    out['normals'] = {
        'mean_deviation_deg': round(float(ang.mean()), 3),
        'p95_deviation_deg': round(float(np.percentile(ang, 95)), 3),
        'samples_over_90deg_pct': round(100 * float((ang > 90).mean()), 3),
    }
    cents = np.array([f.calc_center_median()[:] for f in B.bm.faces]).reshape(-1, 3)
    tc, d2 = A.closest_compatible(cents, B.poly_normal, diag)
    out['normals']['flipped_faces'] = int((d2 < -0.3).sum())
    # winding consistency inside the result itself
    incons = 0
    for e in B.bm.edges:
        if len(e.link_loops) == 2 and e.link_loops[0].vert == e.link_loops[1].vert:
            incons += 1
    out['normals']['inconsistent_winding_edges'] = incons

    # ---- UV
    uvres = {}
    for k, name in enumerate(A.uv_names):
        if k >= len(B.tri_uv):
            uvres[name] = {'missing': True}
            continue
        la = A.bm.loops.layers.uv[k]
        lb = B.bm.loops.layers.uv[k]
        ia, nega, zeroa, seama = uv_islands(A.bm, la)
        ib, negb, zerob, seamb = uv_islands(B.bm, lb)
        # uv of result sample vs uv of the closest original point
        uvb = (B.tri_uv[k][tb] * bb[:, :, None]).sum(1)
        uva = (A.tri_uv[k][tqb] * bqb[:, :, None]).sum(1)
        duv = np.linalg.norm(uvb - uva, axis=1)
        ext = A.tri_uv[k].reshape(-1, 2)
        scale = float(np.linalg.norm(ext.max(0) - ext.min(0))) or 1.0
        # stretch: (uv area / 3d area) of the result triangle vs original triangle
        def dens(M, t):
            u = M.tri_uv[k][t]
            a2 = np.abs((u[:, 1, 0] - u[:, 0, 0]) * (u[:, 2, 1] - u[:, 0, 1]) - (u[:, 2, 0] - u[:, 0, 0]) * (u[:, 1, 1] - u[:, 0, 1])) * 0.5
            return a2 / np.maximum(M.tri_area[t], 1e-20)
        ra = dens(A, tqb)
        rb = dens(B, tb)
        ok = (ra > 1e-12) & (rb > 1e-12)
        st = np.abs(np.log2(np.sqrt(rb[ok] / ra[ok]))) if ok.any() else np.zeros(1)
        def signs(M, layer):
            out_ = np.zeros(len(M.bm.faces))
            for f in M.bm.faces:
                sa = 0.0
                ls = f.loops
                for i in range(len(ls)):
                    a_ = ls[i][layer].uv
                    b_ = ls[(i + 1) % len(ls)][layer].uv
                    sa += a_.x * b_.y - b_.x * a_.y
                out_[f.index] = 0 if abs(sa) < 1e-12 else (1 if sa > 0 else -1)
            return out_
        sa_ = signs(A, la)[A.tri_poly[tc]]
        sb_ = signs(B, lb)
        flipped_vs_orig = int(((sa_ * sb_) < 0).sum())
        uvres[name] = {
            'flipped_uv_faces_vs_original': flipped_vs_orig,
            'islands_original': ia, 'islands_result': ib,
            'seam_edges_original': seama, 'seam_edges_result': seamb,
            'flipped_uv_faces_original': nega, 'flipped_uv_faces_result': negb,
            'zero_area_uv_faces_original': zeroa, 'zero_area_uv_faces_result': zerob,
            'uv_shift_mean_pct_of_uv_extent': round(100 * float(np.median(duv)) / scale, 4),
            'uv_shift_p90_pct_of_uv_extent': round(100 * float(np.percentile(duv, 90)) / scale, 4),
            'stretch_mean_log2': round(float(st.mean()), 4),
            'stretch_p95_log2': round(float(np.percentile(st, 95)), 4),
            'stretch_max_log2': round(float(st.max()), 4),
        }
        if k == 0 and texture:
            try:
                if isinstance(texture, dict):
                    # {material index: image path}: every sample uses the image of its own material
                    ma = A.poly_mat[A.tri_poly[tqb]]
                    mb = B.poly_mat[B.tri_poly[tb]]
                    ca = np.zeros((len(uva), 3))
                    cb = np.zeros((len(uvb), 3))
                    have = np.zeros(len(uva), dtype=bool)
                    for mi_, path_ in texture.items():
                        if not path_:
                            continue
                        img = load_image(path_)
                        sa_m = ma == int(mi_)
                        sb_m = mb == int(mi_)
                        ca[sa_m] = sample_tex(img, uva[sa_m])
                        cb[sb_m] = sample_tex(img, uvb[sb_m])
                        have |= sb_m
                    ca, cb = ca[have], cb[have]
                    uvres[name]['texture_samples_other_material_pct'] = round(100 * float((ma != mb).mean()), 3)
                else:
                    img = load_image(texture)
                    ca = sample_tex(img, uva)
                    cb = sample_tex(img, uvb)
                err = np.abs(ca - cb).mean(1) * 255
                # noise floor: the original against itself with the same spatial jitter
                uvres[name]['texture_error_mean_0_255'] = round(float(err.mean()), 3)
                uvres[name]['texture_error_p95_0_255'] = round(float(np.percentile(err, 95)), 3)
                uvres[name]['texture_samples_over_30_pct'] = round(100 * float((err > 30).mean()), 3)
            except Exception as ex:      # texture is optional
                uvres[name]['texture_error'] = 'failed: %s' % ex
    out['uv'] = uvres

    # ---- loops
    la_ = closed_loops(A.bm)
    lb_ = closed_loops(B.bm)
    A.bm.faces.ensure_lookup_table()
    B.bm.faces.ensure_lookup_table()
    ba_ = closed_bands(A.bm)
    bb_ = closed_bands(B.bm)
    out['loops'] = {'closed_loops_original': len(la_), 'closed_loops_result': len(lb_),
                    'closed_quad_rings_original': len(ba_), 'closed_quad_rings_result': len(bb_)}
    if regions:
        reg = {}
        for name, r in regions.items():
            c = np.array(r['center'], dtype=np.float64)
            rad = float(r['radius'])

            def cnt(M, bands):
                n = 0
                tri_touch = 0
                for fs in bands:
                    cs = np.array([M.bm.faces[i].calc_center_median()[:] for i in fs])
                    if np.linalg.norm(cs.mean(0) - c) < rad and np.linalg.norm(cs - c, axis=1).max() < 3.0 * rad:
                        n += 1
                for f in M.bm.faces:
                    if len(f.verts) == 3:
                        d_ = np.array(f.calc_center_median()[:]) - c
                        if d_.dot(d_) < rad * rad:
                            tri_touch += 1
                return n, tri_touch
            na_, ta_ = cnt(A, ba_)
            nb_, tb_ = cnt(B, bb_)
            reg[name] = {'ring_loops_original': na_, 'ring_loops_result': nb_,
                         'tris_original': ta_, 'tris_result': tb_}
        out['loops']['regions'] = reg
    A.free()
    B.free()
    return out


def summary_row(m):
    """flat one-line summary used for tables"""
    r = m['result']
    g = m['geometry']
    uv0 = next(iter(m['uv'].values())) if m['uv'] else {}
    return {
        'faces': r['faces'], 'quads': r['quads'], 'tris': r['tris'], 'ngons': r['ngons'],
        'unity_tris': r['tris_after_triangulation'], 'quad_pct': r['quad_pct'],
        'hausdorff_pct': g['hausdorff_pct'], 'mean_pct': g['mean_dist_pct'],
        'normal_dev_deg': m['normals']['mean_deviation_deg'], 'flipped': m['normals']['flipped_faces'],
        'nonmanifold': r['non_manifold_edges'], 'degenerate': r['degenerate_faces'],
        'uv_islands': '%s/%s' % (uv0.get('islands_result'), uv0.get('islands_original')),
        'uv_flipped': uv0.get('flipped_uv_faces_vs_original'),
        'tex_err': uv0.get('texture_error_mean_0_255'),
        'quad_rings': '%s/%s' % (m['loops']['closed_quad_rings_result'], m['loops']['closed_quad_rings_original']),
    }
