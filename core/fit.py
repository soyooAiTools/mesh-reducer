"""Least-squares re-fit of the reduced mesh to the original surface.

Loop removal always cuts corners (a 12-sided arm that becomes 8-sided lies
inside the original).  Here every vertex is moved along its normal so that the
reduced surface is as close as possible to sample points of the original
(per connected part, so the skirt never pulls on the legs).  Movement is along
the normal only, so UVs stay valid.
"""
import numpy as np
from mathutils.bvhtree import BVHTree


def _triangulate(pos, faces):
    tris = []
    for vs in faces:
        n = len(vs)
        if n == 3:
            tris.append(vs)
        elif n == 4:
            a, b, c, d = vs
            if np.sum((pos[a] - pos[c]) ** 2) <= np.sum((pos[b] - pos[d]) ** 2):
                tris.append((a, b, c)); tris.append((a, c, d))
            else:
                tris.append((a, b, d)); tris.append((b, c, d))
        else:
            for i in range(1, n - 1):
                tris.append((vs[0], vs[i], vs[i + 1]))
    return tris


def _bary(q, a, b, c):
    v0, v1, v2 = b - a, c - a, q - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-30:
        return 1 / 3, 1 / 3, 1 / 3
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1 - v - w
    u, v, w = max(u, 0), max(v, 0), max(w, 0)
    s = u + v + w
    return u / s, v / s, w / s


def fit(red, iterations=4, n_samples=24000, seed=7, strength=0.85):
    m = red.m
    rng = np.random.default_rng(seed)
    opos = red.orig_pos
    # ---- samples on the original, per component
    ncomp = m.n_comp
    otris = [[] for _ in range(ncomp)]
    for fid, vs in enumerate(red.orig_faces):
        c = red.orig_fcomp[fid]
        for i in range(1, len(vs) - 1):
            otris[c].append((vs[0], vs[i], vs[i + 1]))
    areas = []
    for c in range(ncomp):
        t = np.array(otris[c], dtype=np.int64).reshape(-1, 3)
        p = opos[t]
        areas.append(0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1))
    total = sum(float(a.sum()) for a in areas) or 1.0
    samples = []
    for c in range(ncomp):
        t = np.array(otris[c], dtype=np.int64).reshape(-1, 3)
        a = areas[c]
        if len(t) == 0 or a.sum() <= 0:
            samples.append(np.zeros((0, 3)))
            continue
        n = max(60, int(n_samples * float(a.sum()) / total))
        cdf = np.cumsum(a)
        idx = np.minimum(np.searchsorted(cdf, rng.random(n) * cdf[-1]), len(a) - 1)
        r1 = np.sqrt(rng.random(n))
        r2 = rng.random(n)
        b = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], axis=1)
        pts = (opos[t[idx]] * b[:, :, None]).sum(1)
        vs = np.unique(t)
        samples.append(np.concatenate([pts, opos[vs]]))

    obvh = []
    for c in range(ncomp):
        if not otris[c]:
            obvh.append(None)
            continue
        vids = sorted({v for t in otris[c] for v in t})
        loc = {v: i for i, v in enumerate(vids)}
        obvh.append(BVHTree.FromPolygons([tuple(opos[v]) for v in vids], [tuple(loc[v] for v in t) for t in otris[c]]))
    vcomp = {}
    faces_by_comp = [[] for _ in range(ncomp)]
    for fid, vs in enumerate(m.faces):
        if vs is not None:
            faces_by_comp[m.fcomp[fid]].append(vs)
            for v in vs:
                vcomp[v] = m.fcomp[fid]
    pos = m.pos
    nv = len(pos)
    mir = red.mirror
    ax = red.sym_axis
    # local size limit for the correction
    elen = np.zeros(nv)
    ecnt = np.zeros(nv)
    for vs in m.faces:
        if vs is None:
            continue
        n = len(vs)
        for i in range(n):
            a, b = vs[i], vs[(i + 1) % n]
            l = float(np.linalg.norm(pos[a] - pos[b]))
            elen[a] += l; ecnt[a] += 1
            elen[b] += l; ecnt[b] += 1
    elen = elen / np.maximum(ecnt, 1)
    start = pos.copy()
    for it in range(iterations):
        vn = np.zeros((nv, 3))
        for vs in m.faces:
            if vs is None:
                continue
            n = m.face_normal(vs)
            for v in vs:
                vn[v] += n
        ln = np.linalg.norm(vn, axis=1)
        ln[ln < 1e-20] = 1.0
        vn /= ln[:, None]
        acc = np.zeros((nv, 3))
        wacc = np.zeros(nv)
        for c in range(ncomp):
            if not faces_by_comp[c] or len(samples[c]) == 0:
                continue
            tris = _triangulate(pos, faces_by_comp[c])
            vids = sorted({v for t in tris for v in t})
            loc = {v: i for i, v in enumerate(vids)}
            bvh = BVHTree.FromPolygons([tuple(pos[v]) for v in vids], [tuple(loc[v] for v in t) for t in tris])
            fn = bvh.find_nearest
            for s in samples[c]:
                co, nrm, idx, dist = fn(s)
                if co is None:
                    continue
                q = np.array(co)
                i, j, k = tris[idx]
                bu, bv, bw = _bary(q, pos[i], pos[j], pos[k])
                r = s - q
                acc[i] += bu * r; wacc[i] += bu
                acc[j] += bv * r; wacc[j] += bv
                acc[k] += bw * r; wacc[k] += bw
        ok = wacc > 1e-9
        delta = np.zeros((nv, 3))
        delta[ok] = acc[ok] / wacc[ok][:, None]
        off = (delta * vn).sum(1) * strength
        if ax is not None:
            for v in range(nv):
                mv = mir[v]
                if mv >= 0 and mv > v:
                    o = 0.5 * (off[v] + off[mv])
                    off[v] = off[mv] = o
        newpos = pos + vn * off[:, None]
        # clamp total displacement
        d = newpos - start
        dl = np.linalg.norm(d, axis=1)
        lim = 0.35 * elen
        over = dl > lim
        if over.any():
            newpos[over] = start[over] + d[over] * (lim[over] / dl[over])[:, None]
        if ax is not None:
            for v in range(nv):
                mv = mir[v]
                if mv == v:
                    newpos[v][ax] = pos[v][ax]
                elif mv > v:
                    q = newpos[v].copy(); q[ax] = -q[ax]
                    newpos[mv] = q
        # a vertex may not end up further from the original than the error it repairs
        dres = np.linalg.norm(delta, axis=1)
        for v, c in vcomp.items():
            if obvh[c] is None or abs(off[v]) < 1e-12:
                continue
            d0 = obvh[c].find_nearest(tuple(pos[v]))[3] or 0.0
            lim_v = max(d0, 0.6 * dres[v]) + 1e-9
            for _try in range(3):
                d1 = obvh[c].find_nearest(tuple(newpos[v]))[3] or 0.0
                if d1 <= lim_v:
                    break
                newpos[v] = pos[v] + 0.5 * (newpos[v] - pos[v])
            else:
                newpos[v] = pos[v]
        if ax is not None:
            for v in range(nv):
                mv = mir[v]
                if mv > v:
                    q = newpos[v].copy(); q[ax] = -q[ax]
                    q2 = newpos[mv].copy(); q2[ax] = -q2[ax]
                    # keep the smaller move of the pair
                    if np.sum((q - pos[mv]) ** 2) <= np.sum((newpos[mv] - pos[mv]) ** 2):
                        newpos[mv] = q
                    else:
                        newpos[v] = q2
        # reject moves that flip / fold a face
        bad = set()
        for vs in m.faces:
            if vs is None:
                continue
            n0 = m.face_normal(vs, pos)
            n1 = m.face_normal(vs, newpos)
            if float(n0 @ n1) < 0.5 * float(np.linalg.norm(n0)) * float(np.linalg.norm(n1)):
                bad.update(vs)
        for v in bad:
            newpos[v] = pos[v]
            if ax is not None and mir[v] >= 0:
                newpos[mir[v]] = pos[mir[v]]
        pos[:] = newpos
    return float(np.linalg.norm(pos - start, axis=1).max())
