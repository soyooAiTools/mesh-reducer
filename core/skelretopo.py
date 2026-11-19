"""core/skelretopo.py: skeleton driven automatic retopology ("骨骼驱动重拓扑").

The low model is NOT derived from the source topology.  A new cage is built from
the skeleton and the surface of the high model:

  1. analyse_skeleton(): torso / head, arms, legs are found from the bone
     hierarchy, the mirror symmetry and the skinned surface (no bone names).
  2. every high face gets an owner: a body chain, or an "extra" bone group
     (cape, skirt, hair ...: bones that are not part of the body chains).
  3. body: rings of N vertices are swept along the chains and fitted to the
     outline of the owner's surface in the ring plane (surface samples, radial
     function, spikes removed); 3 loops at elbow / knee; arms are bridged into
     holes of the torso tube, legs into the two halves of its lowest ring.
  4. extras: large thin ones become open (or closed) arc grids around the axis
     of the chain they hang on, compact ones box prisms along their main axis.
     What sticks far out of the body cage (fins, pads) gets a prism too.
  5. props (weapons, retopo.detect_parts): prisms along their main axis.
  6. new UVs (seams along the tubes), bake BC / N / AO per owner group with a
     cage (core.retopo.bake_maps), skin weights per owner group, one mesh.
  7. (round 3) thin free standing pieces of the body (hat brim, feather, fins,
     fingers: find_protrusions) are not part of the cage: flat two sided sheets
     with alpha outline (build_sheet) or box + point rods (build_rod).  Props:
     box strips for what is thick + one flat sheet over the outline.
     Budget: body (cage, rods, compact extras) + sheets (arcs, flat sheets).

All lengths are relative to the height H of the main part."""
import os
import math
import time
import collections

import bpy
import bmesh
import numpy as np
from mathutils import Vector, geometry as mgeo
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from . import api, retopo


# ====================================================================== high model
class High:
    """the part of the source that is retopologised, in world space"""

    def __init__(self, obj, faces=None, n_samples=400000, seed=3):
        me = obj.data
        mw = np.array(obj.matrix_world)
        co = np.empty(len(me.vertices) * 3, dtype=np.float64)
        me.vertices.foreach_get('co', co)
        co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        self.obj = obj
        self.co = co
        self.gnames = [g.name for g in obj.vertex_groups]
        ng = len(self.gnames)
        W = np.zeros((len(me.vertices), max(ng, 1)), dtype=np.float32)
        for v in me.vertices:
            for g in v.groups:
                if g.weight > 0 and g.group < ng:
                    W[v.index, g.group] = g.weight
        s = W.sum(1, keepdims=True)
        W = W / np.where(s > 0, s, 1)
        self.W = W
        keep = None if faces is None else set(faces)
        polys = [p for p in me.polygons if keep is None or p.index in keep]
        self.poly_index = np.array([p.index for p in polys], dtype=np.int64)
        tris, tp = [], []
        for k, p in enumerate(polys):
            vs = p.vertices
            for i in range(1, len(vs) - 1):
                tris.append((vs[0], vs[i], vs[i + 1]))
                tp.append(k)
        self.tris = np.array(tris, dtype=np.int64).reshape(-1, 3)
        self.tri_poly = np.array(tp, dtype=np.int64)          # index into self.poly_index
        a, b, c = co[self.tris[:, 0]], co[self.tris[:, 1]], co[self.tris[:, 2]]
        cr = np.cross(b - a, c - a)
        self.tri_area = 0.5 * np.linalg.norm(cr, axis=1)
        self.tri_n = cr / np.maximum(np.linalg.norm(cr, axis=1, keepdims=True), 1e-30)
        # weight of every polygon per vertex group
        Wp = np.zeros((len(polys), W.shape[1]), dtype=np.float32)
        cnt = np.zeros(len(polys))
        for j in range(3):
            np.add.at(Wp, self.tri_poly, W[self.tris[:, j]])
        np.add.at(cnt, self.tri_poly, 3)
        self.Wp = Wp / np.maximum(cnt, 1)[:, None]
        self.poly_area = np.zeros(len(polys))
        np.add.at(self.poly_area, self.tri_poly, self.tri_area)
        self.poly_mat = np.array([p.material_index for p in polys], dtype=np.int64)
        used = np.unique(self.tris)
        self.lo, self.hi = co[used].min(0), co[used].max(0)
        self.H = float(self.hi[2] - self.lo[2])
        self.diag = float(np.linalg.norm(self.hi - self.lo))
        # surface samples
        rng = np.random.default_rng(seed)
        pr = self.tri_area / max(self.tri_area.sum(), 1e-30)
        ti = rng.choice(len(self.tris), size=n_samples, p=pr)
        u, v = rng.random(n_samples), rng.random(n_samples)
        f = u + v > 1
        u[f], v[f] = 1 - u[f], 1 - v[f]
        self.pts = a[ti] + (b[ti] - a[ti]) * u[:, None] + (c[ti] - a[ti]) * v[:, None]
        self.pt_poly = self.tri_poly[ti]
        self.pt_n = self.tri_n[ti]
        self.area = float(self.tri_area.sum())

    def group_mass(self):
        """area weighted skin weight per vertex group"""
        return (self.Wp * self.poly_area[:, None]).sum(0)


# ====================================================================== free standing thin parts
_CONE = None


def _cone_dirs():
    """1 + 6 directions (local frame, z = axis): the axis and a ring 55 degrees off it"""
    global _CONE
    if _CONE is None:
        a = math.radians(55)
        _CONE = np.array([[0, 0, 1.0]] + [[math.sin(a) * math.cos(t), math.sin(a) * math.sin(t), math.cos(a)]
                                          for t in np.arange(6) * math.pi / 3 + 0.3])
    return _CONE


def core_size(sk, high, owners, own):
    """-> per polygon of `high`: the diameter of the body chain it belongs to at its place along the chain
    (2 * median distance of the chain's surface from its bone axis there); 0 for polygons of extra groups"""
    H = sk.H
    out = np.zeros(len(own))
    pt_own = own[high.pt_poly]
    cen = np.zeros((len(own), 3))
    np.add.at(cen, high.tri_poly, high.co[high.tris].mean(1) * high.tri_area[:, None])
    cen /= np.maximum(high.poly_area, 1e-30)[:, None]
    for ch in [sk.torso] + sk.legs + sk.arms:
        i = owners.index(ch.name)
        J = np.array(ch.joints)
        if len(J) < 2:
            J = np.array([J[0], J[0] + sk.up * 0.1 * H])
        seg = J[1:] - J[:-1]
        L = np.maximum(np.linalg.norm(seg, axis=1), 1e-9)
        cum = np.r_[0, np.cumsum(L)]

        def station(P):
            best_r = np.full(len(P), np.inf)
            best_s = np.zeros(len(P))
            for k in range(len(seg)):
                d = seg[k] / L[k]
                t = (P - J[k]) @ d
                lo_ = -np.inf if k == 0 else 0.0
                hi_ = np.inf if k == len(seg) - 1 else L[k]
                tc = np.clip(t, lo_, hi_)
                r = np.linalg.norm(P - J[k] - np.outer(tc, d), axis=1)
                m = r < best_r
                best_r[m], best_s[m] = r[m], cum[k] + tc[m]
            return best_s, best_r

        P = high.pts[pt_own == i]
        pk = np.flatnonzero(own == i)
        if len(P) < 50 or not len(pk):
            continue
        s, r = station(P)
        h = 0.02 * H
        b = np.floor(s / h).astype(int)
        b0 = b.min()
        nbin = b.max() - b0 + 1
        D = np.full(nbin, np.nan)
        for k in range(nbin):
            m = b - b0 == k
            if m.sum() >= 30:
                D[k] = 2 * float(np.median(r[m]))
        if np.isnan(D).all():
            continue
        D = _fill1(D)
        D = np.array([np.median(D[max(0, k - 1):k + 2]) for k in range(nbin)])
        sp, _r = station(cen[pk])
        out[pk] = D[np.clip(np.floor(sp / h).astype(int) - b0, 0, nbin - 1)]
    return out


def free_standing(high, t_thin, reach, open_min=0.4, log=print):
    """-> (free, thin) bool per polygon of `high`.  free: the polygon is part of something thin (thinner than t_thin[polygon]) that has
    open space on BOTH sides: a fin, a flange, a hat brim, a feather, a finger, a strap that stands off.  A layer
    that lies on the body (coat on the chest, a plate on the thigh) or the wall of a hollow shell (the space
    behind it is enclosed) is not.  Thin: a ray from the surface against the normal leaves the solid again within
    t_thin (or hits nothing at all: single sided card).  Open: at least open_min of 7 rays (cone around the
    direction) hit nothing within reach[polygon].  thin: only the first test.  Polygons with t_thin <= 0 are skipped."""
    H = high.H
    co, tris = high.co, high.tris
    bvh = BVHTree.FromPolygons([tuple(p) for p in co], [tuple(int(i) for i in t) for t in tris])
    eps = 1e-4 * H
    cone = _cone_dirs()
    npoly = len(high.poly_index)
    votes = np.zeros(npoly)
    tvotes = np.zeros(npoly)
    cnt = np.zeros(npoly)
    # up to two samples per polygon: the centroids of its two largest triangles
    order = np.lexsort((-high.tri_area, high.tri_poly))
    first = np.r_[True, np.diff(high.tri_poly[order]) != 0]
    second = np.r_[False, first[:-1]] & ~first
    pick = order[first | second]

    def openness(q, d, R):
        d = d / np.linalg.norm(d)
        u = np.cross(d, [0, 0, 1.0])
        if np.linalg.norm(u) < 0.1:
            u = np.cross(d, [1.0, 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(d, u)
        free = 0
        for c in cone:
            r = c[0] * u + c[1] * v + c[2] * d
            if bvh.ray_cast(Vector(q), Vector(r), R)[0] is None:
                free += 1
        return free / len(cone)

    for t in pick:
        k = high.tri_poly[t]
        if t_thin[k] <= 0:
            continue
        a, b, c = co[tris[t]]
        p = (a + b + c) / 3
        n = high.tri_n[t]
        cnt[k] += 1
        loc, hn, _i, dist = bvh.ray_cast(Vector(p - n * eps), Vector(-n), 0.3 * H)
        if loc is None:
            q = p - n * (2 * eps)                      # single sided card
        elif dist <= t_thin[k] and float(np.dot(hn, -n)) > 0.2:
            q = np.array(loc) - n * (2 * eps)          # left the thin solid through its back side
        else:
            continue
        tvotes[k] += 1
        if openness(q, -n, reach[k]) < open_min:
            continue
        if openness(p + n * (2 * eps), n, reach[k]) < open_min:
            continue
        votes[k] += 1
    return (votes >= np.maximum(cnt, 1)) & (cnt > 0), (tvotes >= np.maximum(cnt, 1)) & (cnt > 0)


def poly_clusters(high, me, mask):
    """connected sets (shared vertices) of the polygons of `high` selected by mask -> list of arrays of local indices"""
    par = {}

    def find(x):
        while par.setdefault(x, x) != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    sel = np.flatnonzero(mask)
    for k in sel:
        vs = me.polygons[int(high.poly_index[k])].vertices
        r0 = find(vs[0])
        for v in vs[1:]:
            par[find(v)] = r0
    cl = collections.defaultdict(list)
    for k in sel:
        cl[find(me.polygons[int(high.poly_index[k])].vertices[0])].append(int(k))
    return [np.array(v) for v in cl.values()]


# ====================================================================== skeleton
class Chain:
    """a body chain: joints (world points; the last one is the virtual tip), the bones that own its surface"""

    def __init__(self, name, joints, bones, side=0):
        self.name, self.joints, self.bones, self.side = name, [np.asarray(j, float) for j in joints], set(bones), side


class Skeleton:
    pass


def _bones_of(arm):
    mw = arm.matrix_world
    names = [b.name for b in arm.data.bones]
    idx = {n: i for i, n in enumerate(names)}
    parent = [idx[b.parent.name] if b.parent else -1 for b in arm.data.bones]
    head = np.array([(mw @ b.head_local)[:] for b in arm.data.bones])
    kids = [[] for _ in names]
    for i, p in enumerate(parent):
        if p >= 0:
            kids[p].append(i)
    return names, parent, kids, head


def analyse_skeleton(arm, high, exclude_bones=(), log=print):
    """-> Skeleton with .torso (Chain), .arms / .legs ([Chain L, Chain R] or []), .extra (list of bone index sets),
    .owner_of_bone {bone index: owner name}.  Raises ValueError when no humanoid is found."""
    names, parent, kids, head = _bones_of(arm)
    nb = len(names)
    H = high.H
    g2b = np.array([names.index(g) if g in names else -1 for g in high.gnames])
    mass = np.zeros(nb)
    gm = high.group_mass()
    for g, b in enumerate(g2b):
        if b >= 0:
            mass[b] += gm[g]
    # dominant bone of every polygon / sample
    Wb = np.zeros((high.Wp.shape[0], nb), dtype=np.float32)
    for g, b in enumerate(g2b):
        if b >= 0:
            Wb[:, b] += high.Wp[:, g]
    pdom = Wb.argmax(1)
    pdom[Wb.max(1) <= 0] = -1
    ptdom = pdom[high.pt_poly]
    excl = set()
    for n in exclude_bones:
        if n in names:
            st = [names.index(n)]
            while st:
                b = st.pop()
                excl.add(b)
                st.extend(kids[b])

    def subtree(b, stop=()):
        out, st = [], [b]
        while st:
            x = st.pop()
            if x in excl or x in stop:
                continue
            out.append(x)
            st.extend(kids[x])
        return out

    smass = np.zeros(nb)
    for b in range(nb):
        smass[b] = sum(mass[x] for x in subtree(b))
    alive = [b for b in range(nb) if b not in excl and smass[b] > 0]
    if not alive:
        raise ValueError('no skinned bones')
    # ---- lateral axis and mirror plane
    best = None
    for ax in (0, 1):
        c0 = float(np.median(head[alive, ax]))
        for _ in range(3):
            pair = {}
            for b in alive:
                m = head[b].copy()
                m[ax] = 2 * c0 - m[ax]
                d = np.linalg.norm(head[alive] - m, axis=1)
                j = alive[int(d.argmin())]
                if d.min() < 0.02 * H:
                    pair[b] = j
            mids = [(head[b, ax] + head[j, ax]) / 2 for b, j in pair.items() if j != b]
            if mids:
                c0 = float(np.mean(mids))
        n = sum(1 for b, j in pair.items() if j != b and abs(head[b, ax] - c0) > 0.02 * H)
        if best is None or n > best[0]:
            best = (n, ax, c0, pair)
    npairs, ax, c0, pair = best
    lat = np.zeros(3)
    lat[ax] = 1.0
    up = np.array([0.0, 0.0, 1.0])
    fwd = np.cross(up, lat)
    central = {b for b in alive if abs(head[b, ax] - c0) < 0.02 * H}
    ground = high.lo[2]

    def downlen(b):
        best_ = 0.0
        for c in kids[b]:
            if c in excl:
                continue
            best_ = max(best_, float(np.linalg.norm(head[c] - head[b])) + downlen(c))
        return best_

    def main_chain(b):
        ch = [b]
        while True:
            cur = ch[-1]
            cand = [c for c in kids[cur] if c not in excl and np.linalg.norm(head[c] - head[cur]) > 0.02 * H]
            if not cand:
                break
            if len(cand) >= 3 and len(ch) > 1:
                break
            ch.append(max(cand, key=lambda c: float(np.linalg.norm(head[c] - head[cur])) + downlen(c)))
            if len(ch) > 12:
                break
        return ch

    def tubeness(ch):
        """is the skinned surface of the chain a tube around its bones (limb) or a sheet / strand (cloth, hair)?
        -> median over the segments of (1 - largest angular gap / 360) * sqrt(lambda2 / lambda1)"""
        own = np.isin(ptdom, subtree(ch[0]))
        P = high.pts[own]
        if len(P) < 50:
            return 0.0
        sc = []
        for a, b in zip(ch[:-1], ch[1:]):
            d = head[b] - head[a]
            L = np.linalg.norm(d)
            if L < 0.03 * H:
                continue
            d = d / L
            m = (head[a] + head[b]) / 2
            q = P - m
            t = q @ d
            q = q - np.outer(t, d)
            r = np.linalg.norm(q, axis=1)
            sel = (np.abs(t) < 0.25 * L) & (r < 0.2 * H)
            if sel.sum() < 12:
                sc.append(0.0)
                continue
            q = q[sel]
            e1 = np.cross(d, [0, 0, 1.0])
            if np.linalg.norm(e1) < 0.1:
                e1 = np.cross(d, [1.0, 0, 0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(d, e1)
            x, y = q @ e1, q @ e2
            ang = np.sort(np.arctan2(y, x))
            gap = float(np.max(np.diff(np.concatenate([ang, [ang[0] + 2 * math.pi]]))))
            ev = np.linalg.eigvalsh(np.cov(np.vstack([x, y])))
            sc.append((1 - gap / (2 * math.pi)) * math.sqrt(max(ev[0], 0) / max(ev[1], 1e-30)))
        return float(np.median(sc)) if sc else 0.0

    # ---- mirrored limb candidates: off-centre bones whose parent is central (or the root)
    cands = []
    for b in alive:
        j = pair.get(b)
        if j is None or j == b or b in central or head[b, ax] < c0:
            continue
        p = parent[b]
        if p >= 0 and p not in central and pair.get(p, p) != p:
            continue
        ch, chm = main_chain(b), main_chain(j)
        n_ = min(len(ch), len(chm))          # a prop bone under one hand makes one side longer
        ch, chm = ch[:n_], chm[:n_]
        if len(ch) < 2:
            continue
        length = sum(np.linalg.norm(head[y] - head[x]) for x, y in zip(ch[:-1], ch[1:]))
        cands.append(dict(b=b, m=j, ch=ch, chm=chm, length=float(length), zmin=float(head[ch, 2].min()), tube=tubeness(ch),
                          reach=float(np.abs(head[ch, ax] - c0).max())))
    for c in cands:
        log('  limb candidate %-22s bones %d length %.2fH lowest %.2fH tube %.2f' % (
            names[c['b']], len(c['ch']), c['length'] / H, (c['zmin'] - ground) / H, c['tube']))
    legs = [c for c in cands if c['tube'] > 0.35 and c['length'] > 0.25 * H and c['zmin'] - ground < 0.15 * H]
    if not legs:
        raise ValueError('no legs found (not a humanoid skeleton?)')
    leg = max(legs, key=lambda c: (c['tube'] > 0.25, c['length']))
    arms = [c for c in cands if c is not leg and c['tube'] > 0.35 and c['length'] > 0.15 * H and head[c['b'], 2] > head[leg['b'], 2] + 0.1 * H]
    arm_c = max(arms, key=lambda c: c['length']) if arms else None
    # ---- central path: root .. head
    def anc(b):
        out = []
        while b >= 0:
            out.append(b)
            b = parent[b]
        return out
    top = parent[arm_c['b']] if arm_c else parent[leg['b']]
    if top < 0:
        raise ValueError('limbs hang on no bone')
    while True:
        nxt = [c for c in kids[top] if c in central and c not in excl and head[c, 2] > head[top, 2] + 0.01 * H
               and (min(head[x, 2] for x in subtree(c)) > head[top, 2] - 0.05 * H or mass[c] >= 0.3 * mass[top])]
        if not nxt:
            break
        top = max(nxt, key=lambda c: max(head[x, 2] for x in subtree(c)))
    path = anc(top)[::-1]
    torso_bones = set(path) | set(anc(parent[leg['b']]))
    path_pts = np.isin(ptdom, list(torso_bones))

    def half_width(z):
        sel = path_pts & (np.abs(high.pts[:, 2] - z) < 0.03 * H)
        if sel.sum() < 10:
            return 0.0
        return float(np.percentile(np.abs(high.pts[sel, ax] - c0), 90))

    sk = Skeleton()
    sk.names, sk.parent, sk.kids, sk.head = names, parent, kids, head
    sk.lat, sk.up, sk.fwd, sk.c0, sk.ax, sk.H = lat, up, fwd, c0, ax, H
    sk.Wb, sk.pdom = Wb, pdom
    limb_bones = {}

    def make_limb(c, kind):
        out = []
        for side, key in ((1, 'ch'), (-1, 'chm')):
            ch = list(c[key])
            lead = []
            if kind == 'arm':
                while len(ch) > 2:
                    mid = (head[ch[0]] + head[ch[1]]) / 2
                    if abs(mid[ax] - c0) < 0.9 * half_width(mid[2]):
                        lead.append(ch.pop(0))
                    else:
                        break
            torso_bones.update(lead)
            core = ch[:3]
            bones = set(subtree(ch[0]))
            # the end effector: everything below the last core bone
            endb = subtree(core[-1])
            E = high.pts[np.isin(ptdom, endb)]
            J = [head[b] for b in core]
            leaves = [x for x in endb if not [k for k in kids[x] if k not in excl] and x != core[-1]]
            if leaves:
                d = head[leaves].mean(0) - J[-1]
            elif len(E) > 20:
                d = E.mean(0) - J[-1]
            else:
                d = J[-1] - head[parent[core[-1]]]
            d = d / max(np.linalg.norm(d), 1e-12)
            ext = float(np.percentile((E - J[-1]) @ d, 99)) if len(E) > 20 else 0.05 * H
            J.append(J[-1] + d * max(ext, 0.03 * H))
            ln = '%s_%s' % (kind, 'L' if side > 0 else 'R')
            lc = Chain(ln, J, bones, side)
            lc.core = core
            lc.kind = kind
            out.append(lc)
            limb_bones[ln] = bones
        return out

    sk.legs = make_limb(leg, 'leg')
    sk.arms = make_limb(arm_c, 'arm') if arm_c else []
    limbs = sk.legs + sk.arms
    for lc in limbs:
        torso_bones -= lc.bones
    # helper bones lying on a limb axis (twist bones that hang on the clavicle), fingers etc. are in the subtree already
    owner = {}
    for b in torso_bones:
        owner[b] = 'torso'
    for lc in limbs:
        for b in lc.bones:
            owner[b] = lc.name

    def seg_dist(p, a, b):
        d = b - a
        t = np.clip((p - a) @ d / max(d @ d, 1e-30), 0, 1)
        return float(np.linalg.norm(p - (a + t * d)))

    for b in alive:
        if b in owner or mass[b] <= 0:
            continue
        bestd, bestn = 0.02 * H, None
        for lc in limbs:
            for a, c_ in zip(lc.joints[:-1], lc.joints[1:]):
                d = seg_dist(head[b], a, c_)
                if d < bestd:
                    bestd, bestn = d, lc.name
        if bestn:
            owner[b] = bestn
            next(l for l in limbs if l.name == bestn).bones.add(b)
    # central bones under the torso that carry no surface of their own shape (root, pelvis): torso
    # ---- torso chain: straight axis from the lowest to the highest torso joint
    tb = sorted(torso_bones, key=lambda b: head[b, 2])
    sk.torso = Chain('torso', [head[b] for b in path if b in torso_bones], torso_bones, 0)
    sk.torso.core = path
    sk.torso.kind = 'torso'
    # ---- extra bones: groups of bones (not body) that share polygons
    extra = [b for b in alive if b not in owner and mass[b] > 0]
    par = {b: b for b in extra}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    if extra:
        E = np.array(extra)
        sub = Wb[:, E] > 0.15
        rows = np.where(sub.sum(1) > 1)[0]
        for r in rows:
            bs = E[sub[r]]
            for x in bs[1:]:
                ra, rb = find(int(bs[0])), find(int(x))
                if ra != rb:
                    par[ra] = rb
    groups = collections.defaultdict(set)
    for b in extra:
        groups[find(b)].add(b)
    sk.extra = []
    for k, bs in enumerate(sorted(groups.values(), key=lambda s: -sum(mass[b] for b in s))):
        # the body chain it hangs on: owner of the nearest classified ancestor
        host = collections.Counter()
        for b in bs:
            p = parent[b]
            while p >= 0 and p not in owner:
                p = parent[p]
            host[owner.get(p, 'torso')] += mass[b]
        sk.extra.append(dict(name='extra%d' % k, bones=bs, host=host.most_common(1)[0][0], mass=float(sum(mass[b] for b in bs))))
        for b in bs:
            owner[b] = 'extra%d' % k
    sk.owner = owner
    sk.mass = mass
    log('  skeleton: lateral axis %s, mirror at %.3f, %d mirrored bones; torso %s' % ('XY'[ax], c0, npairs, [names[b] for b in path]))
    for lc in limbs:
        log('  %s: %s (+%d bones), tip %.2fH from the last joint' % (lc.name, [names[b] for b in lc.core], len(lc.bones) - len(lc.core),
                                                              np.linalg.norm(lc.joints[-1] - lc.joints[-2]) / H))
    for e in sk.extra:
        log('  %s: %d bones (%s ...) on %s, weight mass %.3f' % (e['name'], len(e['bones']), names[min(e['bones'])], e['host'], e['mass']))
    return sk


def face_owners(sk, high):
    """owner name of every polygon of `high` (argmax of the summed weights per owner)"""
    owners = ['torso'] + [l.name for l in sk.legs + sk.arms] + [e['name'] for e in sk.extra]
    oi = {n: i for i, n in enumerate(owners)}
    M = np.zeros((sk.Wb.shape[0], len(owners)), dtype=np.float32)
    for b, o in sk.owner.items():
        M[:, oi[o]] += sk.Wb[:, b]
    own = M.argmax(1)
    own[M.max(1) <= 0] = 0
    return owners, own


# ====================================================================== ring fitting
def _frame(n, ref):
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    u = np.asarray(ref, float) - n * float(np.dot(ref, n))
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(n, [1.0, 0, 0])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(n, [0, 1.0, 0])
    u = u / np.linalg.norm(u)
    return n, u, np.cross(n, u)


def _circ_fill(r):
    """nan entries of a circular array from their neighbours"""
    ok = ~np.isnan(r)
    if ok.all() or not ok.any():
        return r
    n = len(r)
    i = np.arange(n)
    x = np.concatenate([i[ok] - n, i[ok], i[ok] + n])
    y = np.tile(r[ok], 3)
    return np.interp(i, x, y)


def _circ_filter(r, w, fn):
    out = r.copy()
    for s in range(1, w + 1):
        out = fn(out, np.roll(r, s))
        out = fn(out, np.roll(r, -s))
    return out


class Ring:
    __slots__ = ('P', 'n', 'u', 'v', 'C', 'rb', 'cover', 'verts', 'r', 'tag')


def radial_profile(P, n, u, pts, H, nb=48, slab=0.012, rmax=0.3, lock_u=False, sym=False, spike=2, pct=92, fixed=False, gap=0.07):
    """outline of the surface samples `pts` in the plane (P, n): radial function over nb angular bins around
    the centre of the outline.  spikes narrower than 2*spike bins are removed (fins, buckles).
    lock_u: the centre stays on the plane u = 0 (mirror plane); sym: outline mirrored in u.
    -> Ring (without vertices) or None"""
    n, u, v = _frame(n, u)
    q = pts - P
    d = q @ n
    sel = np.abs(d) < slab * H
    if sel.sum() < 12:
        return None
    q = q[sel]
    x, y = q @ u, q @ v
    near = np.hypot(x, y) < rmax * H
    x, y = x[near], y[near]
    if len(x) < 12:
        return None
    if sym:
        x, y = np.concatenate([x, -x]), np.concatenate([y, y])
    cx, cy = (0.0, 0.0) if fixed else ((0.0 if lock_u else float(np.median(x))), float(np.median(y)))
    rb = None
    for it in range(3):
        ang = np.arctan2(y - cy, x - cx)
        r = np.hypot(x - cx, y - cy)
        b = np.minimum(((ang + math.pi) / (2 * math.pi) * nb).astype(int), nb - 1)
        rb = np.full(nb, np.nan)
        order = np.argsort(b, kind='stable')
        bs, rs = b[order], r[order]
        cuts = np.searchsorted(bs, np.arange(nb + 1))
        for k in range(nb):
            if cuts[k + 1] > cuts[k]:
                seg = np.sort(rs[cuts[k]:cuts[k + 1]])
                # what lies beyond a wide empty gap is something else (a sword on the back, the other hand)
                far = np.flatnonzero(np.diff(seg) > gap * H)
                if len(far):
                    seg = seg[:far[0] + 1]
                rb[k] = np.percentile(seg, pct)
        cover = float((~np.isnan(rb)).mean())
        if cover < 0.2:
            return None
        rb = _circ_fill(rb)
        if spike:
            rb = _circ_filter(_circ_filter(rb, spike, np.minimum), spike, np.maximum)
        th = -math.pi + (np.arange(nb) + 0.5) * 2 * math.pi / nb
        ox, oy = cx + rb * np.cos(th), cy + rb * np.sin(th)
        # centroid of the outline polygon
        x1, y1 = np.roll(ox, -1), np.roll(oy, -1)
        cr = ox * y1 - x1 * oy
        A = cr.sum() / 2
        if abs(A) < 1e-12:
            break
        nx, ny = ((ox + x1) * cr).sum() / (6 * A), ((oy + y1) * cr).sum() / (6 * A)
        if lock_u:
            nx = 0.0
        if it < 2 and not fixed:
            cx, cy = nx, ny
    R = Ring()
    R.P, R.n, R.u, R.v = P, n, u, v
    R.C = P + cx * u + cy * v
    R.rb, R.cover = rb, cover
    R.verts = None
    R.r = float(np.mean(rb))
    return R


def ring_verts(R, N, phase=0.0, keep_area=True):
    """N vertices on the outline, vertex k at angle phase + 2 pi k / N from u"""
    nb = len(R.rb)
    th = -math.pi + (np.arange(nb) + 0.5) * 2 * math.pi / nb
    out = np.zeros((N, 3))
    rs = np.zeros(N)
    half = math.pi / N * 0.5
    for k in range(N):
        a = phase + 2 * math.pi * k / N
        da = np.abs((th - a + math.pi) % (2 * math.pi) - math.pi)
        m = da <= max(half, math.pi / nb)
        rs[k] = float(R.rb[m].mean()) if m.any() else float(R.rb[int(da.argmin())])
    if keep_area:
        a_out = 0.5 * float((R.rb * np.roll(R.rb, -1)).sum()) * math.sin(2 * math.pi / nb)
        a_pol = 0.5 * float((rs * np.roll(rs, -1)).sum()) * math.sin(2 * math.pi / N)
        if a_pol > 1e-12:
            rs = rs * min(max(math.sqrt(a_out / a_pol), 1.0), 1.15)
    for k in range(N):
        a = phase + 2 * math.pi * k / N
        out[k] = R.C + rs[k] * (math.cos(a) * R.u + math.sin(a) * R.v)
    R.verts = out
    return out


# ====================================================================== mesh builder
class Builder:
    def __init__(self):
        self.v = []
        self.f = []
        self.fg = []          # bake / skin group of every face
        self.seams = set()
        self.keep_winding = set()      # groups whose faces are built with the right orientation (open sheets)
        self.join_tris = set()         # groups whose triangles are joined to quads where two of them are nearly flat
        self.join_shape = 35.0         # degrees: how far from a rectangle such a quad may be

    def add(self, pts):
        i0 = len(self.v)
        self.v.extend([np.asarray(p, float) for p in pts])
        return list(range(i0, i0 + len(pts)))

    def face(self, vs, g):
        vs = list(vs)
        if len(set(vs)) < 3:
            return
        self.f.append(vs)
        self.fg.append(g)

    def seam(self, a, b):
        self.seams.add((min(a, b), max(a, b)))

    def strip(self, A, B, g, closed=True):
        n = len(A)
        for k in range(n if closed else n - 1):
            k1 = (k + 1) % n
            self.face([A[k], A[k1], B[k1], B[k]], g)

    def bridge(self, A, B, g, reverse=True):
        """quads between two closed loops of the same length: B is rotated / reversed (reverse=False: only
        rotated, both loops run the same way round) to the nearest fit.
        -> B reordered so that B[k] is connected to A[k]"""
        n = len(A)
        pa = np.array([self.v[i] for i in A])
        best = None
        for rev in ((False, True) if reverse else (False,)):
            Bx = B[::-1] if rev else B
            pb = np.array([self.v[i] for i in Bx])
            for s in range(n):
                c = float(((pa - np.roll(pb, -s, axis=0)) ** 2).sum())
                if best is None or c < best[0]:
                    best = (c, [Bx[(k + s) % n] for k in range(n)])
        B2 = best[1]
        self.strip(A, B2, g)
        return B2

    def cap(self, R, g):
        """closes a ring with a ladder of quads; the rungs run across the narrow side of the ring"""
        n = len(R)
        if n == 3:
            self.face(list(R), g)
            for k in range(3):
                self.seam(R[k], R[(k + 1) % 3])
            return
        h = n // 2
        P = np.array([self.v[i] for i in R])
        best = None
        for s in range(h):
            a = [(s + j) % n for j in range(h)]
            b = [(s - 1 - j) % n for j in range(h)]
            c = sum(float(np.linalg.norm(P[x] - P[y])) for x, y in zip(a[1:-1], b[1:-1])) if h > 2 else 0.0
            if best is None or c < best[0]:
                best = (c, a, b)
        a, b = [R[i] for i in best[1]], [R[i] for i in best[2]]
        for j in range(h - 1):
            self.face([a[j], a[j + 1], b[j + 1], b[j]], g)
        for k in range(n):
            self.seam(R[k], R[(k + 1) % n])


def support_ring(P, n, u, pts, H, N, slab=0.008, rmax=0.2, top=3.0):
    """round 9 (boots): ring whose vertex k is the support point of the section of `pts` with the plane (P, n) in
    the direction at angle 2 pi k / N from u (mean of the `top` percent of the samples that reach furthest that
    way).  Unlike the radial outline this keeps corners: the two edges of a flat sole, a heel.
    -> Ring with vertices, or None"""
    n, u, v = _frame(n, u)
    q = pts - P
    sel = np.abs(q @ n) < slab * H
    if sel.sum() < 30:
        return None
    q = q[sel]
    x, y = q @ u, q @ v
    near = np.hypot(x - np.median(x), y - np.median(y)) < rmax * H
    x, y = x[near], y[near]
    if len(x) < 30:
        return None
    out = np.zeros((N, 3))
    xy = np.zeros((N, 2))
    for k in range(N):
        a = 2 * math.pi * k / N
        sup = x * math.cos(a) + y * math.sin(a)
        m = sup >= np.percentile(sup, 100.0 - top)
        xy[k] = (float(x[m].mean()), float(y[m].mean()))
        out[k] = P + xy[k, 0] * u + xy[k, 1] * v
    R = Ring()
    R.P, R.n, R.u, R.v = P, n, u, v
    cx, cy = xy.mean(0)
    R.C = P + cx * u + cy * v
    R.rb = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    R.cover = 1.0
    R.r = float(R.rb.mean())
    R.verts = out
    return R


def _boot_plan(pts, A, fwd, up, lat, tip, H):
    """side profile of a foot that stands on the ground (round 9: "the shoes are badly deformed").
    A: ankle joint, fwd: horizontal direction of the foot, tip: length from the ankle to the toe.
    -> dict(g ground, Q pivot in front of the instep crease, K lowest point of the back of the heel) or None.
    The rings of the boot fan around the lateral axis through Q: shaft (across the shin) -> heel (through K) ->
    vertical slices of the shoe up to the toe."""
    s = (pts - A) @ fwd
    z = pts @ up
    g = float(np.percentile(z, 0.3))
    za = float(A @ up)
    if za - g < 0.02 * H or tip < 0.04 * H:
        return None
    low = z < g + 0.012 * H
    if low.sum() < 30:
        return None
    sK = float(np.percentile(s[low], 1.0))
    # front outline f(z) of shin + instep
    zt = za + 0.1 * H
    zz = np.linspace(g + 0.004 * H, zt, 40)
    prof = []
    for a_, b_ in zip(zz[:-1], zz[1:]):
        m = (z >= a_) & (z < b_)
        if m.sum() > 8:
            prof.append((float(np.percentile(s[m], 98)), 0.5 * (a_ + b_)))
    if len(prof) < 10:
        return None
    prof = np.array(prof)
    mt = (s > 0.6 * tip) & (s < 0.85 * tip)
    if mt.sum() < 10:
        return None
    p_toe = np.array([0.72 * tip, float(np.percentile(z[mt], 97))])
    p_top = prof[-1]
    ch = p_top - p_toe
    nrm = np.array([ch[1], -ch[0]])
    nrm = nrm / max(np.linalg.norm(nrm), 1e-12)
    if nrm[0] < 0:
        nrm = -nrm                      # forward / up: the open side of the bend
    sel = prof[(prof[:, 1] > p_toe[1]) & (prof[:, 0] < p_toe[0])]
    if len(sel) < 3:
        return None
    dist = (sel - p_toe) @ nrm
    I = sel[int(np.argmin(dist))]       # the crease: deepest point of the outline behind the chord
    if dist.min() > -0.004 * H:
        return None
    Q2 = I + nrm * BOOT_PIVOT * H
    lat0 = float(np.median(pts[(z > za - 0.01 * H) & (z < za + 0.03 * H)] @ lat)) if ((z > za - 0.01 * H) & (z < za + 0.03 * H)).sum() > 10 else float(A @ lat)
    def P3(p2):
        return A + fwd * p2[0] + up * (p2[1] - za) + lat * (lat0 - float(A @ lat))
    return dict(g=g, Q=P3(Q2), K=P3(np.array([sK, g])), sQ=float(Q2[0]), I=P3(I))


def _limb_rings(lc, pts, sk, N, par, log=print):
    """rings of one limb, from the first full ring below the shoulder / hip to the tip.
    -> list of Ring (vertices set), every ring has .tag"""
    H = sk.H
    for flat_foot, boot in ((True, True), (True, False), (False, False)):
        rings = _limb_rings_1(lc, pts, sk, N, par, flat_foot, log, boot=boot and BOOT)
        if rings is not None:
            return rings


def _limb_rings_1(lc, pts, sk, N, par, flat_foot, log=print, boot=True):
    H = sk.H
    J = list(lc.joints)
    m = len(J) - 1
    d = [(J[i + 1] - J[i]) / max(np.linalg.norm(J[i + 1] - J[i]), 1e-12) for i in range(m)]
    foot_pts = None
    if lc.kind == 'leg' and m >= 2 and flat_foot:
        # a foot that stands on the ground is swept horizontally: its rings are vertical slices of the shoe
        # (flat sole), the ring at the ankle (bisector of shin and foot) reaches down to the heel
        dh = d[-1] - sk.up * float(d[-1] @ sk.up)
        if J[-2][2] - float(np.percentile(pts[:, 2], 0.5)) < 0.12 * H and np.linalg.norm(dh) > 0.5:
            d[-1] = dh / np.linalg.norm(dh)
            J[-1] = J[-2] + d[-1] * float(np.linalg.norm(J[-1] - J[-2]))
            foot_pts = pts[pts[:, 2] < J[-2][2] + 0.015 * H]          # a shin that leans forward is not part of the slices
    # the tip: where the surface that belongs to the cage ends (fingers that are modelled on their own do not count)
    q = (pts if foot_pts is None else foot_pts) - J[-2]
    t = q @ d[-1]
    near = np.linalg.norm(q - np.outer(t, d[-1]), axis=1) < 0.15 * H
    if near.sum() > 20:
        ext = float(np.percentile(t[near], 99.5))
        J[-1] = J[-2] + d[-1] * float(np.clip(ext, 0.03 * H, np.linalg.norm(J[-1] - J[-2])))
    L = [float(np.linalg.norm(J[i + 1] - J[i])) for i in range(m)]
    if DEBUG.get('limb'):
        log('   limb', lc.name, 'J', np.round(np.array(J) / H, 3).tolist(), 'd', np.round(np.array(d), 2).tolist())
    ref = sk.up if lc.kind == 'arm' else sk.lat * (1 if lc.side >= 0 else 1)

    def prof(P, n):
        return radial_profile(P, n, ref, pts, H, rmax=0.22)

    st = []          # (point, normal, tag)
    # first full ring
    t0 = None
    for t in np.linspace(0.04, 0.5, 14):
        R = prof(J[0] + d[0] * (t * L[0]), d[0])
        if R is not None and R.cover >= 0.9:
            t0 = t
            break
    if t0 is None:
        t0 = 0.3
    t0 = min(t0 + 0.08, 0.5)
    rj = []
    for i in range(1, m):
        R = prof(J[i], d[i - 1] + d[i])
        rj.append(R.r if R is not None else 0.04 * H)
    st.append((J[0] + d[0] * (t0 * L[0]), d[0], 'start'))
    for i in range(m - 1):          # segments between real joints
        a = t0 * L[0] if i == 0 else min(0.9 * rj[i - 1], 0.3 * L[i])
        j = i + 1                   # joint at the end of this segment
        last_real = j == m - 1      # wrist / ankle: one loop;  elbow / knee: three
        dl = min(0.9 * rj[j - 1], 0.3 * L[i], 0.3 * L[j])
        b = L[i] - (dl if not last_real else 0.0)
        if i > 0:
            st.append((J[i] + d[i] * a, d[i], 'joint+'))
        nm = par['mids']
        for q in range(nm):
            s = a + (b - a) * (q + 1) / (nm + 1)
            st.append((J[i] + d[i] * s, d[i], 'mid'))
        if not last_real:
            st.append((J[i] + d[i] * b, d[i], 'joint-'))
        st.append((J[j], d[i] + d[j], 'joint' if not last_real else 'end_joint'))
    # end effector (or, with a single real joint, the whole limb)
    i = m - 1
    ne = par['end_rings']
    if m == 1:
        fr = np.linspace(0.0, 0.95, ne + 3)[1:]
    else:
        fr = [0.45, 0.88] if ne <= 2 else [0.33, 0.64, 0.92]
    if foot_pts is not None:
        # the first slice lies in front of the ankle ring (which leans 45 degrees forward); a foot too short for that
        # is swept along its bone as before
        t1 = (0.75 * rj[-1] + 0.01 * H) / L[i]
        if t1 > 0.6:
            return None
        if t1 > fr[0]:
            fr = list(np.linspace(t1, 0.92, len(fr)))
    use_boot, boot = boot, None
    if foot_pts is not None and use_boot:
        boot = _boot_plan(pts[pts[:, 2] < J[-2][2] + 0.13 * H], J[-2], d[-1], sk.up, sk.lat, L[-1], H)
    if boot is not None and (L[-1] - boot['sQ'] < 0.045 * H or L[-1] < 2.5 * rj[-1]):
        boot = None                         # a stub of a foot (chibi): swept as before
    if boot is not None:
        # the ankle ring and the slices of the foot are replaced by the fan: shaft, heel, instep, slices, toe
        assert st[-1][2] == 'end_joint'
        st.pop()
        Q, K = boot['Q'], boot['K']
        n1 = d[-2] - sk.lat * float(d[-2] @ sk.lat)
        n1 = n1 / np.linalg.norm(n1)
        w_ = K - Q
        w_ = w_ - sk.lat * float(w_ @ sk.lat)
        n2 = np.cross(w_, sk.lat)
        n2 = n2 / np.linalg.norm(n2)
        if n2 @ (d[-1] - sk.up) < 0:
            n2 = -n2
        st.append((Q - n1 * 0.004 * H, n1, 'boot_shaft'))
        st.append((Q, n2, 'boot_heel'))
        st.append((Q + d[-1] * 0.004 * H, d[-1], 'boot'))
        rest = L[-1] - boot['sQ']
        for t in ([0.5, 0.95] if ne <= 2 else BOOT_SLICES):
            st.append((Q + d[-1] * (t * rest), d[-1], 'boot'))
        if DEBUG.get('limb'):
            log('   boot: ground %.3fH, crease %s, pivot %s, heel %s, heel plane %s' % (boot['g'] / H, np.round(boot['I'] / H, 3), np.round(Q / H, 3), np.round(K / H, 3), np.round(n2, 2)))
    else:
        for q, t in enumerate(fr):
            st.append((J[i] + d[i] * (t * L[i]), d[i], 'end'))
    rings = []
    u = None
    for P, n, tag in st:
        n = n / np.linalg.norm(n)
        if u is None:
            u = _frame(n, ref)[1]
        else:
            u = _frame(n, u)[1]
        if tag.startswith('boot'):
            bp = pts[pts[:, 2] < boot['Q'][2] + (0.1 if tag == 'boot_shaft' else 0.03) * H]
            R = support_ring(P, n, u, bp, H, N)
            if R is None:
                R = support_ring(P, n, u, bp, H, N, slab=0.016)
        else:
            R = radial_profile(P, n, u, foot_pts if tag == 'end' and foot_pts is not None else pts, H, rmax=0.22)
        if R is None:
            R = Ring()
            R.P, R.n, R.u, R.v = P, n, u, np.cross(n, u)
            R.C, R.rb, R.cover, R.r = P, None, 0.0, 0.0
        R.tag = tag
        rings.append(R)
    if DEBUG.get('limb'):
        for R in rings:
            log('    ring', R.tag, 'P', np.round(R.P / H, 3).tolist(), 'n', np.round(R.n, 2).tolist(), 'cover', round(R.cover, 2), 'r', round(R.r / H, 4), 'C', np.round(R.C / H, 3).tolist())
    # rings without a usable outline: radius from the neighbours
    good = [k for k, R in enumerate(rings) if R.rb is not None and R.cover >= 0.5]
    if not good:
        raise ValueError('no surface around ' + lc.name)
    for k, R in enumerate(rings):
        if k in good:
            continue
        j = min(good, key=lambda g: abs(g - k))
        R.rb = rings[j].rb * (0.8 if k == len(rings) - 1 else 1.0)
        R.r = float(R.rb.mean())
    for R in rings:
        if R.tag.startswith('boot') and getattr(R, 'verts', None) is not None and R.cover == 1.0 and len(R.verts) == N:
            continue
        ring_verts(R, N)
    if boot is not None:
        for R in rings:                     # the sole is flat across every slice (a rocker sole made a notch)
            if R.tag == 'boot' and getattr(R, 'verts', None) is not None:
                zz_ = R.verts[:, 2]
                lowm = zz_ < zz_.min() + 0.3 * (zz_.max() - zz_.min())
                R.verts[lowm, 2] = zz_[lowm].min()
    if boot is not None and len(rings) >= 2 and rings[-1].tag == 'boot' and rings[-2].tag == 'boot' and getattr(rings[-2], 'verts', None) is not None:
        # the toe: the slice before it, moved to the tip and scaled to the width / height of the shoe there (about the
        # middle of the sole) - the few samples of the tip give no reliable ring of their own
        a, b = rings[-2], rings[-1]
        fd, upv = d[-1], sk.up
        side = np.cross(upv, fd)
        g0 = float(np.percentile(pts[:, 2], 0.3))
        def ext_(P0):
            q_ = pts[(np.abs((pts - P0) @ fd) < 0.008 * H) & (pts[:, 2] < boot['Q'][2] + 0.03 * H)]
            q_ = q_[np.abs((q_ - a.verts.mean(0)) @ side) < 0.2 * H]
            if len(q_) < 12:
                return None
            w_ = q_ @ side
            return float(np.percentile(w_, 97) - np.percentile(w_, 3)), float(np.percentile(q_[:, 2], 97)) - g0
        ea, eb = ext_(a.P), ext_(b.P)
        sw, sh_ = (0.7, 0.8) if ea is None or eb is None else (float(np.clip(eb[0] / max(ea[0], 1e-9), BOOT_TOE[0], 1.0)), float(np.clip(eb[1] / max(ea[1], 1e-9), BOOT_TOE[1], 1.0)))
        if os.environ.get('SR_DBG'):
            log('   toe ring: width / height of the slice before it %s, at the toe %s -> scale %.2f / %.2f' % (ea, eb, sw, sh_))
        ca = a.verts.mean(0)
        o_ = ca + upv * (g0 - float(ca @ upv))
        rel = a.verts - o_
        b.verts = o_ + fd * float((b.P - a.P) @ fd) + np.outer(rel @ side, side) * sw + np.outer(rel @ upv, upv) * sh_ + np.outer(rel @ fd, fd)
        # round 9c: round in plan view - the sides of the toe ring lie further back than its middle
        ls_ = (b.verts - b.verts.mean(0)) @ side
        hw_ = max(float(np.abs(ls_).max()), 1e-9)
        b.verts = b.verts - np.outer((0.15 if BOOT_NOSE else 0.45) * hw_ * (ls_ / hw_) ** 2, fd)
        b.C, b.cover = b.verts.mean(0), 1.0
        b.rb = np.linalg.norm(b.verts - b.C, axis=1)
        b.r = float(b.rb.mean())
        if BOOT_NOSE:
            # round 9d ("the toe tip is a flat, blunt box face"): one more small ring in front of the toe ring - the toe
            # tapers on in width and height (the sole stays on the ground), the flat end is only its small cap
            cb_ = b.verts.mean(0)
            ob_ = cb_ + upv * (g0 - float(cb_ @ upv))
            relb = b.verts - ob_
            adv = max(float((J[-2] + fd * L[-1] - b.P) @ fd), 0.004 * H) + 0.2 * float(np.abs(relb @ side).max())
            nz = Ring()
            nz.verts = ob_ + fd * adv + np.outer(relb @ side, side) * BOOT_NOSE[0] + np.outer(relb @ upv, upv) * BOOT_NOSE[1] + np.outer(relb @ fd, fd) * 0.5
            nz.P, nz.n, nz.u, nz.v = b.P + fd * adv, b.n, b.u, b.v
            nz.C, nz.cover, nz.tag = nz.verts.mean(0), 1.0, 'boot'
            nz.rb = np.linalg.norm(nz.verts - nz.C, axis=1)
            nz.r = float(nz.rb.mean())
            rings.append(nz)
    if boot is not None:
        # a fan that folds over itself (a foot that is no boot shape): swept as before
        bad = 0.0
        for a, b in zip(rings[:-1], rings[1:]):
            if b.tag.startswith('boot'):
                bad = max(bad, float((((b.verts - a.verts) @ a.n) < -0.001 * H).mean()))
                ca, cb = a.verts.mean(0), b.verts.mean(0)
                if float((cb - ca) @ a.n) <= 0:
                    bad = 1.0
        log('  %s: boot fan around the instep (crease %.3fH above the ground, heel %.3fH behind the ankle, foot %.3fH)%s' % (
            lc.name, (boot['I'][2] - boot['g']) / H, -float((boot['K'] - J[-2]) @ d[-1]) / H, L[-1] / H, ' -> folds over itself, not used' if bad > 0.2 else ''))
        if bad > 0.2:
            return None
    # no ring may cross the one before it
    if foot_pts is not None:
        g = float(np.percentile(pts[:, 2], 0.3))
        for R in rings:
            if R.tag in ('end', 'end_joint'):
                R.verts[:, 2] = np.maximum(R.verts[:, 2], g)          # nothing below the sole
            elif R.tag in ('boot', 'boot_heel'):
                R.verts[:, 2] = np.where(R.verts[:, 2] < g + 0.009 * H, g, R.verts[:, 2])      # flat sole
    for a, b in zip(rings[:-1], rings[1:]):
        dd = (b.verts - a.verts) @ a.n
        for k in np.where(dd < 0.002 * H)[0]:
            b.verts[k] += a.n * (0.002 * H - dd[k])
    return rings


def build_body(sk, high, owners, own, par, log=print):
    """own: owner index per polygon (-1: not part of the cage).  par: one of BODY_TIERS.
    -> Builder with the closed body cage, info dict"""
    H = sk.H
    Nt = par['Nt']
    hc, hr = par['hole_cols'], par['hole_rows']
    Na = 2 * (hc + hr)
    Nl = Nt // 2 + par['crotch']
    pt_own = own[high.pt_poly]
    oi = {n: i for i, n in enumerate(owners)}
    P_of = {n: high.pts[pt_own == i] for n, i in oi.items()}

    def mirror(p):
        q = np.array(p, float)
        q[..., sk.ax] = 2 * sk.c0 - q[..., sk.ax]
        return q

    B = Builder()
    info = {'params': par, 'arm_sides': Na if sk.arms else 0, 'leg_sides': Nl, 'torso_sides': Nt}
    # ---- limbs: fitted on the left one with the samples of both sides, mirrored to the right
    limb_rings = {}
    for pair in (sk.legs, sk.arms):
        if not pair:
            continue
        Lc, Rc = pair
        N = Nl if Lc.kind == 'leg' else Na
        symm = len(Lc.joints) == len(Rc.joints) and max(np.linalg.norm(mirror(a) - b) for a, b in zip(Lc.joints, Rc.joints)) < 0.03 * H
        if symm:
            pts = np.concatenate([P_of[Lc.name], mirror(P_of[Rc.name])])
            rl = _limb_rings(Lc, pts, sk, N, par, log)
            limb_rings[Lc.name] = [r.verts for r in rl]
            limb_rings[Rc.name] = [mirror(r.verts) for r in rl]
            tags = [r.tag for r in rl]
        else:
            for c in (Lc, Rc):
                rl = _limb_rings(c, P_of[c.name], sk, N, par, log)
                limb_rings[c.name] = [r.verts for r in rl]
                tags = [r.tag for r in rl]
        info[Lc.kind + '_rings'] = tags
    # ---- torso: horizontal rings, symmetric
    tor = P_of['torso']
    legp = np.concatenate([P_of[c.name] for c in sk.legs])
    low_pts = np.concatenate([tor, legp])
    lat, up, fwd = sk.lat, sk.up, sk.fwd
    base = np.zeros(3)
    base[sk.ax] = sk.c0
    zhip = float(np.mean([c.joints[0][2] for c in sk.legs]))
    Lth = float(np.linalg.norm(sk.legs[0].joints[1] - sk.legs[0].joints[0]))
    nearp = low_pts[np.abs(low_pts[:, sk.ax] - sk.c0) < 0.012 * H]
    nearp = nearp[nearp[:, 2] < zhip + 0.02 * H]
    zcr = float(np.percentile(nearp[:, 2], 2)) if len(nearp) > 20 else zhip - 0.25 * Lth
    zcr = max(zcr, zhip - 0.6 * Lth)
    # the legs start below the crotch
    ztop_leg = max(float(v[:, 2].max()) for v in (limb_rings[c.name][0] for c in sk.legs))
    z0 = max(zcr + 0.012 * H, ztop_leg + 0.02 * H)
    ztop = float(np.percentile(tor[:, 2], 99.7))
    zs_m = [z0]
    if sk.arms:
        a0 = limb_rings[sk.arms[0].name][0]
        za, zb = float(a0[:, 2].min()), float(a0[:, 2].max())
        zc = (za + zb) / 2
        hgt = max(zb - za, 0.05 * H) * 1.35          # the loop around the shoulder is larger than the arm
        za, zb = zc - hgt / 2, zc + hgt / 2
        rows = [za + (zb - za) * k / hr for k in range(hr + 1)]
        zs_m += rows
        sh_off = abs(float(sk.arms[0].joints[0][sk.ax]) - sk.c0)
    zs_m.append(ztop - 0.012 * H)
    z_sh = zs_m[-2] + 0.03 * H if sk.arms else ztop          # above the shoulders: head, hat

    def tprof(z, cy=None):
        P = base + up * z
        if cy is not None:
            return radial_profile(P + fwd * cy, up, lat, low_pts if z < zhip + 0.06 * H else tor, H, rmax=0.6, lock_u=True, sym=True,
                                  spike=3 if z < z_sh else 0, fixed=True)
        near_ = np.abs(tor[:, 2] - z) < 0.03 * H
        P = P + fwd * (float(np.median(tor[near_] @ fwd)) if near_.any() else 0.0)
        return radial_profile(P, up, lat, low_pts if z < zhip + 0.06 * H else tor, H, rmax=0.6, lock_u=True, sym=True, spike=3)

    # round 9 ("the hair is painted on the cape, it should hang over it"): the hair is part of the head volume down to
    # z_att; the ring right below is fitted WITHOUT the hair (a step), the cape ends under the step and the rest of
    # the hair hangs from the rim of the step over the cape (add_hair_flap, after the cape is built)
    z_hb = None
    hp = getattr(sk, 'hair_polys', None)
    if HAIR_STEP and hp is not None and hp.any():
        front_sign0 = 1.0 if float((sk.legs[0].joints[-1] - sk.legs[0].joints[-2]) @ fwd) > 0 else -1.0
        hm = hp[high.pt_poly] & (pt_own == oi['torso'])
        Ph = high.pts[hm]
        if len(Ph) > 200:
            back = Ph[(Ph @ fwd) * front_sign0 < float(np.median(tor @ fwd)) * front_sign0]
            if len(back) > 100:
                z_hb = float(np.percentile(back[:, 2], HAIR_PCT if getattr(sk, 'cape_to_collar', False) else HAIR_PCT0))
                tor_nh = high.pts[(pt_own == oi['torso']) & ~hp[high.pt_poly]]
    dense = {}
    zz = sorted(set(list(np.arange(z0, ztop - 0.012 * H, 0.008 * H)) + zs_m))
    cen = {}
    for z in zz:
        R = tprof(float(z))
        if R is not None and R.cover > 0.5:
            cen[float(z)] = float(R.C @ fwd)
    ck = sorted(cen)
    for z in ck:           # centre line smoothed along the height: no zigzag between neighbouring rings
        w = np.array([math.exp(-((k - z) / (0.05 * H)) ** 2) for k in ck])
        cy = float((w * np.array([cen[k] for k in ck])).sum() / w.sum())
        R = tprof(z, cy)
        if R is not None and R.cover > 0.5:
            ring_verts(R, Nt)
            dense[z] = R
    keys = sorted(dense)
    # below the shoulders a ring that is much wider than the rings just above and below it caught something thin
    # and flat (the rim of a coat skirt, a belt plate): median of the outline over +-2% of the height
    raw_rb = {k: dense[k].rb.copy() for k in keys}
    for k in keys:
        if k >= z_sh:
            continue
        nb_ = [raw_rb[j] for j in keys if abs(j - k) <= 0.02 * H and j < z_sh]
        if len(nb_) >= 3:
            dense[k].rb = np.minimum(dense[k].rb, np.median(np.array(nb_), axis=0) * 1.05)
            ring_verts(dense[k], Nt)
    if sk.arms and keys:
        # the top row of the arm holes goes up to where the torso outline is no wider than the shoulder joints:
        # the loop then runs over the top of the shoulder instead of cutting a shelf into it
        top = zs_m[-2]
        for k in keys:
            if top <= k <= top + 0.08 * H and k < zs_m[-1] - 0.03 * H:
                side = abs(float(dense[k].verts[0][sk.ax]) - sk.c0)
                zs_m[-2] = k
                if side <= 1.05 * sh_off:
                    break
        for k in range(1, hr):
            zs_m[1 + k] = zs_m[1] + (zs_m[-2] - zs_m[1]) * k / hr
    sel = sorted({min(keys, key=lambda k: abs(k - z)) for z in zs_m})
    while len(sel) < par['torso_rings']:
        best = None
        for k in keys:
            if k in sel or min(abs(k - s) for s in sel) < 0.022 * H or (sk.arms and zs_m[1] - 1e-9 <= k <= zs_m[-2] + 1e-9):
                continue          # (no extra ring inside the rows of the arm holes)
            lo_ = max(s for s in sel if s < k) if any(s < k for s in sel) else None
            hi_ = min(s for s in sel if s > k) if any(s > k for s in sel) else None
            if lo_ is None or hi_ is None:
                continue
            t = (k - lo_) / (hi_ - lo_)
            interp = dense[lo_].verts * (1 - t) + dense[hi_].verts * t
            err = float(np.linalg.norm(dense[k].verts - interp, axis=1).max())
            if best is None or err > best[0]:
                best = (err, k)
        if best is None or best[0] < 0.004 * H:
            break
        sel = sorted(sel + [best[1]])
    step_i = None
    if z_hb is not None:
        z_att = max(z_hb, zs_m[-2] + 0.014 * H) if sk.arms else z_hb
        if float(np.percentile(Ph[:, 2], 97)) - z_att < 0.03 * H:
            z_hb = None
    if z_hb is not None:
        def fit(z, pts_):
            w = np.array([math.exp(-((k - z) / (0.05 * H)) ** 2) for k in ck])
            cy = float((w * np.array([cen[k] for k in ck])).sum() / w.sum())
            R = radial_profile(base + up * z + fwd * cy, up, lat, pts_, H, rmax=0.6, lock_u=True, sym=True, spike=3 if z < z_sh else 0, fixed=True)
            if R is not None and R.cover > 0.5:
                ring_verts(R, Nt)
                return R
            return None
        za_, zb2_ = z_att, z_att - 0.006 * H
        Ra, Rb = fit(za_, tor), fit(zb2_, tor_nh)
        run = []
        if Ra is not None and Rb is not None:
            # hair columns: where hair hangs below the step
            C = Rb.verts.mean(0)
            qh = Ph[(Ph[:, 2] < za_) & (Ph[:, 2] > za_ - 0.06 * H)] - C
            ah = np.arctan2(qh @ fwd, qh @ lat)
            hcol = []
            for k in range(Nt):
                a_ = math.atan2(float((Ra.verts[k] - C) @ fwd), float((Ra.verts[k] - C) @ lat))
                hcol.append(int((np.abs((ah - a_ + math.pi) % (2 * math.pi) - math.pi) < math.pi / Nt).sum()))
            kb0 = (3 * Nt // 4) if front_sign0 > 0 else (Nt // 4)
            thr = max(12, 0.15 * hcol[kb0])
            # never over the columns of the arm holes: one run of columns around the back
            hole_cols = {(k0 + j) % Nt for k0 in (0, Nt // 2) for j in range(-hc // 2 + 1, hc // 2)} if sk.arms else set()
            run = [kb0] if hcol[kb0] >= thr else []
            for stp_ in (1, -1):
                k = kb0
                while run and len(run) < Nt - 2 and hcol[(k + stp_) % Nt] >= thr and (k + stp_) % Nt not in run and (k + stp_) % Nt not in hole_cols:
                    k += stp_
                    run = run + [k % Nt] if stp_ > 0 else [k % Nt] + run
        if len(run) < 2:                    # no hair down the back: the head stays as it was
            Ra = None
        zlo_h = float(np.percentile(Ph[:, 2], 0.5)) - 0.01 * H
        if Ra is not None and Rb is not None:
            for k in keys:                  # below the step the cage does not know the hair
                if zlo_h < k < zb2_ and not (sk.arms and k <= zs_m[-2] + 1e-9):
                    R = fit(k, tor_nh)
                    if R is not None:
                        dense[k] = R
            dense[za_], dense[zb2_] = Ra, Rb
            fixed_ = set(zs_m[1:-1]) if sk.arms else set()
            sel = sorted({z for z in sel if z in fixed_ or not (zb2_ - 0.016 * H < z < za_ + 0.014 * H)} | {za_, zb2_})
            step_i = sel.index(zb2_)
            info['hair_step_z'] = float(zb2_)
    info['torso_ring_heights'] = [round((z - high.lo[2]) / H, 3) for z in sel]
    tr = [B.add(dense[z].verts) for z in sel]
    if step_i is not None:
        # elsewhere the two rings of the step are one (no sliver band around the front of the neck)
        Va, Vb = dense[sel[step_i + 1]].verts, dense[sel[step_i]].verts
        for k in range(Nt):
            if k not in run:
                B.v[tr[step_i][k]] = Vb[k] if np.linalg.norm(Va[k] - Vb[k]) < 0.02 * H else 0.5 * (Va[k] + Vb[k])
                tr[step_i + 1][k] = tr[step_i][k]
        if len(run) >= 2:
            info['hair_flap'] = dict(A=[tr[step_i + 1][k] for k in run], B=[tr[step_i][k] for k in run],
                                     Wa=tr[step_i][(run[0] - 1) % Nt], Wb=tr[step_i][(run[-1] + 1) % Nt], C=C, z_att=float(za_), Ph=Ph)
        log('  hair: part of the head down to %.3fH (shoulders %.3fH), step to the ring without hair at %.3fH; hair columns %s of %d (samples per column %s)' % (
            (za_ - high.lo[2]) / H, ((zs_m[-2] if sk.arms else z0) - high.lo[2]) / H, (zb2_ - high.lo[2]) / H, run, Nt, hcol))
    rowz = [min(range(len(sel)), key=lambda i: abs(sel[i] - z)) for z in (zs_m[1:-1] if sk.arms else [])]
    holes = {}
    if sk.arms:
        for side, k0 in ((1, 0), (-1, Nt // 2)):
            cols = [(k0 + j) % Nt for j in range(-hc // 2, hc // 2 + 1)]
            loop = [tr[rowz[0]][k] for k in cols]
            loop += [tr[r][cols[-1]] for r in rowz[1:-1]]
            loop += [tr[rowz[-1]][k] for k in cols[::-1]]
            loop += [tr[r][cols[0]] for r in rowz[1:-1][::-1]]
            holes[side] = (set(cols[:-1]), loop)
    for i in range(len(tr) - 1):
        for k in range(Nt):
            if sk.arms and rowz[0] <= i < rowz[-1] and any(k in holes[s][0] for s in holes):
                continue
            k1 = (k + 1) % Nt
            fq = [tr[i][k], tr[i][k1], tr[i + 1][k1], tr[i + 1][k]]
            fq = [v_ for j_, v_ in enumerate(fq) if v_ != fq[j_ - 1]]
            B.face(fq, 'body')
    B.cap(tr[-1], 'body')
    # seam down the back
    front_sign = 1.0 if float((sk.legs[0].joints[-1] - sk.legs[0].joints[-2]) @ fwd) > 0 else -1.0
    kb = (3 * Nt // 4) if front_sign > 0 else (Nt // 4)
    kf = (Nt // 4) if front_sign > 0 else (3 * Nt // 4)
    for i in range(len(tr) - 1):
        if tr[i][kb] != tr[i + 1][kb]:
            B.seam(tr[i][kb], tr[i + 1][kb])
    if step_i is not None:                  # the rim of the overhang is a seam: the underside is an island of its own
        for k in range(Nt):
            a_, b_ = tr[step_i + 1][k], tr[step_i + 1][(k + 1) % Nt]
            if tr[step_i][k] != a_ or tr[step_i][(k + 1) % Nt] != b_:
                B.seam(a_, b_)
    # ---- limbs
    def tube(rings, name, first_loop):
        idx = [B.add(r) for r in rings]
        first = B.bridge(first_loop, idx[0], 'body')
        # keep the columns of the tube aligned with the bridge
        sh = idx[0].index(first[0])
        rev = idx[0][(sh + 1) % len(idx[0])] != first[1]
        def reorder(r):
            r2 = r[::-1] if rev else r
            s2 = r2.index(r[sh])
            return r2[s2:] + r2[:s2]
        idx = [reorder(r) for r in idx]
        for a, b in zip(idx[:-1], idx[1:]):
            B.strip(a, b, 'body')
        B.cap(idx[-1], 'body')
        n = len(first_loop)
        for k in range(n):
            B.seam(first_loop[k], first_loop[(k + 1) % n])
        return idx

    tubes = {}
    for c in sk.arms:
        loop = holes[c.side][1]
        idx = tube(limb_rings[c.name], c.name, loop)
        # seam along the column under the arm (the lowest vertex of the hole)
        ks = min(range(len(loop)), key=lambda k: B.v[loop[k]][2])
        B.seam(loop[ks], idx[0][ks])
        for a, b in zip(idx[:-1], idx[1:]):
            B.seam(a[ks], b[ks])
        tubes[c.name] = idx
    # crotch vertices between the front and the back centre of the lowest ring
    pf, pb = B.v[tr[0][kf]], B.v[tr[0][kb]]
    cz = nearp[nearp[:, 2] < zcr + 0.02 * H] if len(nearp) > 20 else None
    pc = base + up * zcr + fwd * (float(np.median(cz @ fwd)) if cz is not None and len(cz) else float((pf + pb) @ fwd / 2))
    nc = par['crotch'] - 1
    cpts = []
    for q in range(nc):
        t = (q + 1) / (nc + 1) * 2
        cpts.append(pf + (pc - pf) * t if t <= 1 else pc + (pb - pc) * (t - 1))
    cv = B.add(cpts)                   # front -> back
    for a, b in zip([tr[0][kf]] + cv, cv + [tr[0][kb]]):
        B.seam(a, b)
    for c in sk.legs:
        ks_ = []
        k = kb
        stp = 1 if ((kf - kb) % Nt == Nt // 2 and c.side > 0) == (front_sign > 0) else -1
        # walk from the back centre to the front centre over this leg's side
        side_k = 0 if c.side > 0 else Nt // 2
        stp = 1 if (side_k - kb) % Nt < Nt // 2 else -1
        while True:
            ks_.append(k % Nt)
            if k % Nt == kf:
                break
            k += stp
        loop = [tr[0][k] for k in ks_] + cv
        idx = tube(limb_rings[c.name], c.name, loop)
        ks = len(ks_) + nc // 2            # the column that starts at the crotch
        B.seam(loop[ks], idx[0][ks])
        for a, b in zip(idx[:-1], idx[1:]):
            B.seam(a[ks], b[ks])
        tubes[c.name] = idx
    info['torso_rows'] = len(tr)
    return B, info


def builder_to_object(B, name, recalc=True):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    vs = [bm.verts.new(p) for p in B.v]
    bm.verts.ensure_lookup_table()
    lay = bm.faces.layers.int.new('sr_group')
    names = sorted(set(B.fg))
    for f, g in zip(B.f, B.fg):
        try:
            bf = bm.faces.new([vs[i] for i in f])
        except ValueError:
            continue
        bf[lay] = names.index(g)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    if recalc:
        kw = {names.index(g) for g in B.keep_winding if g in names}
        # the builder's winding is kept only for open sheets; a closed component is always wound consistently here
        # (a prism in a group that also holds relief solids) and then turned outward by its signed volume
        closed_f = _closed_faces(bm)
        bmesh.ops.recalc_face_normals(bm, faces=[f for f in bm.faces if f[lay] not in kw or f in closed_f])
        _orient_closed(bm)
    for e in bm.edges:
        e.seam = False
    idx = {v: i for i, v in enumerate(vs)}
    for e in bm.edges:
        a, b = idx[e.verts[0]], idx[e.verts[1]]
        if (min(a, b), max(a, b)) in B.seams:
            e.seam = True
    jt = {names.index(g) for g in B.join_tris if g in names}
    if jt:
        bmesh.ops.join_triangles(bm, faces=[f for f in bm.faces if f[lay] in jt and len(f.verts) == 3], cmp_seam=True, cmp_sharp=False, cmp_uvs=False,
                                 cmp_vcols=False, cmp_materials=False, angle_face_threshold=math.radians(20), angle_shape_threshold=math.radians(B.join_shape))
    bm.verts.index_update()
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    return o, names


def _closed_faces(bm):
    """the faces of the closed components (every edge with exactly 2 faces)"""
    seen, out = set(), set()
    for f0 in bm.faces:
        if f0 in seen:
            continue
        st, fs = [f0], []
        seen.add(f0)
        while st:
            g = st.pop()
            fs.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h not in seen:
                        seen.add(h)
                        st.append(h)
        if all(len(e.link_faces) == 2 for g in fs for e in g.edges):
            out.update(fs)
    return out


def _orient_closed(bm):
    """every closed component (all edges with 2 faces) gets its normals outward: wound consistently
    (recalc_face_normals did that, or the builder), the signed volume decides the side"""
    seen = set()
    for f0 in bm.faces:
        if f0 in seen:
            continue
        st, fs = [f0], []
        seen.add(f0)
        while st:
            g = st.pop()
            fs.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h not in seen:
                        seen.add(h)
                        st.append(h)
        if not all(len(e.link_faces) == 2 for g in fs for e in g.edges):
            continue
        vol = 0.0
        for g in fs:
            vv = [l.vert.co for l in g.loops]
            for k in range(1, len(vv) - 1):
                vol += vv[0].dot(vv[k].cross(vv[k + 1]))
        if vol < 0:
            bmesh.ops.reverse_faces(bm, faces=fs)


# ====================================================================== extras: arcs and prisms
def _pca(P):
    c = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    order = np.argsort(-w)
    return c, V[:, order].T


def arc_stats(pts, nrm, axis_p, sk):
    """cluster `pts` seen from the vertical axis through axis_p -> dict (arc range, radial spread, inward share)"""
    q = pts - axis_p
    x, y = q @ sk.lat, q @ sk.fwd
    th = np.arctan2(y, x)
    r = np.hypot(x, y)
    nb = 36
    h = np.bincount(np.minimum(((th + math.pi) / (2 * math.pi) * nb).astype(int), nb - 1), minlength=nb)
    empty = h < max(2, 0.002 * len(pts))
    # the longest empty run (circular)
    best, run, start = (0, 0), 0, 0
    for k in range(2 * nb):
        if empty[k % nb]:
            if run == 0:
                start = k
            run += 1
            if run > best[0]:
                best = (min(run, nb), start)
        else:
            run = 0
    gap = best[0] * 2 * math.pi / nb
    centre = -math.pi + (best[1] + best[0] / 2.0) * 2 * math.pi / nb + math.pi      # opposite of the gap centre
    rel = (th - centre + math.pi) % (2 * math.pi) - math.pi
    # radial spread inside (angle, height) cells
    zi = ((pts[:, 2] - pts[:, 2].min()) / max(np.ptp(pts[:, 2]), 1e-9) * 12).astype(int)
    ai = ((rel + math.pi) / (2 * math.pi) * 24).astype(int)
    cell = zi * 32 + ai
    order = np.argsort(cell)
    cs, rs = cell[order], r[order]
    cut = np.flatnonzero(np.diff(cs)) + 1
    spreads = [np.percentile(g, 85) - np.percentile(g, 15) for g in np.split(rs, cut) if len(g) >= 8]
    rad = q - np.outer(q @ sk.up, sk.up)
    rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-12)
    inward = float(((nrm * rad).sum(1) < -0.2).mean())
    return dict(gap=gap, centre=centre, rel=rel, r=r, spread=float(np.median(spreads)) if spreads else 0.0, inward=inward)


def build_arc(B, name, pts, st, axis_p, sk, quads, double, log=print, body=None, extend=None, margin=0.008, out=None):
    """open (or closed) grid around the vertical axis through axis_p, fitted to the outer side of the cluster.
    body: BVH (or list of BVHs: the body cage, sheets that lie below this one) the grid must stay outside of.
    extend: dict(pts, z1, lo, hi): the sheet goes on upward to the height z1 over the surface samples `pts`
    inside the angle range lo .. hi (a cape whose top is hidden under a collar: up to the neck).
    out: dict that receives the grid (P, quads) of the outer side"""
    H = sk.H
    rel, r = st['rel'], st['r']
    z = pts[:, 2]
    z0, z1 = np.percentile(z, 0.7), np.percentile(z, 99.3)
    closed = st['gap'] < math.radians(50)
    Lz = z1 - z0
    Lth = (2 * math.pi - st['gap']) * float(np.median(r))
    cols = int(np.clip(round(math.sqrt(quads * Lth / max(Lz, 1e-9))), 3 if closed else 2, 16))
    rows = int(np.clip(round(quads / cols), 2, 14))
    if extend is not None and not closed:
        tp = extend['pts']
        q = tp - axis_p
        rel_t = (np.arctan2(q @ sk.fwd, q @ sk.lat) - st['centre'] + math.pi) % (2 * math.pi) - math.pi
        m = (tp[:, 2] > z1 - 0.01 * H) & (tp[:, 2] < extend['z1'] + 0.004 * H) & (rel_t > extend['lo']) & (rel_t < extend['hi'])
        if m.sum() > 50:
            rel = np.concatenate([rel, rel_t[m]])
            r = np.concatenate([r, np.hypot(q @ sk.lat, q @ sk.fwd)[m]])
            z = np.concatenate([z, tp[m, 2]])
            rows = int(min(rows + max(1, round(rows * (extend['z1'] - z1) / max(Lz, 1e-9))), 16))
            log('  %s: goes on upward %.3fH to the collar (%d more rows)' % (name, (extend['z1'] - z1) / H, max(1, round(rows * (extend['z1'] - z1) / max(Lz, 1e-9)))))
            z1 = extend['z1']
            Lz = z1 - z0
    if closed and cols % 2:
        cols += 1
    # dense grid of the radius over (height, position across the sheet); the rows and columns of the low grid
    # are then put where the shape changes most (folds, the flare of the hem) instead of evenly
    NZ, NS = max(25, 3 * rows + 1), max(49, 4 * cols + 1)
    zd = np.linspace(z0, z1, NZ)
    slab = max(0.008 * H, 0.6 * Lz / (NZ - 1))
    lohi = []
    for zc in zd:
        m = np.abs(z - zc) < max(slab, 0.012 * H)
        if m.sum() < 8:
            lohi.append((np.nan, np.nan))
            continue
        lohi.append((-math.pi, math.pi) if closed else (float(np.percentile(rel[m], 1.5)), float(np.percentile(rel[m], 98.5))))
    lo = _fill1(np.array([a_ for a_, _ in lohi]))
    hi = _fill1(np.array([b_ for _, b_ in lohi]))
    if not closed:                         # the outline of the sheet must not jump from row to row
        for _ in range(4):
            lo, hi = _smooth1(lo), _smooth1(hi)
    ncd = NS if not closed else NS - 1
    G = np.full((NZ, ncd), np.nan)
    for i, zc in enumerate(zd):
        m = np.abs(z - zc) < slab
        a_, r_ = rel[m], r[m]
        w = (hi[i] - lo[i]) / (NS - 1)
        for j in range(ncd):
            ph = lo[i] + w * j
            mm = np.abs((a_ - ph + math.pi) % (2 * math.pi) - math.pi) < 0.75 * w
            if mm.sum() >= 3:
                G[i, j] = np.percentile(r_[mm], 88)
    if np.isnan(G).all():
        return 0
    for i in range(NZ):
        G[i] = _fill1(G[i], circular=closed)
    for j in range(ncd):
        G[:, j] = _fill1(G[:, j])
    for i in range(NZ):                    # light smoothing of the dense grid (sampling noise, single strands)
        G[i] = 0.25 * np.roll(G[i], 1) + 0.5 * G[i] + 0.25 * np.roll(G[i], -1) if closed else _smooth1(G[i])
    for j in range(ncd):
        G[:, j] = _smooth1(G[:, j])
    rmean = float(np.mean(G))

    def pick(n, N, err, start, circular=False):
        sel = sorted(start)
        gap = max(1, int(0.4 * N / max(n, 1)))
        while len(sel) < n:
            best = None
            for k in range(N):
                if k in sel or min(min(abs(k - s_), N - abs(k - s_)) if circular else abs(k - s_) for s_ in sel) < gap:
                    continue
                lo_ = max((s_ for s_ in sel if s_ < k), default=None)
                hi_ = min((s_ for s_ in sel if s_ > k), default=None)
                if circular:
                    lo_ = sel[-1] - N if lo_ is None else lo_
                    hi_ = sel[0] + N if hi_ is None else hi_
                if lo_ is None or hi_ is None:
                    continue
                e = err(k, lo_, hi_)
                if best is None or e > best[0]:
                    best = (e, k)
            if best is None:
                break
            sel = sorted(sel + [best[1]])
        return sel

    def row_err(k, a_, b_):
        t = (k - a_) / (b_ - a_)
        e = float(np.abs(G[k] - (G[a_] * (1 - t) + G[b_] * t)).mean())
        return e + rmean * (abs(lo[k] - (lo[a_] * (1 - t) + lo[b_] * t)) + abs(hi[k] - (hi[a_] * (1 - t) + hi[b_] * t)))

    def col_err(k, a_, b_):
        t = (k - a_) / (b_ - a_)
        return float(np.abs(G[:, k % ncd] - (G[:, a_ % ncd] * (1 - t) + G[:, b_ % ncd] * t)).mean())

    ri = pick(rows + 1, NZ, row_err, [0, NZ - 1])
    if closed:
        ci = pick(cols, ncd, col_err, [0, ncd // 2], circular=True)
    else:
        ci = pick(cols + 1, ncd, col_err, [0, ncd - 1])
    rows, cols = len(ri) - 1, (len(ci) if closed else len(ci) - 1)
    nc = len(ci)
    zs = zd[ri]
    P = np.zeros((rows + 1, nc, 3))
    for i, k in enumerate(ri):
        w = (hi[k] - lo[k]) / (NS - 1)
        for j, cj in enumerate(ci):
            a_ = st['centre'] + lo[k] + w * cj
            P[i, j] = axis_p + sk.up * (zd[k] - axis_p[2]) + G[k, cj] * (math.cos(a_) * sk.lat + math.sin(a_) * sk.fwd)
    obst = [] if body is None else (list(body) if isinstance(body, (list, tuple)) else [body])
    if obst:
        # a sheet that lies on the body (hair on the chest and the back, the top of a cape) must not dip into the
        # body cage or into a sheet below it: its vertices stay a little outside, and so do its faces (centre and
        # edge middles: a flat quad between four vertices that are outside can still cut through a shoulder)
        def height(p_):
            best = None
            for bv in obst:
                loc, nrm, _fi, d_ = bv.find_nearest(Vector(p_))
                if loc is None or d_ > 0.06 * H:
                    continue
                hgt = float((Vector(p_) - loc).dot(nrm))
                if best is None or hgt < best[0]:
                    best = (hgt, np.array(loc), np.array(nrm))
            return best

        P0 = P.copy()
        for i in range(rows + 1):
            for j in range(nc):
                b_ = height(P[i, j])
                if b_ is not None and b_[0] < margin * H:
                    P[i, j] = b_[1] + b_[2] * (margin * H)
        for _ in range(5):
            moved = 0
            for i in range(rows):
                for j in range(cols):
                    quad = [(i, j), (i, (j + 1) % nc), (i + 1, (j + 1) % nc), (i + 1, j)]
                    Q = np.array([P[a_] for a_ in quad])
                    need, dirn = 0.0, None
                    for sp in [Q.mean(0)] + [(Q[k] + Q[(k + 1) % 4]) / 2 for k in range(4)] + [(Q[k] + Q.mean(0)) / 2 for k in range(4)]:
                        b_ = height(sp)
                        if b_ is not None and b_[0] < 0.35 * margin * H and margin * H - b_[0] > need:
                            need, dirn = margin * H - b_[0], b_[2]
                    if need > 0:
                        for a_ in quad:
                            if np.linalg.norm(P[a_] - P0[a_]) < 0.06 * H:
                                P[a_] = P[a_] + dirn * need * 0.8
                        moved += 1
            if not moved:
                break
    if out is not None:
        out['P'] = P
        out['quads'] = [((i * nc + j), (i * nc + (j + 1) % nc), ((i + 1) * nc + (j + 1) % nc), ((i + 1) * nc + j)) for i in range(rows) for j in range(cols)]
    idx = [B.add(P[i]) for i in range(rows + 1)]
    n0 = len(B.f)
    for i in range(rows):
        for j in range(cols):
            j1 = (j + 1) % nc
            B.face([idx[i][j], idx[i][j1], idx[i + 1][j1], idx[i + 1][j]], name)      # counter clockwise seen from outside (z up, angle growing)
    if closed:
        for i in range(rows):
            B.seam(idx[i][0], idx[i + 1][0])
    if not double:
        B.keep_winding.add(name)
    if double:
        t = 0.006 * H
        inner = []
        for i in range(rows + 1):
            row = []
            for j in range(nc):
                border = False              # round 7: the rim has its own inner vertices and a wall (shared rim vertices made the sides touch)
                if border:
                    row.append(idx[i][j])
                else:
                    # along the normal of the grid (not radially: where the sheet lies flat on the shoulders a
                    # radial offset is no thickness at all and the two sides cut through each other)
                    p = P[i, j]
                    du = P[i, min(j + 1, nc - 1) if not closed else (j + 1) % nc] - P[i, max(j - 1, 0) if not closed else (j - 1) % nc]
                    dv = P[min(i + 1, rows), j] - P[max(i - 1, 0), j]
                    nn = np.cross(du, dv)
                    rd = p - axis_p
                    rd = rd - sk.up * float(rd @ sk.up)
                    rd = rd / max(np.linalg.norm(rd), 1e-9)
                    ln_ = float(np.linalg.norm(nn))
                    nn = nn / ln_ if ln_ > 1e-12 else rd
                    q_in = p - nn * t
                    # ... and behind every outer quad around the vertex (a curled edge: the sides would cross)
                    for _pass in range(2):
                        for ci_, cj_ in ((i - 1, j - 1), (i - 1, j), (i, j - 1), (i, j)):
                            if not (0 <= ci_ < rows) or (not closed and not (0 <= cj_ < cols)):
                                continue
                            ca, cb = cj_ % nc, (cj_ + 1) % nc
                            nq = np.cross(P[ci_, cb] - P[ci_, ca], P[ci_ + 1, ca] - P[ci_, ca]) + np.cross(P[ci_ + 1, ca] - P[ci_ + 1, cb], P[ci_, cb] - P[ci_ + 1, cb])
                            lq = float(np.linalg.norm(nq))
                            if lq < 1e-12:
                                continue
                            nq = nq / lq
                            for corner in (P[ci_, ca], P[ci_, cb], P[ci_ + 1, ca], P[ci_ + 1, cb]):
                                dd = float((q_in - corner) @ nq)
                                if dd > -0.5 * t:
                                    q_in = q_in - nq * (dd + 0.5 * t)
                    if obst:                # the inner side stays outside the body too (the outer vertex moves with it)
                        b_ = height(q_in)
                        if b_ is not None and b_[0] < 0.002 * H:
                            sh_ = b_[2] * (0.002 * H - b_[0])
                            q_in = q_in + sh_
                            P[i, j] = P[i, j] + sh_
                            B.v[idx[i][j]] = B.v[idx[i][j]] + sh_
                    row.append(B.add([q_in])[0])
            inner.append(row)
        for i in range(rows):
            for j in range(cols):
                j1 = (j + 1) % nc
                B.face([inner[i][j], inner[i + 1][j], inner[i + 1][j1], inner[i][j1]], name)
        for j in range(cols):               # wall around the rim
            j1 = (j + 1) % nc
            B.face([idx[0][j1], idx[0][j], inner[0][j], inner[0][j1]], name)
            B.face([idx[rows][j], idx[rows][j1], inner[rows][j1], inner[rows][j]], name)
            B.seam(inner[0][j], inner[0][j1])
            B.seam(inner[rows][j], inner[rows][j1])
        if not closed:
            for i in range(rows):
                B.face([idx[i][0], idx[i + 1][0], inner[i + 1][0], inner[i][0]], name)
                B.face([idx[i + 1][nc - 1], idx[i][nc - 1], inner[i][nc - 1], inner[i + 1][nc - 1]], name)
                B.seam(inner[i][0], inner[i + 1][0])
                B.seam(inner[i][nc - 1], inner[i + 1][nc - 1])
            B.seam(idx[0][0], inner[0][0])
        if False and not closed and rows >= 2 and cols >= 2:
            # the 4 corner cells share 3 of their 4 vertices with their twin on the other side: both are split
            # along the diagonal through the one vertex that differs (split the other way they would coincide)
            n_arc = rows * cols
            base = len(B.f) - 2 * n_arc
            drop = []
            for ci_, cj_, ii_, jj_ in ((0, 0, 1, 1), (0, cols - 1, 1, cols - 1), (rows - 1, 0, rows - 1, 1), (rows - 1, cols - 1, rows - 1, cols - 1)):
                kc = (0 if ci_ == 0 else rows, 0 if cj_ == 0 else nc - 1)           # the corner vertex of the grid
                for layer, grid_ in ((0, idx), (1, inner)):
                    fi_ = base + layer * n_arc + ci_ * cols + cj_
                    f_ = B.f[fi_]
                    a_, b_ = grid_[kc[0]][kc[1]], grid_[ii_][jj_]
                    if a_ in f_ and b_ in f_ and abs(f_.index(a_) - f_.index(b_)) == 2:
                        k0 = f_.index(a_)
                        r_ = f_[k0:] + f_[:k0]
                        B.f.append([r_[0], r_[1], r_[2]])
                        B.fg.append(name)
                        B.f.append([r_[0], r_[2], r_[3]])
                        B.fg.append(name)
                        drop.append(fi_)
            for fi_ in sorted(drop, reverse=True):
                del B.f[fi_]
                del B.fg[fi_]
        # the rim is the seam between the two sides
        for i in range(rows):
            if not closed:
                B.seam(idx[i][0], idx[i + 1][0])
                B.seam(idx[i][nc - 1], idx[i + 1][nc - 1])
            else:
                B.seam(inner[i][0], inner[i + 1][0])
        for j in range(cols):
            j1 = (j + 1) % nc
            B.seam(idx[0][j], idx[0][j1])
            B.seam(idx[rows][j], idx[rows][j1])
    log('  %s: arc %s %d x %d%s, %.0f deg' % (name, 'closed' if closed else 'open', rows, cols, ' double sided' if double else '', math.degrees(2 * math.pi - st['gap'])))
    return len(B.f) - n0


def _fill1(a, circular=False):
    a = np.asarray(a, float).copy()
    ok = ~np.isnan(a)
    if ok.all() or not ok.any():
        return a
    if circular:
        return _circ_fill(a)
    i = np.arange(len(a))
    return np.interp(i, i[ok], a[ok])


def _smooth1(a):
    b = a.copy()
    b[1:-1] = 0.25 * a[:-2] + 0.5 * a[1:-1] + 0.25 * a[2:]
    return b


def build_prism(B, name, pts, sk, tris, N=4, axis=None, max_rings=None, tol=0.004, extend=(0.0, 0.0), smooth_axis=0, log=print):
    """box / N sided prism along the main axis of the cluster; ring positions chosen where the outline changes.
    extend: the two end rings are moved outward by these lengths (into the piece the prism is stuck in)"""
    H = sk.H
    c, V = _pca(pts)
    e1 = V[0] if axis is None else np.asarray(axis, float)
    e2 = V[1] - e1 * float(V[1] @ e1)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    q = pts - c
    t, x, y = q @ e1, q @ e2, q @ e3
    t0, t1 = np.percentile(t, 0.5), np.percentile(t, 99.5)
    L = t1 - t0
    K = max(2, int((tris - 2 * (N - 2)) // (2 * N)) + 1)
    if max_rings:
        K = min(K, max_rings)
    nd = int(np.clip(L / (0.004 * H), 8, 80))
    ts = np.linspace(t0, t1, nd)
    slab = max(L / nd, 0.004 * H)
    rings = []
    for tc in ts:
        m = np.abs(t - tc) < slab
        if m.sum() < 6:
            rings.append(None)
            continue
        if N == 4:
            xa, xb = np.percentile(x[m], 3), np.percentile(x[m], 97)
            ya, yb = np.percentile(y[m], 3), np.percentile(y[m], 97)
            rings.append(np.array([c + e1 * tc + e2 * a + e3 * b for a, b in ((xa, ya), (xb, ya), (xb, yb), (xa, yb))]))
        else:
            R = radial_profile(c + e1 * tc, e1, e2, pts, H, slab=slab / H, rmax=10.0, spike=1, pct=95, nb=32)
            rings.append(ring_verts(R, N) if R is not None else None)
    ok = [k for k in range(nd) if rings[k] is not None]
    if len(ok) < 2:
        return 0
    for k in range(nd):
        if rings[k] is None:
            j = min(ok, key=lambda g: abs(g - k))
            rings[k] = rings[j] + e1 * (ts[k] - ts[j])
    if N != 4 and smooth_axis:              # the centre line of a shaft is smooth: ornaments on it pull single rings aside (a kink)
        C = np.array([r_.mean(0) for r_ in rings])
        Cs = C.copy()
        for _ in range(smooth_axis):
            Cs[1:-1] = 0.25 * Cs[:-2] + 0.5 * Cs[1:-1] + 0.25 * Cs[2:]
        Cs[0], Cs[-1] = 2 * Cs[1] - Cs[2], 2 * Cs[-2] - Cs[-3]
        rings = [r_ + (cs_ - c_) for r_, c_, cs_ in zip(rings, C, Cs)]
        # ... and its thickness changes slowly (median over 3 slices)
        rad = np.array([np.linalg.norm(r_ - r_.mean(0), axis=1) for r_ in rings])
        radm = rad.copy()
        radm[1:-1] = np.median(np.stack([rad[:-2], rad[1:-1], rad[2:]]), axis=0)
        rings = [r_.mean(0) + (r_ - r_.mean(0)) * (rm_ / np.maximum(r0_, 1e-12))[:, None] for r_, r0_, rm_ in zip(rings, rad, radm)]
    sel = [0, nd - 1]
    while len(sel) < K:
        best = None
        for k in range(nd):
            if k in sel:
                continue
            a = max(s for s in sel if s < k)
            b = min(s for s in sel if s > k)
            w = (k - a) / (b - a)
            err = float(np.linalg.norm(rings[k] - (rings[a] * (1 - w) + rings[b] * w), axis=1).max())
            if best is None or err > best[0]:
                best = (err, k)
        if best is None or best[0] < tol * H:
            break
        sel = sorted(sel + [best[1]])
    for end, nxt_, ln in ((sel[0], sel[1], extend[0]), (sel[-1], sel[-2], extend[1])):
        if ln:                              # along the prism as it runs there (a bent shaft)
            d = rings[end].mean(0) - rings[nxt_].mean(0)
            rings[end] = rings[end] + d / max(np.linalg.norm(d), 1e-12) * ln
    idx = [B.add(rings[k]) for k in sel]
    n0 = len(B.f)
    for a, b in zip(idx[:-1], idx[1:]):
        B.strip(a, b, name)
        B.seam(a[0], b[0])
    B.cap(idx[0], name)
    B.cap(idx[-1], name)
    log('  %s: prism %d sides, %d rings, length %.2fH' % (name, N, len(sel), L / H))
    return len(B.f) - n0


# ====================================================================== character
def _body_bvh(B):
    bm = bmesh.new()
    vs = [bm.verts.new(p) for p in B.v]
    for f in B.f:
        try:
            bm.faces.new([vs[i] for i in f])
        except ValueError:
            pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    t = BVHTree.FromBMesh(bm)
    bm.free()
    return t


def sample_polys(high, ks, n, seed=11):
    """n surface samples of the polygons ks (local indices) of `high`"""
    tm = np.flatnonzero(np.isin(high.tri_poly, ks))
    ar = high.tri_area[tm]
    if not len(tm) or ar.sum() <= 0:
        return np.zeros((0, 3))
    rng = np.random.default_rng(seed)
    ti = tm[rng.choice(len(tm), size=n, p=ar / ar.sum())]
    a, b, c = (high.co[high.tris[ti, j]] for j in range(3))
    u, v = rng.random(n), rng.random(n)
    f = u + v > 1
    u[f], v[f] = 1 - u[f], 1 - v[f]
    return a + (b - a) * u[:, None] + (c - a) * v[:, None]


def _extents(P):
    c, V = _pca(P)
    q = P - c
    return c, V, np.array([np.percentile(q @ V[i], 98.5) - np.percentile(q @ V[i], 1.5) for i in range(3)])


def build_sheet(B, name, P, quads, H, frame=None, min_cover=0.12, sep=0.004, max_cells=16, inside=None, symmetric=False, log=print):
    """a thin part (hat brim, fin, flange, blade, the limbs of a bow) as a flat-ish plane: a grid over the main
    plane of the samples P, every node at the height of the middle of the part there, only the cells the part
    covers (not more than `quads` of them).  Two unconnected layers sep * H apart (both sides of the part are
    visible); what a cell holds besides the part is cut out by the alpha of the bake.
    inside(point) -> True: cells that lie completely inside something else (the body cage) are left out.
    -> number of faces"""
    if frame is None:
        c, V = _pca(P)
        e1, e2 = V[0], V[1]
    else:
        c, e1, e2 = frame
    e3 = np.cross(e1, e2)
    q = P - c
    u, v, w = q @ e1, q @ e2, q @ e3
    u0, u1 = np.percentile(u, 0.3), np.percentile(u, 99.7)
    v0, v1 = np.percentile(v, 0.3), np.percentile(v, 99.7)
    if symmetric:                           # e1 is the mirror direction, c lies on the mirror plane
        u1 = max(abs(u0), abs(u1))
        u0 = -u1
    Lu, Lv = max(u1 - u0, 1e-9), max(v1 - v0, 1e-9)
    # projected area from a fine occupancy grid
    fine = 48
    fi = np.clip(((u - u0) / Lu * fine).astype(int), 0, fine - 1) * fine + np.clip(((v - v0) / Lv * fine).astype(int), 0, fine - 1)
    area = len(np.unique(fi)) * (Lu / fine) * (Lv / fine)
    cell = math.sqrt(area / max(quads, 1))
    sub = 4
    for _try in range(12):
        nu = int(np.clip(round(Lu / cell), 1, max_cells))
        nv = int(np.clip(round(Lv / cell), 1, max_cells))
        du, dv = Lu / nu, Lv / nv
        si = np.clip(((u - u0) / du * sub).astype(int), 0, nu * sub - 1)
        sj = np.clip(((v - v0) / dv * sub).astype(int), 0, nv * sub - 1)
        occ = np.zeros((nu * sub, nv * sub), dtype=bool)
        occ[si, sj] = True
        cover = occ.reshape(nu, sub, nv, sub).sum((1, 3)) / float(sub * sub)
        keep = cover >= min_cover
        if not keep.any():
            keep[np.unravel_index(cover.argmax(), cover.shape)] = True
        # two kept cells that touch only at a corner: fill one of the other two (no pinched vertex)
        for i in range(nu - 1):
            for j in range(nv - 1):
                if keep[i, j] and keep[i + 1, j + 1] and not keep[i + 1, j] and not keep[i, j + 1]:
                    keep[i + 1, j] = True
                if keep[i + 1, j] and keep[i, j + 1] and not keep[i, j] and not keep[i + 1, j + 1]:
                    keep[i, j] = True
        if symmetric:
            keep |= keep[::-1]
        if keep.sum() <= max(quads, 1) * 1.1 or (nu == 1 and nv == 1):
            break
        cell *= 1.1
    # node heights: the middle of the part
    mid = np.full((nu + 1, nv + 1), np.nan)
    gu = (u - u0) / du
    gv = (v - v0) / dv
    for i in range(nu + 1):
        mu = np.abs(gu - i) < 0.5
        if not mu.any():
            continue
        wv, gvv = w[mu], gv[mu]
        for j in range(nv + 1):
            m = np.abs(gvv - j) < 0.5
            if m.sum() >= 3:
                mid[i, j] = np.median(wv[m])
    ok = ~np.isnan(mid)
    if not ok.any():
        return 0
    oi, oj = np.nonzero(ok)
    for i in range(nu + 1):
        for j in range(nv + 1):
            if not ok[i, j]:
                k = int(((oi - i) ** 2 + (oj - j) ** 2).argmin())
                mid[i, j] = mid[oi[k], oj[k]]
    if symmetric:
        mid = (mid + mid[::-1]) / 2

    def pos(a, b, off=0.0):
        return c + e1 * (u0 + a * du) + e2 * (v0 + b * dv) + e3 * (mid[a, b] + off)

    if inside is not None and keep.sum() > 1:
        ins = np.array([[inside(pos(a, b)) for b in range(nv + 1)] for a in range(nu + 1)])
        for i in range(nu):
            for j in range(nv):
                if keep[i, j] and ins[i, j] and ins[i + 1, j] and ins[i, j + 1] and ins[i + 1, j + 1]:
                    keep[i, j] = False
        if not keep.any():
            return 0
    n0 = len(B.f)
    for layer, off in ((0, sep * H / 2), (1, -sep * H / 2)):
        idx = {}
        for i in range(nu):
            for j in range(nv):
                if not keep[i, j]:
                    continue
                vs = []
                for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                    if (a, b) not in idx:
                        idx[(a, b)] = B.add([pos(a, b, off)])[0]
                    vs.append(idx[(a, b)])
                B.face(vs if layer == 0 else vs[::-1], name)
    B.keep_winding.add(name)
    return len(B.f) - n0


def build_rod(B, name, P, H, tris, log=print, cage_dist=None):
    """a thin long part (finger, claw, spike): a box from its thick end to (if `tris` allows) a ring where it
    bends most, then a point at the tip when it tapers (else a second ring and a cap).  -> number of faces"""
    c, V = _pca(P)
    e1, e2, e3 = V[0], V[1], np.cross(V[0], V[1])
    q = P - c
    t = q @ e1
    t0, t1 = np.percentile(t, 0.5), np.percentile(t, 99.5)
    L = t1 - t0
    if L < 1e-9:
        return 0
    r = np.linalg.norm(q - np.outer(t, e1), axis=1)
    flip = np.median(r[t > t0 + 0.75 * L]) > np.median(r[t < t0 + 0.25 * L])      # the tip is the thin end
    if cage_dist is not None:               # ... or, better, the end away from the cage
        a_, b_ = P[t < t0 + 0.2 * L], P[t > t0 + 0.8 * L]
        if len(a_) and len(b_):
            flip = float(np.mean([cage_dist(p_) for p_ in a_[::max(1, len(a_) // 30)]])) > float(np.mean([cage_dist(p_) for p_ in b_[::max(1, len(b_) // 30)]]))
    if flip:
        e1, e3, t = -e1, -e3, -t
        t0, t1 = -t1, -t0
    x, y = q @ e2, q @ e3
    nd = 10
    cen, half = [], []
    for k in range(nd):
        m = (t >= t0 + L * k / nd) & (t <= t0 + L * (k + 1) / nd)
        if m.sum() < 5:
            cen.append(None)
            half.append(None)
            continue
        xa, xb, ya, yb = np.percentile(x[m], 6), np.percentile(x[m], 94), np.percentile(y[m], 6), np.percentile(y[m], 94)
        cen.append(np.array([(xa + xb) / 2, (ya + yb) / 2]))
        hx, hy = max(xb - xa, 0.002 * H) / 2, max(yb - ya, 0.002 * H) / 2
        half.append(np.array([max(hx, 0.6 * hy), max(hy, 0.6 * hx)]))
    ok = [k for k in range(nd) if cen[k] is not None]
    if len(ok) < 3:
        return 0

    def at(k):
        k = min(ok, key=lambda g: abs(g - k))
        return cen[k], half[k]

    def ring(k, tt):
        cc, hh = at(k)
        return [c + e1 * tt + e2 * (cc[0] + sx * hh[0]) + e3 * (cc[1] + sy * hh[1]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]

    cb, hb = at(1)
    ct, ht = at(nd - 1)
    taper = True          # fingers, claws, spikes, strands: all end in a point (a blunt end would need 6 more triangles)
    rings = [ring(1, t0)]
    best = 1
    if tris >= 14:
        # the ring where the axis deviates most from the straight line base .. tip
        best = max(range(3, nd - 2), key=lambda k: float(np.linalg.norm(at(k)[0] - (cb + (ct - cb) * (k - 1) / (nd - 2)))))
        if float(np.linalg.norm(at(best)[0] - (cb + (ct - cb) * (best - 1) / (nd - 2)))) < 0.08 * L:
            best = nd // 2
        rings.append(ring(best, t0 + L * (best + 0.5) / nd))
    n0 = len(B.f)
    idx = [B.add(r_) for r_ in rings]
    for a_, b_ in zip(idx[:-1], idx[1:]):
        B.strip(a_, b_, name)
        B.seam(a_[0], b_[0])
    if taper:
        apex = B.add([c + e1 * t1 + e2 * ct[0] + e3 * ct[1]])[0]
        last = idx[-1]
        for k in range(4):
            B.face([last[k], last[(k + 1) % 4], apex], name)
        B.seam(last[0], apex)
    else:
        cc, hh = at(nd - 1)
        hp = at(best if len(rings) > 1 else 1)[1]
        hh = np.minimum(hh, 0.7 * hp)
        end = B.add([c + e1 * t1 + e2 * (cc[0] + sx * hh[0]) + e3 * (cc[1] + sy * hh[1]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        B.strip(idx[-1], end, name)
        B.seam(idx[-1][0], end[0])
        B.cap(end, name)
    B.cap(idx[0], name)
    return len(B.f) - n0


def _parity_bad(B, group, new_faces, H, centres=None):
    """how many of the faces `new_faces` (vertex tuples) of the closed cage `group` point into it: a ray from the
    face along its normal must cross the cage an even number of times (the test of dev/skelretopo_check.py)"""
    co = [tuple(p) for p in B.v]
    tris, owner, polys = [], [], []
    for f, g in zip(B.f, B.fg):
        if g != group:
            continue
        k = len(polys)
        polys.append(f)
        for i in range(1, len(f) - 1):
            tris.append((f[0], f[i], f[i + 1]))
            owner.append(k)
    if not tris:
        return 0
    bv = BVHTree.FromPolygons(co, tris)
    eps = 1e-5 * H
    jit = np.array([0.0137, 0.0071, -0.0113])
    want = None if new_faces is None else {frozenset(f) for f in new_faces}       # None: all faces of the group
    bad = 0
    for k, f in enumerate(polys):
        if want is not None and frozenset(f) not in want:
            continue
        P = np.array([B.v[i] for i in f], float)
        n = np.zeros(3)
        for i in range(len(f)):             # Newell
            a, b = P[i], P[(i + 1) % len(f)]
            n += np.cross(a, b)
        ln = float(np.linalg.norm(n))
        if ln < 1e-9 * H * H:
            continue
        n /= ln
        d = n + jit
        d /= np.linalg.norm(d)
        o = Vector(P.mean(0) + n * eps)
        cnt = 0
        for _ in range(100):
            hit = bv.ray_cast(o, Vector(d))
            if hit[0] is None:
                break
            if owner[hit[2]] != k:
                cnt += 1
            o = hit[0] + Vector(d) * eps
        bad += cnt % 2
        if cnt % 2 and centres is not None:
            centres.append(P.mean(0))
    return bad


def plate_spikes(P, H, px=0.0025, r_open=0.008, min_len=0.016):
    """the spikes of a flat piece: its samples P in its own plane, rastered; what an opening (radius r_open * H)
    takes away and reaches further than min_len * H from the rest is a spike.
    -> list of dict(tip, root (3D), length, width) , longest first"""
    c, Vp = _pca(P)
    q = P - c
    u, v = q @ Vp[0], q @ Vp[1]
    s = px * H
    i = np.floor((u - u.min()) / s).astype(int) + 6
    j = np.floor((v - v.min()) / s).astype(int) + 6
    mask = np.zeros((i.max() + 7, j.max() + 7), dtype=bool)
    mask[i, j] = True
    mask = _morph(_morph(mask, 1, True), 1, False)
    r = max(2, int(round(r_open / px)))
    core = _morph(_morph(mask, r, False), r, True) & mask
    if not core.any():
        return []
    rest = mask & ~core
    ci, cj = np.nonzero(core & ~_morph(core, 1, False))          # the rim of the core
    C = np.stack([ci, cj], axis=1).astype(float)
    lab = np.zeros(mask.shape, dtype=int)
    out = []
    n = 0
    for a, b in zip(*np.nonzero(rest)):
        if lab[a, b]:
            continue
        n += 1
        stack, cells = [(a, b)], []
        lab[a, b] = n
        while stack:
            x, y = stack.pop()
            cells.append((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                x2, y2 = x + dx, y + dy
                if rest[x2, y2] and not lab[x2, y2]:
                    lab[x2, y2] = n
                    stack.append((x2, y2))
        cells = np.array(cells, dtype=float)
        if len(cells) < 6:
            continue
        D = np.linalg.norm(cells[:, None, :] - C[None, :, :], axis=2)
        dmin = D.min(1)
        kt = int(np.argmax(dmin))
        L = float(dmin[kt]) * s
        if L < min_len * H:
            continue
        tip2 = cells[kt]
        near = cells[dmin <= 1.5]
        if not len(near):
            near = cells[dmin <= dmin.min() + 0.5]
        # the root: where the spike leaves the core, on the side of the tip
        dr = np.linalg.norm(near - tip2, axis=1)
        root2 = near[dr <= dr.min() + max(2.0, 0.5 * r)].mean(0)
        width = float(min(len(near), 4 * r)) * s

        def to3(p2, rad):
            d2 = np.hypot(i - p2[0], j - p2[1])
            m = d2 <= rad
            if not m.any():
                m = d2 <= d2.min() + 1.0
            return P[m].mean(0)

        out.append(dict(tip=to3(tip2, 1.6), root=to3(root2, 2.5), length=L, width=width))
    out.sort(key=lambda d: -d['length'])
    return out


def spike_fin(B, it, H, bv, fi_body, max_spikes=3, log=print):
    """round 8: a plate that LIES on the cage (the wing of a bracer: tangential, its spikes stick out beyond the
    limb).  The body of the plate is in the bake (the cage rings already go around it); every main spike is a
    pyramid on a small quad set into the cage quad under its root (4 frame quads + 4 triangles), its point at the
    real tip of the spike.  One piece with the cage.  -> faces added"""
    V = np.array(B.v)
    spikes = plate_spikes(it['P'], H)
    nf, used, n0 = 0, set(), len(B.f)
    made = []
    for sp in spikes:
        if len(made) >= max_spikes:
            break
        tip, root = sp['tip'], sp['root']
        loc_t, nrm_t, _fk, _d = bv.find_nearest(Vector(tip))
        if loc_t is None or float((Vector(tip) - loc_t).dot(nrm_t)) < 0.008 * H:
            continue                                        # lies on the limb: nothing to see
        # the base: under the root, pulled towards the tip so that the spike leans less
        best = None
        for t_ in (0.0, 0.25, 0.5):
            loc, nrm, fk, _d = bv.find_nearest(Vector(root + (tip - root) * t_))
            if loc is None:
                continue
            g = fi_body[fk]
            if len(B.f[g]) == 4 and g not in used and B.fg[g] == 'body':
                best = (g, np.array(loc), np.array(nrm))
                break
        if best is None:
            continue
        g, foot, nq = best
        f = B.f[g]
        p = [V[k] for k in f]
        es, et = 0.5 * ((p[1] - p[0]) + (p[2] - p[3])), 0.5 * ((p[3] - p[0]) + (p[2] - p[1]))
        ls, lt = float(np.linalg.norm(es)), float(np.linalg.norm(et))
        if min(ls, lt) < 0.008 * H:
            continue
        o_ = 0.25 * (p[0] + p[1] + p[2] + p[3])
        cs, ct = 0.5 + float((foot - o_) @ es) / ls ** 2, 0.5 + float((foot - o_) @ et) / lt ** 2
        w = float(np.clip(sp['width'], 0.014 * H, 0.03 * H))
        hs_, ht_ = float(np.clip(0.5 * w / ls, 0.12, 0.36)), float(np.clip(0.5 * w / lt, 0.12, 0.36))
        cs, ct = float(np.clip(cs, hs_ + 0.1, 0.9 - hs_)), float(np.clip(ct, ht_ + 0.1, 0.9 - ht_))

        def bil(a, b):
            return (p[0] * (1 - a) + p[1] * a) * (1 - b) + (p[3] * (1 - a) + p[2] * a) * b

        base = [bil(cs - hs_, ct - ht_), bil(cs + hs_, ct - ht_), bil(cs + hs_, ct + ht_), bil(cs - hs_, ct + ht_)]
        bc = np.mean(base, axis=0)
        apex = np.array(tip, float)
        # the sides of the pyramid stay outside the cage: the point is lifted along the normal of the quad
        for _ in range(10):
            worst = None
            for b_ in base:
                for t_ in (0.35, 0.6, 0.85, 1.0):
                    sp_ = b_ + (apex - b_) * t_
                    loc, nrm, _fk, _d = bv.find_nearest(Vector(sp_))
                    if loc is None:
                        continue
                    hg = float((Vector(sp_) - loc).dot(nrm)) - 0.004 * H * t_
                    if worst is None or hg < worst:
                        worst = hg
            if worst is None or worst >= 0.0:
                break
            apex = apex + nq * (0.003 * H - worst)
        if float((apex - bc) @ nq) < 0.006 * H:
            continue
        bi = B.add(base)
        ai = B.add([apex])[0]
        for k in range(4):
            k1 = (k + 1) % 4
            B.face([f[k], f[k1], bi[k1], bi[k]], 'body')
            B.face([bi[k], bi[k1], ai], 'body')
            B.seam(bi[k], bi[k1])
        B.seam(bi[0], ai)
        used.add(g)
        made.append(round(sp['length'] / H, 3))
    nf = len(B.f) - n0
    for g in sorted(used, reverse=True):
        del B.f[g]
        del B.fg[g]
    if made:
        log('  fin: plate on the cage, %d spikes (lengths %s H) of %d found' % (len(made), made, len(spikes)))
    return nf


def _fill_poly(B, idx, e3, g):
    """a flat polygon (vertex indices, counter clockwise seen from e3) -> triangles, pairs joined to quads"""
    P = [Vector(B.v[k]) for k in idx]
    tris = []
    for t in mgeo.tessellate_polygon([P]):
        t = list(t)
        n = (P[t[1]] - P[t[0]]).cross(P[t[2]] - P[t[0]])
        if n.length < 1e-14:
            continue
        tris.append(t if float(n.dot(Vector(e3))) > 0 else t[::-1])
    used = [False] * len(tris)
    em = {}
    for ti, t in enumerate(tris):
        for k in range(3):
            em[(t[k], t[(k + 1) % 3])] = (ti, k)
    cand = []
    for ti, t in enumerate(tris):
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]
            o = em.get((b, a))
            if o is None or o[0] <= ti:
                continue
            t2 = tris[o[0]]
            d = t2[(o[1] + 2) % 3]
            quad = [t[(k + 2) % 3], a, d, b]
            Q = [P[x] for x in quad]
            ang = []
            for i in range(4):
                x, y = Q[(i - 1) % 4] - Q[i], Q[(i + 1) % 4] - Q[i]
                if x.length < 1e-12 or y.length < 1e-12 or float(y.cross(x).dot(Vector(e3))) <= 0:
                    ang = None
                    break
                ang.append(math.degrees(x.angle(y)))
            if ang is None or max(ang) > 160:
                continue
            cand.append((max(abs(z - 90) for z in ang), ti, o[0], quad))
    cand.sort(key=lambda c_: c_[0])
    for _c, t1, t2, quad in cand:
        if used[t1] or used[t2]:
            continue
        used[t1] = used[t2] = True
        B.face([idx[k] for k in quad], g)
    for ti, t in enumerate(tris):
        if not used[ti]:
            B.face([idx[k] for k in t], g)


def plate_fin(B, it, H, bv, fi_body, max_pts=10, log=print):
    """round 8: a plate that LIES on the cage and flares off it (the wing of a bracer: tangential, its spikes
    reach beyond the limb).  The cage quads under it are replaced by a plateau: the top is the outline of the
    plate (simplified to max_pts points: the tips of the spikes and the notches between them, each at its real
    place), a band of faces joins the rim of the removed quads to that outline.  One piece with the cage.
    -> faces added (0: nothing built)"""
    V = np.array(B.v)
    P = it['P']
    hs = np.zeros(len(P))
    fk = np.full(len(P), -1)
    nn = np.zeros((len(P), 3))
    for k, p_ in enumerate(P):
        loc, nrm, f_, _d = bv.find_nearest(Vector(p_))
        if loc is not None:
            hs[k] = float((Vector(p_) - loc).dot(nrm))
            fk[k] = fi_body[f_]
            nn[k] = nrm
    c, Vp = _pca(P)
    e1, e3 = Vp[0], Vp[2]
    if float(nn.mean(0) @ e3) < 0:
        e3 = -e3
    e2 = np.cross(e3, e1)
    up = (fk >= 0) & (hs > 0.004 * H)
    if up.sum() < 60:
        return 0
    q = P - c
    u, v, w = q @ e1, q @ e2, q @ e3
    s = 0.0025 * H
    u0, v0 = float(u[up].min()), float(v[up].min())
    gi, gj = np.floor((u - u0) / s).astype(int) + 4, np.floor((v - v0) / s).astype(int) + 4
    mask = np.zeros((gi[up].max() + 9, gj[up].max() + 9), dtype=bool)
    mask[gi[up], gj[up]] = True
    mask = _morph(_morph(mask, 1.5, True), 1.5, False)
    loops = [np.array(L_, float) for L_ in _trace_loops(mask)]
    loops = [L_ for L_ in loops if _loop_area(L_) > 0]
    if not loops:
        return 0
    L = max(loops, key=_loop_area)
    sig = _dp_significance(L)
    order = np.argsort(-sig)
    keep = sorted(int(k) for k in order[:max_pts] if sig[k] >= 1.2)
    if len(keep) < 4:
        return 0
    O2 = np.stack([(L[keep, 0] - 4) * s + u0, (L[keep, 1] - 4) * s + v0], axis=1)        # outline, counter clockwise seen from e3
    # the cage quads under the plate: grown from the one with the most samples above it, the rim stays one loop
    cnt = collections.Counter(int(f_) for f_ in fk[up] if len(B.f[int(f_)]) == 4 and B.fg[int(f_)] == 'body')
    if not cnt:
        return 0
    tot = sum(cnt.values())
    Lk = np.stack([(L[:, 0] - 4) * s + u0, (L[:, 1] - 4) * s + v0], axis=1)

    def under(g):
        cc = V[B.f[g]].mean(0) - c
        return bool(_inside_loops(np.array([[cc @ e1, cc @ e2]]), [Lk])[0])

    def rim_of(reg):
        ed = {}
        for g in reg:
            f = B.f[g]
            for k in range(4):
                ed[(f[k], f[(k + 1) % 4])] = g
        nxt = {a: b for (a, b) in ed if (b, a) not in ed}
        if not nxt:
            return None
        a0 = next(iter(nxt))
        loop, a = [a0], nxt[a0]
        while a != a0 and len(loop) <= len(nxt):
            loop.append(a)
            if a not in nxt:
                return None
            a = nxt[a]
        return loop if len(loop) == len(nxt) and len(set(loop)) == len(loop) else None

    edge_faces = collections.defaultdict(list)
    for g in fi_body:
        f = B.f[g]
        if len(f) == 4 and B.fg[g] == 'body':
            for k in range(4):
                edge_faces[(min(f[k], f[(k + 1) % 4]), max(f[k], f[(k + 1) % 4]))].append(g)
    start = max((g for g in cnt if under(g)), key=cnt.get, default=None)
    if start is None:
        return 0
    reg = [start]
    while len(reg) < 8:
        nb = set()
        for g in reg:
            f = B.f[g]
            for k in range(4):
                for g2 in edge_faces[(min(f[k], f[(k + 1) % 4]), max(f[k], f[(k + 1) % 4]))]:
                    if g2 not in reg and cnt.get(g2, 0) >= 0.05 * tot and under(g2):
                        nb.add(g2)
        nb = [g2 for g2 in sorted(nb, key=lambda g2: -cnt[g2]) if rim_of(reg + [g2]) is not None]
        if not nb:
            break
        reg.append(nb[0])
    rim = rim_of(reg)
    if rim is None:
        return 0
    # the top loop in space: every outline point at the level of the plate there, clear of the cage
    top = []
    for a, b in O2:
        d2 = np.hypot(u - a, v - b)
        m = up & (d2 <= 3.5 * s)
        if not m.any():
            m = up & (d2 <= d2[up].min() + s)
        top.append(c + e1 * a + e2 * b + e3 * float(np.median(w[m])))
    R2 = np.array([[(V[k] - c) @ e1, (V[k] - c) @ e2] for k in rim])
    o2 = R2.mean(0)
    # both loops counter clockwise seen from outside; they are zipped by the angle around the middle of the rim

    def ang(X):
        a = np.arctan2(X[:, 1] - o2[1], X[:, 0] - o2[0])
        k0 = int(np.argmin(a))
        a = np.roll(a, -k0)
        return k0, np.maximum.accumulate(a)

    kr, ar = ang(R2)
    kt, at = ang(O2)
    rim = rim[kr:] + rim[:kr]
    top = top[kt:] + top[:kt]
    N, M = len(rim), len(top)
    for _ in range(6):                      # the top and the band stay outside the cage
        low = 0.0
        for j in range(M):
            for k in (j + 1, j + 2):
                sp_ = (top[j] + top[k % M]) / 2
                loc, nrm, _f, _d = bv.find_nearest(Vector(sp_))
                if loc is not None:
                    low = min(low, float((Vector(sp_) - loc).dot(nrm)) - 0.004 * H)
            loc, nrm, _f, _d = bv.find_nearest(Vector(top[j]))
            if loc is not None:
                low = min(low, float((Vector(top[j]) - loc).dot(nrm)) - 0.005 * H)
        cen = np.mean(top, axis=0)
        loc, nrm, _f, _d = bv.find_nearest(Vector(cen))
        if loc is not None:
            low = min(low, float((Vector(cen) - loc).dot(nrm)) - 0.004 * H)
        if low >= 0:
            break
        top = [t_ + e3 * (0.001 * H - low) for t_ in top]
    n0 = len(B.f)
    ti = B.add(top)
    i = j = 0
    pend = None
    while i < N or j < M:
        an_r = ar[i + 1] if i + 1 < N else ar[0] + 2 * math.pi
        an_t = at[j + 1] if j + 1 < M else at[0] + 2 * math.pi
        if j >= M or (i < N and an_r <= an_t):
            if pend is not None:
                B.face(pend, 'body')
            pend = [rim[i % N], rim[(i + 1) % N], ti[j % M]]
            i += 1
        else:
            if pend is not None:            # base step + top step = one quad
                B.face([pend[0], pend[1], ti[(j + 1) % M], ti[j % M]], 'body')
                pend = None
            else:
                B.face([rim[i % N], ti[(j + 1) % M], ti[j % M]], 'body')
            j += 1
    if pend is not None:
        B.face(pend, 'body')
    _fill_poly(B, ti, e3, 'body')
    for k in range(N):
        B.seam(rim[k], rim[(k + 1) % N])
    for k in range(M):
        B.seam(ti[k], ti[(k + 1) % M])
    B.seam(rim[0], ti[0])
    nf = len(B.f) - n0
    for g in sorted(reg, reverse=True):
        del B.f[g]
        del B.fg[g]
    log('  fin: plate on the cage -> plateau over %d cage quads, outline of %d points (%d faces)' % (len(reg), M, nf))
    return nf


def extrude_fin(B, it, H, log=print):
    """a flat piece that stands on the cage (the flange of a bracer, a fin): a strip of cage quads below it, along
    its length, becomes a tent - one ridge vertex per rung of the strip, on the upper outline of the piece there
    (the real point of the piece: a swept back spike is followed, the ends reach to its tips), two quads per
    segment from the sides of the strip up to the ridge.  A thin blade with the outline of the piece, part of the
    closed cage.  -> faces added (0: no strip found)"""
    orient_builder(B)
    V = np.array(B.v)
    fi_body = [i for i, (f, g) in enumerate(zip(B.f, B.fg)) if g == 'body']
    bv = BVHTree.FromPolygons([tuple(p) for p in V], [tuple(B.f[i]) for i in fi_body])
    P = it['P'][::max(1, len(it['P']) // 900)]
    if os.environ.get('SR_DUMP'):
        import pickle
        DEBUG['_fin_n'] = DEBUG.get('_fin_n', 0) + 1
        pickle.dump(dict(P=it['P'], V=V, F=[B.f[i] for i in fi_body], H=H), open(os.path.join(os.environ['SR_DUMP'], 'fin%d.pkl' % DEBUG['_fin_n']), 'wb'))
    Hs = np.zeros(len(P))
    F = np.full(len(P), -1)
    NN = np.zeros((len(P), 3))
    for k, p in enumerate(P):
        loc, nrm, fk, d = bv.find_nearest(Vector(p))
        if loc is not None:
            Hs[k] = float((Vector(p) - loc).dot(nrm))
            F[k] = fi_body[fk]
            NN[k] = nrm
    ok = F >= 0
    if ok.sum() < 30:
        return 0
    if abs(float((NN[ok] @ _pca(it['P'])[1][2]).mean())) > 0.6:       # the piece lies on the cage (round 8)
        return plate_fin(B, it, H, bv, fi_body, log=log) or spike_fin(B, it, H, bv, fi_body, log=log)
    hmax = float(np.percentile(Hs[ok], 97))
    if hmax < 0.012 * H:
        return 0
    cnt = collections.Counter(int(f) for f, h in zip(F[ok], Hs[ok]) if h > 0.1 * hmax and len(B.f[int(f)]) == 4)
    if not cnt:
        return 0
    c, Vp = _pca(it['P'])
    e1, e3 = Vp[0], Vp[2]
    start = max(cnt, key=cnt.get)
    q = B.f[start]
    # the two rungs of the start quad: the pair of opposite edges that the length of the piece crosses
    m = [(V[q[k]] + V[q[(k + 1) % 4]]) / 2 for k in range(4)]
    k0 = 0 if abs(float((m[2] - m[0]) @ e1)) >= abs(float((m[3] - m[1]) @ e1)) else 1
    edge_faces = collections.defaultdict(list)
    for i in fi_body:
        f = B.f[i]
        for k in range(len(f)):
            edge_faces[(min(f[k], f[(k + 1) % len(f)]), max(f[k], f[(k + 1) % len(f)]))].append(i)
    tot = sum(cnt.values())
    u = P @ e1

    def walk(face, rung):
        """the quads beyond `rung` (a, b: an edge of `face`), while the piece still lies over them"""
        out = []
        for _ in range(3):
            nxt_ = [g for g in edge_faces[(min(rung), max(rung))] if g != face]
            if not nxt_ or len(B.f[nxt_[0]]) != 4 or nxt_[0] in used:
                break
            g = nxt_[0]
            if cnt.get(g, 0) < 0.04 * tot:
                break
            f = B.f[g]
            ka = f.index(rung[0])
            kb = f.index(rung[1])
            # the opposite edge, in the same left / right order
            if (ka + 1) % 4 == kb:
                far = (f[(ka - 1) % 4], f[(kb + 1) % 4])
            else:
                far = (f[(ka + 1) % 4], f[(kb - 1) % 4])
            out.append((g, far))
            used.add(g)
            face, rung = g, far
        return out

    used = {start}
    r_a, r_b = (q[k0], q[(k0 + 1) % 4]), (q[(k0 + 3) % 4], q[(k0 + 2) % 4])        # left / right ends in the same order
    fwd = walk(start, r_b)
    back = walk(start, r_a)
    rungs = [r for _g, r in reversed(back)] + [r_a, r_b] + [r for _g, r in fwd]
    strip = [g for g, _r in reversed(back)] + [start] + [g for g, _r in fwd]
    mids = np.array([(V[a_] + V[b_]) / 2 for a_, b_ in rungs])
    um = mids @ e1
    if um[-1] < um[0]:
        sgn = -1.0
    else:
        sgn = 1.0
    us, ums = u * sgn, um * sgn
    above = ok & (Hs > 0.15 * hmax)
    SUB = 3                                 # ridge points per segment of the strip (round 7: the spikes of the piece are in the geometry)
    st = []                                 # (position along the piece, point on the strip axis, rung direction, rung length, index of the rung or None)
    for k in range(len(rungs)):
        rv = V[rungs[k][1]] - V[rungs[k][0]]
        st.append((ums[k], mids[k], rv, k))
        if k < len(rungs) - 1:
            rv1 = V[rungs[k + 1][1]] - V[rungs[k + 1][0]]
            for j in range(1, SUB):
                t_ = j / float(SUB)
                st.append((ums[k] * (1 - t_) + ums[k + 1] * t_, mids[k] * (1 - t_) + mids[k + 1] * t_, rv * (1 - t_) + rv1 * t_, None))
    ridge = []
    for k, (uk, mk, rv, rk) in enumerate(st):
        lo_ = -np.inf if k == 0 else (st[k - 1][0] + uk) / 2
        hi_ = np.inf if k == len(st) - 1 else (uk + st[k + 1][0]) / 2
        sel = above & (us >= lo_) & (us < hi_)
        if sel.sum() >= 4:
            if k == 0 and (uk - us[sel].min()) > 0.5 * abs(st[1][0] - uk):          # the piece goes on beyond the strip: to its tip
                top = P[sel][np.argsort(us[sel])[:max(3, sel.sum() // 12)]].mean(0)
            elif k == len(st) - 1 and (us[sel].max() - uk) > 0.5 * abs(uk - st[-2][0]):
                top = P[sel][np.argsort(-us[sel])[:max(3, sel.sum() // 12)]].mean(0)
            else:
                top = P[sel][np.argsort(-Hs[sel])[:max(3, sel.sum() // 8)]].mean(0)
            top = top - e3 * float((top - c) @ e3)                  # on the plane of the piece
        else:
            loc, nrm, _fk, _d = bv.find_nearest(Vector(mk))
            top = mk + (np.array(nrm) if loc is not None else np.zeros(3)) * 0.12 * hmax
        rl = float(np.linalg.norm(rv))
        if rl > 1e-12:                      # the ridge stays over the strip and above the cage
            rv = rv / rl
            off = float((top - mk) @ rv)
            top = top - rv * (off - float(np.clip(off, -0.2 * rl, 0.2 * rl)))
        loc, nrm, _fk, _d = bv.find_nearest(Vector(top))
        if loc is not None and float((Vector(top) - loc).dot(nrm)) < 0.008 * H:
            top = np.array(loc + nrm * (0.008 * H))
        ridge.append(B.add([top])[0])
    # the end caps (rung + tip of the blade) must not dip into the cage
    for k, rk in ((0, 0), (len(st) - 1, len(rungs) - 1)):
        a_, b_ = V[rungs[rk][0]], V[rungs[rk][1]]
        for _ in range(8):
            t_ = np.asarray(B.v[ridge[k]], float)
            worst = None
            for sp in ((a_ + b_ + t_) / 3, (a_ + t_) / 2, (b_ + t_) / 2, (a_ + 3 * t_) / 4, (b_ + 3 * t_) / 4, t_):
                loc, nrm, _fk, _d = bv.find_nearest(Vector(sp))
                if loc is None:
                    continue
                hg = float((Vector(sp) - loc).dot(nrm))
                if worst is None or hg < worst[0]:
                    worst = (hg, np.array(nrm))
            if worst is None or worst[0] >= 0.003 * H:
                break
            B.v[ridge[k]] = t_ + worst[1] * 2.0 * (0.004 * H - worst[0])
    n0 = len(B.f)
    # winding from the strip quads (outward); a side of a segment is a fan: the side edge and the SUB + 1 ridge points over it
    for k in range(len(rungs) - 1):
        g = strip[k]
        f = B.f[g]
        a0, b0 = rungs[k]
        a1, b1 = rungs[k + 1]
        rr = ridge[k * SUB:(k + 1) * SUB + 1]
        h_ = len(rr) // 2
        for x0, x1 in ((a0, a1), (b0, b1)):
            i0, i1 = f.index(x0), f.index(x1)
            tris_ = [[x0, x1, rr[h_]]] + [[x1, rr[j + 1], rr[j]] for j in range(h_, len(rr) - 1)] + [[x0, rr[j + 1], rr[j]] for j in range(h_)]
            for t_ in tris_:
                B.face(t_ if (i0 + 1) % 4 == i1 else t_[::-1], 'body')
            B.seam(x0, x1)
    for k, rk, g in ((0, 0, strip[0]), (len(st) - 1, len(rungs) - 1, strip[-1])):
        f = B.f[g]
        a_, b_ = rungs[rk]
        i0, i1 = f.index(a_), f.index(b_)
        B.face([a_, b_, ridge[k]] if (i0 + 1) % 4 == i1 else [b_, a_, ridge[k]], 'body')
        B.seam(a_, b_)
    nf = len(B.f) - n0
    for g in sorted(strip, reverse=True):
        del B.f[g]
        del B.fg[g]
    return nf


def add_hair_flap(B, hf, sk, H, log=print, thick=0.008, clear=0.004):
    """round 9: the hair below the step of the head (build_body) hangs from the rim of the step over the cape: a
    tongue with an outer side (on the hair surface), an inner side (clear of everything built so far: cape, body),
    a bottom rim and two side walls; its faces replace the underside of the step, so it is part of the one closed
    body mesh (group 'body': baked from the hair, skinned to the hair bones).
    hf: A (rim vertices, hair columns in order), B (the vertices of the ring below them), Wa / Wb (the welded
    vertices before the first / after the last column), C (centre of the ring), z_att, Ph (hair samples)"""
    A, Bv, C, z_att, Ph = list(hf['A']), list(hf['B']), np.asarray(hf['C'], float), hf['z_att'], hf['Ph']
    lat, fwd, up = sk.lat, sk.fwd, sk.up
    m = len(A)
    S_ = set(A) | set(Bv) | {hf['Wa'], hf['Wb']}
    under = [i for i, (f, g) in enumerate(zip(B.f, B.fg)) if g == 'body' and set(f) <= S_ and (set(f) & set(A)) and (set(f) & (set(Bv) | {hf['Wa'], hf['Wb']})) and not set(f) <= (set(A) | {hf['Wa'], hf['Wb']})]
    bvh = BVHTree.FromPolygons([tuple(p_) for p_ in B.v], [tuple(f) for f in B.f])
    VA = np.array([B.v[i] for i in A])
    VB = np.array([B.v[i] for i in Bv])
    th = np.array([math.atan2(float((p_ - C) @ fwd), float((p_ - C) @ lat)) for p_ in VA])
    dth = np.unwrap(th)
    wcol = float(np.median(np.abs(np.diff(dth)))) if m > 1 else math.pi / 6
    qh = Ph - C
    ah = np.arctan2(qh @ fwd, qh @ lat)
    rh = np.hypot(qh @ lat, qh @ fwd)
    zb2 = float(VB[:, 2].mean())

    def rad(a_):
        return lat * math.cos(a_) + fwd * math.sin(a_)

    def r_obst(a_, z, r_from):
        """outermost surface built so far around direction a_ at height z (seen from r_from inward)"""
        best = 0.0
        for da in (-0.5, -0.25, 0.0, 0.25, 0.5):
            for dz in (-0.004, 0.0, 0.004):
                d = rad(a_ + da * wcol)
                o = C + up * (z + dz * H - float(C @ up)) + d * r_from
                hit = bvh.ray_cast(Vector(o), Vector(-d), r_from)
                if hit[0] is not None:
                    best = max(best, r_from - hit[3])
        return best

    def r_hair(a_, z):
        sel = (np.abs((ah - a_ + math.pi) % (2 * math.pi) - math.pi) < 0.5 * wcol) & (np.abs(Ph[:, 2] - z) < 0.008 * H)
        return float(np.percentile(rh[sel], 80)) if sel.sum() >= 6 else None

    # how far the hair hangs in every column
    zb = []
    for j in range(m):
        sel = (np.abs((ah - th[j] + math.pi) % (2 * math.pi) - math.pi) < 0.5 * wcol) & (Ph[:, 2] < z_att)
        zb.append(float(np.clip(np.percentile(Ph[sel, 2], 2), z_att - 0.18 * H, z_att - 0.02 * H)) if sel.sum() >= 12 else z_att - 0.02 * H)
    drop = z_att - min(zb)
    R = int(np.clip(round(drop / (0.026 * H)), 1, 4))
    rA = np.hypot((VA - C) @ lat, (VA - C) @ fwd)
    rB = np.hypot((VB - C) @ lat, (VB - C) @ fwd)
    O, I = [], []                           # [row][column] positions, row 0: rim / inner top
    r_prev_o = rA.copy()
    for r in range(R + 1):
        Or, Ir = [], []
        for j in range(m):
            z = zb2 if r == 0 else z_att + (zb[j] - z_att) * r / R
            if r == 0:                      # the top: between the rim and the ring below the step
                ro = max(float(rA[j]), float(rB[j]) + 0.012 * H)
                ri = float(rB[j]) + max(0.004 * H, 0.35 * (ro - float(rB[j])))
                # round 9b: what comes up to right under the step (the top of the cape) stays inside the inner side
                ob_ = max(r_obst(th[j], z - 0.007 * H, ro + 0.08 * H), r_obst(th[j], z - 0.016 * H, ro + 0.08 * H) - 0.006 * H)
                if ob_ > float(rB[j]) + 0.003 * H:
                    ri = max(ri, ob_ + clear * H + 0.035 * ob_)
                    ro = max(ro, ri + thick * H)
                Or.append(C + up * (z_att - float(C @ up)) + rad(th[j]) * ro)
                Ir.append(C + up * (z - float(C @ up)) + rad(th[j]) * ri)
                continue
            ro_surf = r_obst(th[j], z, float(rA[j]) + 0.08 * H)
            sag = 0.035 * max(ro_surf, float(rA[j]))
            ri = ro_surf + clear * H + sag
            rhh = r_hair(th[j], z)
            ro = max(ri + thick * H, rhh if rhh is not None else 0.0)
            ro = min(ro, ri + 0.03 * H)                      # a stray strand does not blow it up
            if r > 0:
                ri = max(ri, ro - 0.016 * H)
            Or.append(C + up * ((z_att if r == 0 else z) - float(C @ up)) + rad(th[j]) * ro)
            Ir.append(C + up * (z - float(C @ up)) + rad(th[j]) * ri)
        O.append(Or)
        I.append(Ir)
    # round 9b: no shelf where the flap meets something that stands off the neck (the top of the cape, the collar): going
    # up from there the outer side comes in by at most `slope` of the height, the rim of the step moves out with it
    slope = 0.7
    for r in range(R - 1, -1, -1):
        for j in range(m):
            lo_, up_ = np.asarray(O[r + 1][j]), np.asarray(O[r][j])
            r_lo = float(np.hypot((lo_ - C) @ lat, (lo_ - C) @ fwd))
            r_up = float(np.hypot((up_ - C) @ lat, (up_ - C) @ fwd))
            need = r_lo - slope * abs(float((up_ - lo_) @ up))
            if need > r_up:
                O[r][j] = up_ + rad(th[j]) * (need - r_up)
                if r > 0:
                    in_ = np.asarray(I[r][j])
                    r_in = float(np.hypot((in_ - C) @ lat, (in_ - C) @ fwd))
                    if need - 0.016 * H > r_in:
                        I[r][j] = in_ + rad(th[j]) * (need - 0.016 * H - r_in)
    # the rim moves out to the outer side of the flap (never inward)
    for j in range(m):
        ro0 = float(np.hypot((O[0][j] - C) @ lat, (O[0][j] - C) @ fwd))
        if ro0 > rA[j]:
            B.v[A[j]] = np.asarray(B.v[A[j]]) + rad(th[j]) * (ro0 - rA[j])
    for i in sorted(under, reverse=True):
        del B.f[i]
        del B.fg[i]
    n0 = len(B.f)
    oi = [A] + [B.add(O[r]) for r in range(1, R + 1)]
    ii = [B.add(I[r]) for r in range(R + 1)]
    for r in range(R):
        for j in range(m - 1):
            B.face([oi[r][j], oi[r][j + 1], oi[r + 1][j + 1], oi[r + 1][j]], 'body')
            B.face([ii[r][j + 1], ii[r][j], ii[r + 1][j], ii[r + 1][j + 1]], 'body')
        for j, sgn in ((0, 1), (m - 1, -1)):          # side walls
            f = [oi[r][j], oi[r + 1][j], ii[r + 1][j], ii[r][j]]
            B.face(f if sgn > 0 else f[::-1], 'body')
            B.seam(oi[r][j], oi[r + 1][j])
    for j in range(m - 1):
        B.face([oi[R][j], oi[R][j + 1], ii[R][j + 1], ii[R][j]], 'body')              # bottom rim
        B.seam(oi[R][j], oi[R][j + 1])
        B.face([ii[0][j], ii[0][j + 1], Bv[j + 1], Bv[j]], 'body')                    # ceiling: inner top -> ring below the step
        B.seam(Bv[j], Bv[j + 1])
    for W, j in ((hf['Wa'], 0), (hf['Wb'], m - 1)):
        B.face([W, Bv[j], ii[0][j]], 'body')
        B.face([W, ii[0][j], A[j]], 'body')
        B.seam(W, Bv[j])
    orient_builder(B)
    log('  hair: flap of %d faces from the rim of the step (%d columns x %d rows), hangs %.3fH .. %.3fH below it, outer side %.3fH .. %.3fH outside what it lies on' % (
        len(B.f) - n0, m, R, (z_att - max(zb)) / H, drop / H,
        min(float(np.linalg.norm(np.asarray(O[r][j]) - np.asarray(I[r][j]))) for r in range(1, R + 1) for j in range(m)) / H,
        max(float(np.linalg.norm(np.asarray(O[r][j]) - np.asarray(I[r][j]))) for r in range(1, R + 1) for j in range(m)) / H))
    hf['faces'] = len(B.f) - n0


def orient_builder(B):
    """the faces of the builder get their outward winding (closed cage: recalc + signed volume)"""
    bm = bmesh.new()
    vs = [bm.verts.new(p) for p in B.v]
    fs = {}
    for i, f in enumerate(B.f):
        try:
            fs[i] = bm.faces.new([vs[k] for k in f])
        except ValueError:
            pass
    bm.verts.index_update()
    bmesh.ops.recalc_face_normals(bm, faces=list(fs.values()))
    _orient_closed(bm)
    for i, bf in fs.items():
        B.f[i] = [v.index for v in bf.verts]
    bm.free()


def _rod_profile(P, H, cage_dist=None, nd=10):
    """slices of a rod along its main axis, from the base (the end at the cage, else the thick end) to the tip
    -> dict(c, e1, e2, e3, t0, L, at(k) -> (centre 2d, half size 2d), nd) or None"""
    c, V = _pca(P)
    e1, e2, e3 = V[0], V[1], np.cross(V[0], V[1])
    q = P - c
    t = q @ e1
    t0, t1 = np.percentile(t, 0.5), np.percentile(t, 99.5)
    L = t1 - t0
    if L < 1e-9:
        return None
    flip = None
    if cage_dist is not None:
        a, b = P[t < t0 + 0.2 * L], P[t > t0 + 0.8 * L]
        if len(a) and len(b):
            da = float(np.mean([cage_dist(p) for p in a[::max(1, len(a) // 30)]]))
            db = float(np.mean([cage_dist(p) for p in b[::max(1, len(b) // 30)]]))
            flip = da > db
    if flip is None:
        r = np.linalg.norm(q - np.outer(t, e1), axis=1)
        flip = np.median(r[t > t0 + 0.75 * L]) > np.median(r[t < t0 + 0.25 * L])
    if flip:
        e1, e3, t = -e1, -e3, -t
        t0, t1 = -t1, -t0
    x, y = q @ e2, q @ e3
    cen, half = [], []
    for k in range(nd):
        m = (t >= t0 + L * k / nd) & (t <= t0 + L * (k + 1) / nd)
        if m.sum() < 5:
            cen.append(None)
            half.append(None)
            continue
        xa, xb, ya, yb = np.percentile(x[m], 6), np.percentile(x[m], 94), np.percentile(y[m], 6), np.percentile(y[m], 94)
        cen.append(np.array([(xa + xb) / 2, (ya + yb) / 2]))
        hx, hy = max(xb - xa, 0.002 * H) / 2, max(yb - ya, 0.002 * H) / 2
        half.append(np.array([max(hx, 0.6 * hy), max(hy, 0.6 * hx)]))
    ok = [k for k in range(nd) if cen[k] is not None]
    if len(ok) < 3:
        return None

    def at(k):
        k = min(ok, key=lambda g: abs(g - k))
        return cen[k], half[k]

    def ring(k, tt=None, shrink=1.0):
        cc, hh = at(k)
        tt = t0 + L * (k + 0.5) / nd if tt is None else tt
        return [c + e1 * tt + e2 * (cc[0] + sx * hh[0] * shrink) + e3 * (cc[1] + sy * hh[1] * shrink) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]

    def point(k, tt):
        cc, _h = at(k)
        return c + e1 * tt + e2 * cc[0] + e3 * cc[1]

    def ringn(k, n, shrink=1.0):
        """n sided ring (ellipse through the box of the slice), counter clockwise seen from the tip"""
        cc, hh = at(k)
        tt = t0 + L * (k + 0.5) / nd
        return [c + e1 * tt + e2 * (cc[0] + 1.12 * hh[0] * shrink * math.cos(a_)) + e3 * (cc[1] + 1.12 * hh[1] * shrink * math.sin(a_))
                for a_ in (np.arange(n) + 0.5) * 2 * math.pi / n]

    def ring_al(k, u, v, sg, shrink=1.0):
        """round 9 ("the fingers are twisted"): 4 sided ring in a frame (u, v) that is the same for all rings of the
        rod (given by the cage quad the rod grows out of); corner i lies at the signs sg[i] -> the edges of the tube
        run straight from the corners of the quad to the tip"""
        k = min(ok, key=lambda g: abs(g - k))
        m = (t >= t0 + L * k / nd) & (t <= t0 + L * (k + 1) / nd)
        a_, b_ = q[m] @ u, q[m] @ v
        ua, ub, va, vb = np.percentile(a_, 6), np.percentile(a_, 94), np.percentile(b_, 6), np.percentile(b_, 94)
        hu, hv = max(ub - ua, 0.002 * H) / 2, max(vb - va, 0.002 * H) / 2
        hu, hv = max(hu, 0.6 * hv), max(hv, 0.6 * hu)
        tt = t0 + L * (k + 0.5) / nd
        o_ = c + e1 * tt + u * (0.5 * (ua + ub) - float(e1 @ u) * tt) + v * (0.5 * (va + vb) - float(e1 @ v) * tt)
        return [o_ + u * (su * hu * shrink) + v * (sv * hv * shrink) for su, sv in sg]

    return dict(c=c, e1=e1, e2=e2, e3=e3, t0=t0, L=L, at=at, ring=ring, ringn=ringn, ring_al=ring_al, point=point, nd=nd)


def attach_rods(B, rods, H, cage_dist, n_rings=3, log=print):
    """fingers / claws / spikes that grow out of the cage: every rod takes the cage quad at its base (the cells of
    the cap of a hand are the finger bases, a side quad the base of the thumb); the quad is replaced by a 4 sided
    tube through n_rings rings (knuckles, they follow the bend of the finger) to a point at the tip.  The tube is
    part of the cage (group 'body').  A rod that finds no quad at first takes the nearest free quad that faces
    it.  Neighbouring tubes are made thinner where they would run into each other.
    -> list of the rods that found no quad at all"""
    orient_builder(B)                       # the normal of a quad is its outward side: a rod never grows through the cage
    profs = {}
    for ri, it in enumerate(rods):
        pr = _rod_profile(it['P'], H, cage_dist)
        if pr is not None:
            profs[ri] = pr

    def scan():
        V_ = np.array(B.v)
        quads_ = [(i, f) for i, (f, g) in enumerate(zip(B.f, B.fg)) if g == 'body' and len(f) == 4]
        QC_ = np.array([V_[f].mean(0) for _i, f in quads_])
        QE_ = np.array([max(np.linalg.norm(V_[f[k]] - V_[f[(k + 1) % 4]]) for k in range(4)) for _i, f in quads_])
        QN_ = []
        for _i, f in quads_:
            n = np.cross(V_[f[2]] - V_[f[0]], V_[f[3]] - V_[f[1]])
            QN_.append(n / max(np.linalg.norm(n), 1e-30))
        return V_, quads_, QC_, QE_, np.array(QN_)

    def strict(pr, QC_, QE_, QN_):
        base = pr['point'](0, pr['t0'])
        hb = float(pr['at'](1)[1].max())
        d = np.linalg.norm(QC_ - base, axis=1)
        al = QN_ @ pr['e1']
        ahead = (QC_ - base) @ pr['e1']
        w_ = QC_ - base
        lat = np.linalg.norm(w_ - np.outer(w_ @ pr['e1'], pr['e1']), axis=1)      # the quad the axis of the finger passes through
        return (d < max(0.04 * H, 3 * hb)) & (al > 0.35) & (QE_ < 8 * hb + 0.01 * H) & (ahead < 0.4 * pr['L']), lat + 0.3 * d + (1 - al) * QE_

    # round 8: fingers side by side that want the same quad (4 fingers on the 3 cells of the cap of a hand) share it:
    # the quad is cut into one cell per finger (the new vertices lie on its two edges along the row of fingers; the
    # neighbours across these edges become a quad + a triangle).  Without this the finger that came last took a
    # side quad and its base ran through the base of its neighbour.
    for _it in range(10):
        V, quads, QC, QE, QN = scan()
        want = {}
        for ri, pr in profs.items():
            okq, cost = strict(pr, QC, QE, QN)
            ii = np.flatnonzero(okq)
            if len(ii):
                want.setdefault(int(ii[np.argmin(cost[ii])]), []).append(ri)
        multi = [(qi, rs) for qi, rs in want.items() if len(rs) > 1 and QE[qi] > 0.012 * H * len(rs)]
        if not multi:
            break
        qi, rs = max(multi, key=lambda t: len(t[1]))
        fi, f = quads[qi]
        bases = np.array([profs[r]['point'](0, profs[r]['t0']) for r in rs])
        u = bases[np.argmax(bases @ _pca(bases)[1][0])] - bases[np.argmin(bases @ _pca(bases)[1][0])] if len(rs) > 2 else bases[1] - bases[0]
        u = u / max(np.linalg.norm(u), 1e-30)
        k0 = max((0, 1), key=lambda k: abs(float((V[f[(k + 1) % 4]] - V[f[k]]) @ u)) + abs(float((V[f[(k + 3) % 4]] - V[f[(k + 2) % 4]]) @ u)))
        a0, a1, b1, b0 = f[k0], f[(k0 + 1) % 4], f[(k0 + 2) % 4], f[(k0 + 3) % 4]      # edges a0-a1 and b0-b1 run along the row
        ax = 0.5 * (V[a1] - V[a0] + V[b1] - V[b0])
        o_ = 0.5 * (V[a0] + V[b0])
        tt = np.sort(np.clip(((bases - o_) @ ax) / max(float(ax @ ax), 1e-30), 0.0, 1.0))
        cuts = [0.5 * (tt[j] + tt[j + 1]) for j in range(len(tt) - 1)]
        n_ = len(cuts)
        cuts = [min(max(c_, (j + 1) * 0.6 / (n_ + 1) + 0.0), 1.0 - (n_ - j) * 0.6 / (n_ + 1)) for j, c_ in enumerate(cuts)]
        ma = B.add([V[a0] + (V[a1] - V[a0]) * c_ for c_ in cuts])
        mb = B.add([V[b0] + (V[b1] - V[b0]) * c_ for c_ in cuts])
        ra, rb = [a0] + ma + [a1], [b0] + mb + [b1]
        was_seam = [(min(x, y), max(x, y)) in B.seams for x, y in ((a0, a1), (b0, b1))]
        for (x, y), ms, ws in (((a0, a1), ma, was_seam[0]), ((b0, b1), mb, was_seam[1])):
            for gi, g_ in enumerate(B.f):
                if gi == fi or x not in g_ or y not in g_:
                    continue
                n = len(g_)
                ix = g_.index(x)
                if g_[(ix + 1) % n] == y:
                    ring = g_[ix + 1:] + g_[:ix + 1]          # y ... x, the new vertices follow x towards y
                    mid = ms
                elif g_[(ix - 1) % n] == y:
                    ring = g_[ix:] + g_[:ix]                  # x ... y
                    mid = ms[::-1]
                else:
                    continue
                # ring = [p0 ... pn-1] with the cut edge between ring[-1] and ring[0]; `mid` runs from ring[-1] to ring[0]
                grp = B.fg[gi]
                if n == 4:
                    p0, p1, p2, p3 = ring
                    chain = [p3] + list(mid) + [p0]
                    B.f[gi] = [chain[0], chain[1], p1, p2]
                    for j in range(1, len(chain) - 1):
                        B.face([chain[j], chain[j + 1], p1], grp)
                else:
                    chain = [ring[-1]] + list(mid) + [ring[0]]
                    B.f[gi] = [chain[-2], chain[-1]] + ring[1:-1] if n > 3 else [chain[-2], chain[-1], ring[1]]
                    for j in range(len(chain) - 2):
                        B.face([chain[j], chain[j + 1], ring[1] if n == 3 else ring[-2]], grp)
                break
            if ws:
                full = [x] + ms + [y]
                for j in range(len(full) - 1):
                    B.seam(full[j], full[j + 1])
        B.f[fi] = [ra[0], ra[1], rb[1], rb[0]]
        for j in range(1, len(ra) - 1):
            B.face([ra[j], ra[j + 1], rb[j + 1], rb[j]], 'body')
        log('  fingers: cage quad cut into %d cells (rods %s, quad centre %s)' % (len(rs), rs, np.round(QC[qi] / H, 3)))
    orient_builder(B)
    V, quads, QC, QE, QN = scan()
    used_q, done, remove, tubes = set(), set(), [], []
    for relaxed in (False, True):
        cand = []
        for ri, pr in profs.items():
            if ri in done:
                continue
            base = pr['point'](0, pr['t0'])
            hb = float(pr['at'](1)[1].max())
            d = np.linalg.norm(QC - base, axis=1)
            al = QN @ pr['e1']                  # the rod leaves the cage through this quad (round 4 took |.|: fingers inside out)
            ahead = (QC - base) @ pr['e1']      # the quad lies at / before the base, never beyond the middle of the rod
            if relaxed:
                okq = (d < max(0.07 * H, 5 * hb)) & (al > 0.1) & (ahead < 0.5 * pr['L'])
            else:
                okq = (d < max(0.04 * H, 3 * hb)) & (al > 0.35) & (QE < 8 * hb + 0.01 * H) & (ahead < 0.4 * pr['L'])
            for qi in np.flatnonzero(okq):
                if qi not in used_q:
                    w_ = QC[qi] - base
                    lat = float(np.linalg.norm(w_ - pr['e1'] * float(w_ @ pr['e1'])))
                    cand.append((float(lat + 0.3 * d[qi] + (1 - al[qi]) * QE[qi]), ri, int(qi)))
        cand.sort()
        for cost, ri, qi in cand:
            if ri in done or qi in used_q:
                continue
            pr = profs[ri]
            fi, f = quads[qi]
            f_start = len(B.f)
            # the neighbours of a taken quad stay whole (two tubes side by side share an edge, never a face)
            qc = QC[qi]
            nd = pr['nd']
            ks = [int(round(nd * (j + 1) / (n_rings + 1.0))) for j in range(n_rings)]
            # round 9 ("the fingers are twisted"): all rings are 4 sided, in one frame that is given by the cage
            # quad (corner i of every ring belongs to corner i of the quad): no band of triangles, no rotation
            # between the rings, the 4 edges run straight from the quad to the tip
            NS = 4
            w_ = np.array([V[i_] for i_ in f]) - qc
            w_ = w_ - np.outer(w_ @ pr['e1'], pr['e1'])
            u_ = w_[1] + w_[2] - w_[0] - w_[3]
            if np.linalg.norm(u_) < 1e-9:
                u_ = pr['e2']
            u_ = u_ / np.linalg.norm(u_)
            v_ = np.cross(pr['e1'], u_)
            ang = np.arctan2(w_ @ v_, w_ @ u_)
            combos = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            sg = None
            for rev_ in (False, True):
                for s_ in range(4):
                    order = [combos[(s_ + (-i_ if rev_ else i_)) % 4] for i_ in range(4)]
                    dd_ = [abs((math.atan2(o_[1], o_[0]) - ang[i_] + math.pi) % (2 * math.pi) - math.pi) for i_, o_ in enumerate(order)]
                    if sg is None or sum(x_ * x_ for x_ in dd_) < sg[0]:
                        sg = (sum(x_ * x_ for x_ in dd_), order)
            sg = sg[1]
            rings = []
            for k in ks:
                R = pr['ring_al'](k, u_, v_, sg, shrink=FINGER_SHRINK[min(len(rings), len(FINGER_SHRINK) - 1)])
                if (np.mean(R, axis=0) - qc) @ pr['e1'] > 0.12 * pr['L']:
                    rings.append(R)
            tip = pr['point'](nd - 1, pr['t0'] + pr['L'])
            if not rings:
                rings = [pr['ring_al'](nd // 2, u_, v_, sg, shrink=0.8)]
            idx = [B.add(R) for R in rings]
            B.strip(list(f), idx[0], 'body')
            for a_, b_ in zip(idx[:-1], idx[1:]):
                B.strip(a_, b_, 'body')
            apex = B.add([tip])[0]
            last = idx[-1]
            for k in range(NS):
                B.face([last[k], last[(k + 1) % NS], apex], 'body')
            for k in range(4):
                B.seam(f[k], f[(k + 1) % 4])
            B.seam(f[0], idx[0][0])
            for a_, b_ in zip(idx[:-1], idx[1:]):
                B.seam(a_[0], b_[0])
            B.seam(last[0], apex)
            used_q.add(qi)
            done.add(ri)
            remove.append(fi)
            tubes.append(idx)
            rods[ri]['faces'] = NS * len(idx) + NS
            rods[ri]['attached'] = True
            rods[ri]['new_faces'] = [tuple(f_) for f_ in B.f[f_start:]]
    # tubes side by side: thinner where they would run into each other
    min_gap = None
    for _pass in range(3):
        min_gap = None
        for i1 in range(len(tubes)):
            for i2 in range(i1 + 1, len(tubes)):
                for r1 in tubes[i1]:
                    for r2 in tubes[i2]:
                        p1, p2 = np.array([B.v[k] for k in r1]), np.array([B.v[k] for k in r2])
                        c1, c2 = p1.mean(0), p2.mean(0)
                        dd = float(np.linalg.norm(c1 - c2))
                        if dd > 0.06 * H:
                            continue
                        u = (c2 - c1) / max(dd, 1e-12)
                        e1_, e2_ = float(np.abs((p1 - c1) @ u).max()), float(np.abs((p2 - c2) @ u).max())
                        gap = dd - e1_ - e2_
                        min_gap = gap if min_gap is None else min(min_gap, gap)
                        if gap < 0.005 * H:
                            sc = max(0.4, (dd - 0.006 * H) / max(e1_ + e2_, 1e-12))
                            for r_, c_ in ((r1, c1), (r2, c2)):
                                for k in r_:
                                    w_ = B.v[k] - c_
                                    B.v[k] = c_ + w_ - u * float(w_ @ u) * (1 - sc)
    if tubes:
        log('  fingers: %d tubes welded to the cage, smallest gap between neighbouring rings %.4fH' % (len(tubes), (min_gap if min_gap is not None else 0.06 * H) / H))
    DEBUG['finger_min_gap_by_H'] = None if min_gap is None else float(min_gap / H)
    for fi in sorted(remove, reverse=True):
        del B.f[fi]
        del B.fg[fi]
    return [it for ri, it in enumerate(rods) if ri not in done]


def band_profile(P, o, n, e1, H, nb=72, mirror=False):
    """a sheet that runs around an axis (a hat brim around the crown, a collar, a skirt plate): per angle the
    profile of the sheet from its inner to its outer edge, 3 stations.  -> dict or None when it is no ring"""
    e2 = np.cross(n, e1)
    q = P - o
    z = q @ n
    x, y = q @ e1, q @ e2
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    b = np.minimum(((th + math.pi) / (2 * math.pi) * nb).astype(int), nb - 1)
    Rn = np.full((nb, 3), np.nan)
    Zn = np.full((nb, 3), np.nan)
    for j in range(nb):
        m = b == j
        if m.sum() < 8:
            continue
        rr, zz = r[m], z[m]
        A = np.vstack([rr - rr.mean(), zz - zz.mean()])
        w, Vv = np.linalg.eigh(A @ A.T)
        ax = Vv[:, 1]
        if ax[0] < 0:
            ax = -ax
        if abs(ax[0]) < 0.25:                 # the sheet stands upright here: outward = away from the inner edge
            ax = ax if ax[1] > 0 else -ax
        t = A.T @ ax
        ta, tb = np.percentile(t, 3), np.percentile(t, 97)
        for s, tf in enumerate((ta, (ta + tb) / 2, tb)):
            mm = np.abs(t - tf) < max(0.15 * (tb - ta), 1e-9)
            if mm.sum() >= 2:
                Rn[j, s], Zn[j, s] = np.median(rr[mm]), np.median(zz[mm])
    have = ~np.isnan(Rn[:, 2])
    if mirror:                                # bin j <-> nb - 1 - j shifted: theta -> pi - theta
        mj = (nb // 2 - 1 - np.arange(nb)) % nb
        both = have & have[mj]
        only = ~have & have[mj]
        Rn[only], Zn[only] = Rn[mj][only], Zn[mj][only]
        # where both sides exist the wider one is the complete one (the other is hidden behind something)
        take = both & (Rn[mj][:, 2] > Rn[:, 2] * 1.1)
        Rn[take], Zn[take] = Rn[mj][take], Zn[mj][take]
        have = ~np.isnan(Rn[:, 2])
    if have.mean() < 0.85:
        return None
    for s in range(3):
        Rn[:, s], Zn[:, s] = _circ_fill(Rn[:, s]), _circ_fill(Zn[:, s])
    if np.median(Rn[:, 0]) < 0.3 * np.median(Rn[:, 2]) or np.median(Rn[:, 2] - Rn[:, 0]) + np.median(np.abs(Zn[:, 2] - Zn[:, 0])) < 0.01 * H:
        return None
    for s in range(3):                        # sampling noise (the inner edge lies in the body: smoothed more)
        for _ in range(4 if s == 0 else 1):
            Rn[:, s] = 0.25 * np.roll(Rn[:, s], 1) + 0.5 * Rn[:, s] + 0.25 * np.roll(Rn[:, s], -1)
            Zn[:, s] = 0.25 * np.roll(Zn[:, s], 1) + 0.5 * Zn[:, s] + 0.25 * np.roll(Zn[:, s], -1)
    if mirror:
        mj = (nb // 2 - 1 - np.arange(nb)) % nb
        Rn, Zn = (Rn + Rn[mj]) / 2, (Zn + Zn[mj]) / 2
    return dict(o=o, n=n, e1=e1, e2=e2, nb=nb, R=Rn, Z=Zn, mirror=mirror)


def band_contains(bp, pts, tol):
    """which points lie in the band (between its inner and outer edge, within tol of its surface)"""
    q = pts - bp['o']
    z = q @ bp['n']
    x, y = q @ bp['e1'], q @ bp['e2']
    r = np.hypot(x, y)
    nb = bp['nb']
    j = np.minimum(((np.arctan2(y, x) + math.pi) / (2 * math.pi) * nb).astype(int), nb - 1)
    R, Z = bp['R'][j], bp['Z'][j]
    out = np.zeros(len(pts), dtype=bool)
    for s in (0, 1):
        a = np.stack([R[:, s], Z[:, s]], 1)
        d = np.stack([R[:, s + 1] - R[:, s], Z[:, s + 1] - Z[:, s]], 1)
        p = np.stack([r, z], 1) - a
        t = np.clip((p * d).sum(1) / np.maximum((d * d).sum(1), 1e-30), 0, 1)
        out |= np.linalg.norm(p - d * t[:, None], axis=1) < tol
    return out


def build_band(B, name, bp, quads, H, thick=0.008, log=print):
    """the band as one continuous strip of quads around the axis: K columns where the outline changes most
    (corners), 2 quads from the inner to the outer edge on the upper and on the lower side; a closed solid
    (the sides share the rims).  -> number of faces"""
    nb = bp['nb']
    R, Z = bp['R'], bp['Z']
    th = -math.pi + (np.arange(nb) + 0.5) * 2 * math.pi / nb
    X = np.stack([R * np.cos(th)[:, None], R * np.sin(th)[:, None], Z], 2)          # (nb, 3, 3)
    K = int(np.clip(round(quads / 4.0), 8, 28))
    mj = (nb // 2 - 1 - np.arange(nb)) % nb
    if bp['mirror']:
        j0 = nb // 4 - 1                                  # theta = -90 and +90 degrees lie on the mirror plane (between two bins)
        sel = sorted({j0, mj[j0], (j0 + nb // 2) % nb, mj[(j0 + nb // 2) % nb]})
    else:
        sel = [0, nb // 2]
    gap = max(1, int(0.35 * nb / K))
    while len(sel) < K:
        best = None
        for k in range(nb):
            if k in sel or min(min(abs(k - s), nb - abs(k - s)) for s in sel) < gap:
                continue
            lo_ = max((s for s in sel if s < k), default=sel[-1] - nb)
            hi_ = min((s for s in sel if s > k), default=sel[0] + nb)
            t = (k - lo_) / (hi_ - lo_)
            e = float(np.linalg.norm(X[k] - (X[lo_ % nb] * (1 - t) + X[hi_ % nb] * t), axis=1).max())
            if best is None or e > best[0]:
                best = (e, k)
        if best is None:
            break
        sel = sorted(set(sel + [best[1]] + ([int(mj[best[1]])] if bp['mirror'] else [])))
    n0 = len(B.f)
    o, n, e1, e2 = bp['o'], bp['n'], bp['e1'], bp['e2']
    # one closed solid (round 5: the two unconnected layers were rejected): the upper and the lower side share
    # the inner and the outer rim, the middle row is `thick` apart
    cols = []
    for j in sel:
        pts = []
        for s_, sgn in ((0, 0.0), (1, 1.0), (2, 0.0), (1, -1.0)):
            a, b = max(s_ - 1, 0), min(s_ + 1, 2)
            dr, dz = R[j, b] - R[j, a], Z[j, b] - Z[j, a]
            ln = math.hypot(dr, dz) or 1.0
            nr, nz = -dz / ln, dr / ln               # normal of the profile in the (r, z) half plane
            # not thicker than the band is wide there (a steep narrow corner: the two sides would cross)
            wd_ = min(math.hypot(R[j, 1] - R[j, 0], Z[j, 1] - Z[j, 0]), math.hypot(R[j, 2] - R[j, 1], Z[j, 2] - Z[j, 1]))
            th_ = min(1.5 * thick * H, 0.6 * wd_)
            rr, zz = R[j, s_] + sgn * nr * th_ / 2, Z[j, s_] + sgn * nz * th_ / 2
            pts.append(o + rr * (math.cos(th[j]) * e1 + math.sin(th[j]) * e2) + zz * n)
        cols.append(B.add(pts))                      # inner rim, middle above, outer rim, middle below
    # round 8: the middle row is moved apart along the normal of the band AS BUILT (between the chosen columns), not
    # along the normal of its profile at that angle: where the brim turns up steeply between two columns the two
    # normals differ and the upper and the lower side touched / crossed
    nk = len(cols)
    mid = [0.5 * (np.asarray(B.v[c_[1]]) + np.asarray(B.v[c_[3]])) for c_ in cols]
    for k, c_ in enumerate(cols):
        m_, i_, o_ = mid[k], np.asarray(B.v[c_[0]]), np.asarray(B.v[c_[2]])
        mp, mn = mid[(k - 1) % nk], mid[(k + 1) % nk]
        nv = np.cross(o_ - m_, mn - m_) + np.cross(mn - m_, i_ - m_) + np.cross(i_ - m_, mp - m_) + np.cross(mp - m_, o_ - m_)
        old = np.asarray(B.v[c_[1]]) - np.asarray(B.v[c_[3]])
        th_ = float(np.linalg.norm(old))
        if np.linalg.norm(nv) < 1e-20 or th_ < 1e-12:
            continue
        nv = nv / np.linalg.norm(nv)
        if float(nv @ old) < 0:
            nv = -nv
        B.v[c_[1]] = m_ + nv * th_ / 2
        B.v[c_[3]] = m_ - nv * th_ / 2
    def crossing(T1, T2):
        """do the triangles of the upper side (T1) pass through those of the lower side (T2)?"""
        for A_, B_ in ((T1, T2), (T2, T1)):
            for t in A_:
                for k in range(3):
                    p0, p1 = np.asarray(B.v[t[k]]), np.asarray(B.v[t[(k + 1) % 3]])
                    for t2 in B_:
                        if t[k] in t2 and t[(k + 1) % 3] in t2:
                            continue
                        q0, q1, q2 = (np.asarray(B.v[x]) for x in t2)
                        nrm = np.cross(q1 - q0, q2 - q0)
                        d0, d1 = float((p0 - q0) @ nrm), float((p1 - q0) @ nrm)
                        if d0 * d1 >= 0 or abs(d0 - d1) < 1e-30:
                            continue
                        tt = d0 / (d0 - d1)
                        if tt < 1e-4 or tt > 1 - 1e-4:
                            continue
                        x = p0 + (p1 - p0) * tt
                        inside_ = True
                        for i_ in range(3):
                            e0, e1_ = (q0, q1, q2)[i_], (q0, q1, q2)[(i_ + 1) % 3]
                            if float(np.cross(e1_ - e0, x - e0) @ nrm) <= 1e-9 * float(nrm @ nrm):
                                inside_ = False
                                break
                        if inside_:
                            return True
        return False

    for a, b in zip(cols, cols[1:] + cols[:1]):
        # round 8: where the brim is twisted between two columns the two sides of a half (inner / outer) are cut
        # along the diagonal that keeps them apart (a quad is cut along either diagonal by whoever triangulates it)
        for up_, dn_ in (([a[0], b[0], b[1], a[1]], [a[0], a[3], b[3], b[0]]), ([a[1], b[1], b[2], a[2]], [a[3], a[2], b[2], b[3]])):
            # up_ = [p, q, r, s], dn_ lists the same rim with the middle row below
            x0, x1, x2, x3 = up_
            y = [v_ for v_ in dn_ if v_ not in up_]          # the two lower middle vertices
            ya, yb = (y[0], y[1]) if y[0] in a else (y[1], y[0])
            lo_of = {a[1]: a[3], b[1]: b[3]}
            g = [lo_of.get(v_, v_) for v_ in up_]           # the lower side, same order as the upper one
            opt = []
            for d in (0, 1):
                tu = [[up_[d], up_[d + 1], up_[d + 2]], [up_[d], up_[d + 2], up_[(d + 3) % 4]]]
                tl = [[g[d], g[d + 2], g[d + 1]], [g[d], g[(d + 3) % 4], g[d + 2]]]
                opt.append((tu, tl))
            bad = [crossing(tu, tl) for tu, tl in opt]
            tw = 0.0                                        # how far the quads are from flat, against the thickness there
            for qd in (up_, g):
                q_ = [np.asarray(B.v[v_]) for v_ in qd]
                nq_ = np.cross(q_[2] - q_[0], q_[3] - q_[1])
                tw = max(tw, abs(float((q_[1] - q_[0]) @ nq_)) / max(float(np.linalg.norm(nq_)), 1e-30))
            thick_here = 0.25 * (np.linalg.norm(np.asarray(B.v[a[1]]) - np.asarray(B.v[a[3]])) + np.linalg.norm(np.asarray(B.v[b[1]]) - np.asarray(B.v[b[3]])))
            if not bad[0] and not bad[1] and tw < 1.0 * thick_here:
                B.face(up_, name)
                B.face(dn_, name)
            else:
                tu, tl = opt[0] if not bad[0] else opt[1]
                for t in tu + tl:
                    B.face(t, name)
        B.seam(a[0], b[0])
        B.seam(a[2], b[2])
    for k0, k1 in ((0, 1), (1, 2), (2, 3), (3, 0)):
        B.seam(cols[0][k0], cols[0][k1])
    B.keep_winding.add(name)
    return len(B.f) - n0


def ribbon_profile(P, H, cage_dist, e3=None, seed=4):
    """a long thin sheet that curls (a feather, a strap, a ribbon): its centre line from the end at the body,
    by the distance along the sheet (shortest paths in the neighbour graph of the samples).
    -> dict(g per sample, length, mean width, e3) or None"""
    rng = np.random.default_rng(seed)
    S = P[rng.choice(len(P), size=min(len(P), 450), replace=False)]
    kd = KDTree(len(S))
    for i, p in enumerate(S):
        kd.insert(Vector(p), i)
    kd.balance()
    d0 = np.array([cage_dist(p) for p in S])
    src = int(d0.argmin())
    import heapq
    g = np.full(len(S), np.inf)
    g[src] = 0.0
    hq = [(0.0, src)]
    nbrs = [[(i2, d) for _co, i2, d in kd.find_n(Vector(p), 9)[1:]] for p in S]
    while hq:
        d, i = heapq.heappop(hq)
        if d > g[i]:
            continue
        for j, w in nbrs[i]:
            if d + w < g[j]:
                g[j] = d + w
                heapq.heappush(hq, (d + w, j))
    fin = np.isfinite(g)
    if fin.sum() < 30:
        return None
    for i in np.flatnonzero(~fin):            # separate cards: from the nearest reached sample
        j = int(np.linalg.norm(S[fin] - S[i], axis=1).argmin())
        g[i] = g[fin][j] + float(np.linalg.norm(S[fin][j] - S[i]))
    gp = np.array([g[kd.find(Vector(p))[1]] for p in P])
    Lg = float(np.percentile(gp, 99))
    if e3 is None:
        e3 = _pca(P)[1][2]
    return dict(g=gp, L=Lg, e3=e3)


def ribbon_stations(P, rp, K, H):
    """K + 1 stations along the ribbon: (centre, left node, right node)"""
    g, Lg, e3 = rp['g'], rp['L'], rp['e3']
    cs = []
    edges = np.linspace(0, Lg, K + 1)
    win = 0.6 * Lg / K
    for k, ge in enumerate(edges):
        m = np.abs(g - ge) < (win if 0 < k < K else 0.5 * win)
        if m.sum() < 4:
            m = np.abs(g - ge) < 2 * win
        cs.append(P[m].mean(0) if m.any() else None)
    ok = [k for k in range(K + 1) if cs[k] is not None]
    if len(ok) < 2:
        return None
    for k in range(K + 1):
        if cs[k] is None:
            cs[k] = cs[min(ok, key=lambda j: abs(j - k))]
    out = []
    prev = None
    for k, ge in enumerate(edges):
        a, b = cs[max(k - 1, 0)], cs[min(k + 1, K)]
        tg = b - a
        tg = tg - e3 * float(tg @ e3)
        if np.linalg.norm(tg) < 1e-9:
            continue
        tg /= np.linalg.norm(tg)
        nrm = np.cross(e3, tg)
        if prev is not None and float(nrm @ prev) < 0:
            nrm = -nrm
        prev = nrm
        m = np.abs(g - ge) < (win if 0 < k < K else 0.5 * win)
        if m.sum() < 4:
            m = np.abs(g - ge) < 2 * win
        off = (P[m] - cs[k]) @ nrm
        lo_, hi_ = (np.percentile(off, 4), np.percentile(off, 96)) if m.sum() >= 4 else (-0.002 * H, 0.002 * H)
        if hi_ - lo_ < 0.004 * H:
            mid = (lo_ + hi_) / 2
            lo_, hi_ = mid - 0.002 * H, mid + 0.002 * H
        hgt = float(np.median((P[m] - cs[k]) @ e3)) if m.any() else 0.0
        base = cs[k] + e3 * hgt
        out.append((base, base + nrm * lo_, base + nrm * hi_))
    return out if len(out) >= 2 else None


def build_ribbon(B, name, P, rp, quads, H, thick=0.005, log=print):
    """a curled long sheet as a tapered strip along its centre line: a closed solid, 3 quads per segment.
    -> number of faces"""
    st = None
    K = int(np.clip(round(quads), 3, 10))
    st = ribbon_stations(P, rp, K, H)
    if st is None:
        return 0
    e3 = rp['e3']
    n0 = len(B.f)
    # a closed solid with a triangular cross-section (round 5: no two unconnected layers): flat below, a ridge above
    rows = []
    for k, (_c, a, b) in enumerate(st):
        tg = st[min(k + 1, len(st) - 1)][0] - st[max(k - 1, 0)][0]
        nl = np.cross(b - a, tg)
        ln_ = float(np.linalg.norm(nl))
        nl = nl / ln_ if ln_ > 1e-12 else e3
        if float(nl @ e3) < 0:
            nl = -nl
        wd_ = float(np.linalg.norm(b - a))
        rows.append(B.add([a, b, (a + b) / 2 + nl * min(thick * H, 0.6 * max(wd_, 0.002 * H))]))
    for r0, r1 in zip(rows[:-1], rows[1:]):
        B.face([r0[0], r0[2], r1[2], r1[0]], name)
        B.face([r0[2], r0[1], r1[1], r1[2]], name)
        B.face([r0[1], r0[0], r1[0], r1[1]], name)
        B.seam(r0[0], r1[0])
        B.seam(r0[1], r1[1])
    B.face([rows[0][0], rows[0][1], rows[0][2]], name)
    B.face([rows[-1][0], rows[-1][2], rows[-1][1]], name)
    B.keep_winding.add(name)
    return len(B.f) - n0


def poly_opacity(src, high, alpha):
    """mean alpha of the base colour texture over every polygon of `high` with a transparent material (1 elsewhere)"""
    me = src.data
    out = np.ones(len(high.poly_index))
    if not alpha.any() or not me.uv_layers.active:
        return out
    uv = np.empty(len(me.loops) * 2, dtype=np.float32)
    me.uv_layers.active.data.foreach_get('uv', uv)
    uv = uv.reshape(-1, 2)
    imgs = {}
    for k in np.flatnonzero(alpha):
        p = me.polygons[int(high.poly_index[k])]
        mi = p.material_index
        if mi not in imgs:
            A = None
            b = next((n for n in me.materials[mi].node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if b is not None and b.inputs['Alpha'].is_linked:
                n = b.inputs['Alpha'].links[0].from_node
                if n.type == 'TEX_IMAGE' and n.image and n.image.size[0] > 0:
                    w, h = n.image.size
                    px = np.empty(w * h * 4, dtype=np.float32)
                    n.image.pixels.foreach_get(px)
                    A = px.reshape(h, w, 4)[..., 3]
            imgs[mi] = A
        A = imgs[mi]
        if A is None:
            continue
        u = uv[p.loop_start:p.loop_start + p.loop_total]
        pts = np.vstack([u, u.mean(0, keepdims=True), (u + u.mean(0)) / 2])
        h, w = A.shape
        out[k] = float(A[np.clip((pts[:, 1] % 1 * h).astype(int), 0, h - 1), np.clip((pts[:, 0] % 1 * w).astype(int), 0, w - 1)].mean())
    return out


class _HH:
    def __init__(self, H):
        self.H = H


def _split_by_bones(sk, ks, pdom):
    """a cluster of polygons -> sub clusters by bone chain (the fingers of a hand): bones are joined with their
    parent when the parent is in the cluster too and does not branch there"""
    bones = set(int(b) for b in np.unique(pdom[ks]) if b >= 0)
    if len(bones) < 2:
        return [ks]
    root = {}
    for b in bones:
        r = b
        while True:
            p = sk.parent[r]
            if p < 0 or p not in bones or sum(1 for k in sk.kids[p] if k in bones) >= 2:
                break
            r = p
        root[b] = r
    out = collections.defaultdict(list)
    for k in ks:
        out[root.get(int(pdom[k]), -1)].append(int(k))
    return [np.array(v) for v in out.values()]


def find_protrusions(sk, high, me, owners, own, alpha, log=print):
    """thin free standing pieces of the body (see free_standing) that can be modelled on their own:
    -> list of dict(kind 'sheet' | 'rod', polys, P (samples), area, ext)"""
    H = high.H
    body_i = [owners.index(n) for n in ['torso'] + [l.name for l in sk.legs + sk.arms]]
    body = np.isin(own, body_i)
    D = core_size(sk, high, owners, own)
    D[~body] = 0
    seed, thin = free_standing(high, 0.55 * D * ~alpha, np.maximum(D, 0.05 * H))
    seed |= alpha & body                    # cards with a transparent texture (feathers, fringes) are free standing
    thin |= alpha & body
    # a body whose surface is mostly "thin and free" is a hollow shell (low poly sources): nothing protrudes
    for i in body_i:
        m = own == i
        a = float(high.poly_area[m].sum())
        log('  %s: %.1f%% of the surface is thin and free standing' % (owners[i], 100 * float(high.poly_area[m & seed].sum()) / max(a, 1e-30)))
        if a > 0 and float(high.poly_area[m & seed].sum()) > 0.35 * a:
            seed[m] = False
    # a free standing piece is the whole connected thin region around the free polygons (the part of a hat brim
    # next to the crown has the crown on one side); a thin region that is free only at a small part is not
    fs = seed.copy()
    cen = np.zeros((len(own), 3))
    np.add.at(cen, high.tri_poly, high.co[high.tris].mean(1) * high.tri_area[:, None])
    cen /= np.maximum(high.poly_area, 1e-30)[:, None]
    grow = float(GROW)
    for ks in poly_clusters(high, me, thin & body):
        sd = ks[seed[ks]]
        if not len(sd) or len(sd) == len(ks):
            continue
        kd = KDTree(len(sd))
        for i, k in enumerate(sd):
            kd.insert(Vector(cen[k]), i)
        kd.balance()
        add = [k for k in ks[~seed[ks]] if kd.find(Vector(cen[k]))[2] < grow * D[k]]
        fs[add] = True
    vflag = np.zeros(len(me.vertices), dtype=bool)
    for _ in range(2):                      # close small gaps (rim polygons whose normal lies in the sheet)
        for k in np.flatnonzero(fs):
            vflag[list(me.polygons[int(high.poly_index[k])].vertices)] = True
        for k in np.flatnonzero(body & ~fs):
            vs = list(me.polygons[int(high.poly_index[k])].vertices)
            if vflag[vs].sum() * 3 >= 2 * len(vs):
                fs[k] = True
    DEBUG['fs'], DEBUG['seed'], DEBUG['thin'] = fs.copy(), seed, thin
    items = []

    def classify(ks):
        a = float(high.poly_area[ks].sum())
        if a < 0.0004 * H * H:
            return None
        P = sample_polys(high, ks, int(np.clip(a / (H * H) * 4e5, 600, 6000)))
        if len(P) < 50:
            return None
        c, V, L = _extents(P)
        if L[0] < 0.025 * H:
            return None
        if L[1] < 0.5 * L[0] and L[2] > 0.45 * L[1] and L[1] < 0.05 * H:
            kind = 'rod'
        elif L[2] < 0.5 * L[1] or L[2] < 0.015 * H:
            kind = 'sheet'
        else:
            return None                      # a blob: stays in the bake of the body
        return dict(kind=kind, polys=ks, P=P, area=a, ext=L, centre=c)

    for ks0 in poly_clusters(high, me, fs):
        for ks in _split_by_bones(sk, ks0, sk.pdom):
            it = classify(ks)
            if it is None and not seed[ks].all() and seed[ks].any():
                # with the thin surface around it the piece is no sheet / rod: only what is free standing itself
                # counts (the thin surface next to it still goes with it: it is not part of the cage)
                sub = [classify(ks1) for ks1 in poly_clusters(high, me, np.isin(np.arange(len(fs)), ks) & seed)]
                sub = [x for x in sub if x is not None]
                if sub:
                    big = max(sub, key=lambda x: x['area'])
                    rest = np.setdiff1d(ks, np.concatenate([x['polys'] for x in sub]))
                    if len(rest) and float(high.poly_area[rest].sum()) <= 1.5 * big['area']:
                        big['polys'] = np.concatenate([big['polys'], rest])
                        big['P'] = sample_polys(high, big['polys'], len(big['P']))
                    items += sub
            elif it is not None:
                items.append(it)
    # what is most of the surface of a limb bone (a thin shoe, a flat hand) is the limb itself, not something on it
    nbn = len(sk.names)
    dom_area = np.bincount(sk.pdom[body & (sk.pdom >= 0)], weights=high.poly_area[body & (sk.pdom >= 0)], minlength=nbn)
    core = {b for l in sk.legs + sk.arms for b in l.core}
    out = []
    for it in items:
        pd = sk.pdom[it['polys']]
        ok = pd >= 0
        if ok.any():
            ba = np.bincount(pd[ok], weights=high.poly_area[it['polys']][ok], minlength=nbn)
            b = int(ba.argmax())
            if b in core and ba[b] > 0.4 * dom_area[b]:
                continue
        out.append(it)
    items = out
    return items, thin & body


DEBUG = {}
FINGER_SIDES = 4    # cross-section of a finger tube (round 9: 4, in the frame of the cage quad)
FINGER_SHRINK = (0.92, 0.86, 0.74)    # corners of the rings of a finger tube / the box of the slice
CLOSE_K = float(os.environ.get("SR_CLOSE_K", 1.5))    # round 9: closing radius of a coarse weapon outline / its tolerance
HAIR_STEP = True    # round 9: the hair (part of the head) ends in an overhang, the ring below is fitted without it
HAIR_CAPE_TOP = float(os.environ.get("SR_HAIR_CAPE_TOP", 0.026))    # ... and a cape goes on upward to at most this far (in H) below the step (None: up to the collar as before)
HAIR_PCT0 = 12      # round 9c: ... 12% without a cape that comes up to the collar (round 9; the high step made ailisi worse)
HAIR_PCT = float(os.environ.get("SR_HAIR_PCT", 45))       # ... at the height below which this share of the hair samples of the back lies
END_BOOST = float(os.environ.get("SR_END_BOOST", 1.5))    # round 9b: weight of the outline points at the limb ends of a coarse weapon outline
LIMB_STRIP = int(os.environ.get("SR_LIMB_STRIP", 7))    # round 9d: rungs of the swept limb of a coarse mirrored weapon outline (0: off)
LIMB_OTHER = float(os.environ.get("SR_LIMB_OTHER", 0.45))    # ... weight of the other outline points of the limb
SMOOTH_K = float(os.environ.get("SR_SMOOTH_K", 0.8))    # round 9c: smoothing of a coarse weapon outline / its closing radius
SMOOTH_TIP = float(os.environ.get("SR_SMOOTH_TIP", 80.0))    # ... convex corners sharper than this (degrees) stay
END_FROM = float(os.environ.get("SR_END_FROM", 0.5))
BOOT_PIVOT = float(os.environ.get("SR_BOOT_PIVOT", 0.035))    # round 9c: the axis of the fan lies this far (H) outside the instep crease (0.014: the rings were pinched there)
BOOT_NOSE = (0.8, 0.72)    # round 9d: width / height of the small ring in front of the toe ring (None: off)
BOOT_SLICES = [0.3, 0.6, 0.8, 0.92]    # round 9c: slices of the foot between the crease and the tip, the last one is the toe ring
BOOT_TOE = (0.82, 0.86)    # ... the toe ring is at least this wide / high relative to the slice before it (a round, not a pointed toe)
BOOT = os.environ.get("SR_BOOT", "1") == "1"         # round 9: a foot on the ground is a fan of rings around the instep (shaft, heel, slices, toe)
RODS = True         # fingers / claws grown out of the cage quads (round 6b: grouped to at most 4 per hand, all welded to the palm, never free rods)
GROW = 0.3          # find_protrusions: thin polygons this close (in chain diameters) to a free standing one belong to it
BODY_TIERS = [
    dict(Nt=8, hole_cols=2, hole_rows=1, crotch=2, mids=0, torso_rings=10, end_rings=2),
    dict(Nt=8, hole_cols=2, hole_rows=2, crotch=2, mids=0, torso_rings=12, end_rings=2),
    dict(Nt=8, hole_cols=2, hole_rows=2, crotch=2, mids=1, torso_rings=14, end_rings=2),
    dict(Nt=8, hole_cols=2, hole_rows=2, crotch=2, mids=1, torso_rings=18, end_rings=3),
    dict(Nt=12, hole_cols=2, hole_rows=2, crotch=2, mids=1, torso_rings=14, end_rings=2),
    dict(Nt=12, hole_cols=2, hole_rows=2, crotch=2, mids=1, torso_rings=18, end_rings=3),
    dict(Nt=12, hole_cols=2, hole_rows=2, crotch=2, mids=2, torso_rings=18, end_rings=3),
    dict(Nt=12, hole_cols=2, hole_rows=2, crotch=4, mids=2, torso_rings=20, end_rings=3),
    dict(Nt=12, hole_cols=2, hole_rows=2, crotch=4, mids=3, torso_rings=22, end_rings=3),
    dict(Nt=16, hole_cols=2, hole_rows=2, crotch=4, mids=3, torso_rings=22, end_rings=3),
    dict(Nt=16, hole_cols=4, hole_rows=2, crotch=4, mids=3, torso_rings=24, end_rings=3),
]


def build_character(src, faces, arm, budget, exclude_bones=(), cloth_budget=None, cloth_share=0.39, reach=0.035, protrusions=True, log=print):
    """see _build_character.  When the sheets need much less than their budget (a small apron: an arc sheet is not
    meshed much finer than the body), the character is built again with the rest given to the body."""
    r = _build_character(src, faces, arm, budget, exclude_bones, cloth_budget, cloth_share, reach, protrusions, 0, log)
    info = r[2]
    spare = info['cloth_budget'] - info['tris']['cloth'] - info['tris']['sheets']
    if spare > 0.1 * (info['body_budget'] + info['cloth_budget']):
        log('  the sheets use %d of their %d triangles: body again with %d more' % (info['cloth_budget'] - spare, info['cloth_budget'], spare))
        r = _build_character(src, faces, arm, budget, exclude_bones, cloth_budget, cloth_share, reach, protrusions, spare, log)
    return r


def _build_character(src, faces, arm, budget, exclude_bones=(), cloth_budget=None, cloth_share=0.39, reach=0.035, protrusions=True, body_extra=0, log=print):
    """src: welded source object, faces: its polygons that belong to the character.
    budget: triangles of the whole character, or of the body alone (cage, fingers / spikes, compact extras) when
    cloth_budget is given: that of all sheets (cape, skirt, hair as arc grids; hat brim, fins, feathers as flat
    sheets).  Without it the sheets get cloth_share of the budget when the model has cloth, else what the flat
    sheets need (at most 15%).
    -> Builder (all faces tagged with a group), {group: [polygon indices of src]} (bake / skin sources), info, skeleton"""
    high = High(src, faces)
    H = high.H
    sk = analyse_skeleton(arm, high, exclude_bones, log)
    owners, own = face_owners(sk, high)
    total = high.area
    me = src.data
    alpha = np.array([retopo.material_is_alpha(me.materials[m]) if m < len(me.materials) else False for m in high.poly_mat])
    # small extra groups are part of what they hang on
    keep_extra = []
    zsh0 = float(np.mean([l.joints[0][2] for l in sk.arms])) if sk.arms else float(high.lo[2] + 0.8 * H)
    hair_polys = np.zeros(len(own), dtype=bool)
    sk.hair_polys = hair_polys              # round 9: what was merged into the head (build_body: the step under the hair)
    for e in sk.extra:
        i = owners.index(e['name'])
        a = float(high.poly_area[(own == i) & ~alpha].sum())
        # hair (round 6: "not layered, one piece with the head"): an extra group on the torso chain that lies above
        # the shoulders is part of the head / neck volume; the torso rings are fitted around it, the strands are baked
        zz_ = high.pts[np.isin(high.pt_poly, np.flatnonzero(own == i)), 2]
        if e['host'] == 'torso' and len(zz_) > 20 and np.median(zz_) > zsh0 + 0.02 * H and np.percentile(zz_, 3) > zsh0 - 0.12 * H:
            log('  %s: above the shoulders (%.2fH .. %.2fH) -> part of the head' % (e['name'], (np.percentile(zz_, 3) - high.lo[2]) / H, (np.percentile(zz_, 97) - high.lo[2]) / H))
            hair_polys |= own == i
            own[own == i] = owners.index('torso')
            continue
        if a < 0.005 * total:
            own[own == i] = owners.index(e['host'])
        else:
            keep_extra.append((e, i, a))
    info = {'height': round(H, 4), 'extras': [], 'dropped': [], 'protrusions': []}
    if os.environ.get('SR_DUMP'):
        np.savez(os.path.join(os.environ['SR_DUMP'], 'pts.npz'), pts=high.pts, own=own[high.pt_poly], hair=hair_polys[high.pt_poly], H=H, lo=high.lo, owners=np.array(owners))
    body_i = [owners.index(n) for n in ['torso'] + [l.name for l in sk.legs + sk.arms]]
    # ---- extra groups: sheet (arc grid) or compact (prism)
    tor = high.pts[(own == 0)[high.pt_poly]]
    extras = []
    for e, i, a in keep_extra:
        ks = np.flatnonzero((own == i) & ~alpha)
        pmask = np.isin(high.pt_poly, ks)
        P, Nn = high.pts[pmask], high.pt_n[pmask]
        zr = (np.percentile(P[:, 2], 5), np.percentile(P[:, 2], 95))
        tz = tor[(tor[:, 2] > zr[0]) & (tor[:, 2] < zr[1])]
        tz = tz if len(tz) > 50 else tor
        axis_p = np.zeros(3)
        axis_p[sk.ax] = sk.c0
        axis_p = axis_p + sk.fwd * float(np.median(tz @ sk.fwd))
        st = arc_stats(P, Nn, axis_p, sk)
        c, V = _pca(P)
        ext = np.percentile((P - c) @ V[0], 99) - np.percentile((P - c) @ V[0], 1)
        sheet = st['spread'] / math.sqrt(a / 2) < 0.25 and ext > 0.15 * H and a > 0.015 * total and e['host'] == 'torso'
        log('  %s: area %.1f%%, extent %.2fH, radial spread / size %.2f, arc %.0f deg, inward %.2f -> %s' % (
            e['name'], 100 * a / total, ext / H, st['spread'] / math.sqrt(a / 2), math.degrees(2 * math.pi - st['gap']), st['inward'], 'arc' if sheet else 'prism'))
        extras.append(dict(name=e['name'], kind='arc' if sheet else 'prism', polys=ks, P=P, st=st, axis_p=axis_p, area=a,
                           w=math.sqrt(a) * (1.4 if sheet and st['inward'] > 0.25 else 1.0)))
    arcs = [x for x in extras if x['kind'] == 'arc']
    # round 9c: a cape whose top lies around the shoulders will go on up to the collar: the hair then needs the high step
    sk.cape_to_collar = any(zsh0 - 0.15 * H < float(np.percentile(x['P'][:, 2], 99.3)) < zsh0 + 0.08 * H and x['st']['gap'] >= math.radians(50) for x in arcs)
    prisms = [x for x in extras if x['kind'] == 'prism']
    # ---- thin free standing pieces of the body: modelled on their own, not part of the cage
    prot, thin_body = find_protrusions(sk, high, me, owners, own, alpha, log) if protrusions else ([], np.zeros(len(own), dtype=bool))
    if not RODS:                            # round 6: fingers / claws stay in the cage (a mitten, as the artists build it): no tubes stuck on the hand
        prot = [it for it in prot if it['kind'] != 'rod']
    own_core = own.copy()
    for it in prot:
        own_core[it['polys']] = -1
    cell = 0.022 * H
    for it in prot:
        if it['kind'] == 'rod':
            it['want'] = 12 if it['ext'][0] < 0.04 * H else 20
        else:
            it['want'] = 4 * float(np.clip(it['area'] / 2 / (cell * cell), 1, 60))
    # ---- budgets: body = cage + rods + compact extras;  cloth = arc sheets (cape, skirt, hair) + flat sheets
    sheet_want0 = sum(it['want'] for it in prot if it['kind'] == 'sheet')
    if cloth_budget is None:
        cloth_budget = budget * cloth_share if arcs else min(sheet_want0, 0.15 * budget)
        body_budget = budget - cloth_budget
    else:
        body_budget = budget
    body_budget, cloth_budget = body_budget + body_extra, max(cloth_budget - body_extra, 0)
    info['body_budget'], info['cloth_budget'] = round(body_budget), round(cloth_budget)
    rod_want = sum(it['want'] for it in prot if it['kind'] == 'rod')
    if rod_want > 0.2 * body_budget:
        for it in prot:
            if it['kind'] == 'rod':
                it['want'] = 12
        rod_want = sum(it['want'] for it in prot if it['kind'] == 'rod')
    cage_budget = body_budget - rod_want - 28 * len(prisms)
    # ---- body cage: the densest tier that fits; then the protrusions that really stand out of it.  Second pass:
    # what lies on the cage after all is part of the cage fit again and its triangles go to the cage
    assigned = np.zeros(len(high.poly_area), dtype=bool)
    group_polys = {'body': None}
    cand = list(prot)
    for it in cand:
        it['want0'] = it['want']
    for rnd in range(2):
        B = binfo = None
        lo_t, hi_t = 0, len(BODY_TIERS) - 1          # the tiers grow monotonously: bisection
        while lo_t <= hi_t:
            mid_t = (lo_t + hi_t) // 2
            try:
                Bt, bt = build_body(sk, high, owners, own_core, BODY_TIERS[mid_t], log)
            except ValueError:
                if mid_t == 0:
                    raise
                hi_t = mid_t - 1
                continue
            if sum(len(f) - 2 for f in Bt.f) <= 1.03 * cage_budget or mid_t == 0:
                B, binfo = Bt, bt
                lo_t = mid_t + 1
            else:
                hi_t = mid_t - 1
        if B is None:
            B, binfo = build_body(sk, high, owners, own_core, BODY_TIERS[0], log)
        body_tris = sum(len(f) - 2 for f in B.f)
        log('  cage: %d triangles (budget %d), %s' % (body_tris, cage_budget, binfo['params']))
        bvh = _body_bvh(B)

        def outside(P):
            out = np.zeros(len(P))
            for k, p in enumerate(P):
                loc, nrm, _i, d = bvh.find_nearest(Vector(p))
                if loc is not None and (Vector(p) - loc).dot(nrm) > 0:
                    out[k] = d
            return out

        def inside(p):
            loc, nrm, _i, d = bvh.find_nearest(Vector(p))
            return loc is not None and (Vector(p) - loc).dot(nrm) < -0.004 * H

        kept, lost = [], []
        for it in cand:
            d = outside(it['P'][::max(1, len(it['P']) // 400)])
            it['out'] = float(np.percentile(d, 90))
            it['want'] = it['want0']
            if it['kind'] == 'sheet':            # what stands out further gets a finer grid
                it['want'] *= float(np.clip(it['out'] / (0.03 * H), 0.8, 1.3))
                if float(high.poly_area[it['polys']][alpha[it['polys']]].sum()) > 0.5 * it['area']:
                    it['want'] *= 0.4            # transparent cards: the texture has the outline
            stands = it['out'] >= 0.015 * H and (it['kind'] != 'rod' or it['out'] >= 0.35 * it['ext'][0])      # a strand that hangs along the body is no rod
            (kept if stands else lost).append(it)
        if rnd == 1 or not lost:
            break
        cand = kept
        own_core = own.copy()
        for it in kept:
            own_core[it['polys']] = -1
        cage_budget = body_budget - sum(it['want'] for it in kept if it['kind'] == 'rod') - 28 * len(prisms)
    for it in prot:
        if not any(it is x for x in kept):
            info['dropped'].append({'kind': it['kind'], 'area_pct': round(100 * it['area'] / total, 2), 'out_by_H': round(it.get('out', 0.0) / H, 3)})
    info['body'] = binfo
    info['body_tris'] = body_tris
    kept.sort(key=lambda it: -it['area'])
    rods = [it for it in kept if it['kind'] == 'rod']
    sheets = [it for it in kept if it['kind'] == 'sheet']
    def cage_dist(p):
        return bvh.find_nearest(Vector(p))[3] or 0.0

    cen_all = np.zeros((len(own), 3))
    np.add.at(cen_all, high.tri_poly, high.co[high.tris].mean(1) * high.tri_area[:, None])
    cen_all /= np.maximum(high.poly_area, 1e-30)[:, None]
    # ---- a cape whose top is hidden under a collar / capelet goes on up to the neck (round 5: "the cape is short")
    zsh = float(np.mean([l.joints[0][2] for l in sk.arms])) if sk.arms else float(high.lo[2] + 0.8 * H)
    pnrm = np.zeros((len(own), 3))
    np.add.at(pnrm, high.tri_poly, high.tri_n * high.tri_area[:, None])
    pnrm /= np.maximum(np.linalg.norm(pnrm, axis=1, keepdims=True), 1e-30)
    tor_nohair = high.pts[(own == 0)[high.pt_poly] & ~hair_polys[high.pt_poly]] if hair_polys.any() else tor
    for x in arcs:
        x['zone'] = None
        st = x['st']
        zt0 = float(np.percentile(x['P'][:, 2], 99.3))
        if not (zsh - 0.15 * H < zt0 < zsh + 0.08 * H) or st['gap'] < math.radians(50):
            continue
        relp = st['rel'][x['P'][:, 2] > zt0 - 0.06 * H]
        if len(relp) < 30:
            continue
        lo_t, hi_t = float(np.percentile(relp, 4)), float(np.percentile(relp, 96))
        q = tor - x['axis_p']
        rel_t = (np.arctan2(q @ sk.fwd, q @ sk.lat) - st['centre'] + math.pi) % (2 * math.pi) - math.pi
        r_t = np.hypot(q @ sk.lat, q @ sk.fwd)
        mid_a, half_a = (lo_t + hi_t) / 2, (hi_t - lo_t) / 2
        mm = np.abs(rel_t - mid_a) < 0.5 * half_a
        zs = np.arange(zt0, zsh + 0.14 * H, 0.004 * H)
        rr = np.full(len(zs), np.nan)
        for k, zc in enumerate(zs):
            m2 = mm & (np.abs(tor[:, 2] - zc) < 0.006 * H)
            if m2.sum() >= 5:
                rr[k] = np.percentile(r_t[m2], 85)
        if np.isnan(rr).all():
            continue
        rr = _fill1(rr)
        k0 = int(np.searchsorted(zs, max(zt0, zsh - 0.03 * H)))
        if k0 >= len(zs):
            continue
        neck = float(rr[k0:].min())
        kk = k0 + int(np.argmax(rr[k0:] <= 1.3 * neck))
        zt = float(zs[kk])
        if binfo.get('hair_step_z') is not None and HAIR_CAPE_TOP is not None:
            # round 9: the hair hangs over the top of the cape.  A cape that goes on up to the collar under it would
            # push the hair far off the back (the cape lies on the cage, the hair on the cape): it ends below the hair
            zt = min(zt, binfo['hair_step_z'] - HAIR_CAPE_TOP * H)
        if zt - zt0 < 0.015 * H:
            continue
        cq = cen_all - x['axis_p']
        crel = (np.arctan2(cq @ sk.fwd, cq @ sk.lat) - st['centre'] + math.pi) % (2 * math.pi) - math.pi
        crad = cq - np.outer(cq @ sk.up, sk.up)
        crad /= np.maximum(np.linalg.norm(crad, axis=1, keepdims=True), 1e-30)
        zone = (cen_all[:, 2] > zt0 - 0.015 * H) & (cen_all[:, 2] < zt + 0.01 * H) & (crel > lo_t) & (crel < hi_t)
        x['zone'] = dict(pts=tor_nohair, z0=zt0, z1=zt, lo=lo_t, hi=hi_t, mask=zone,
                         polys=np.flatnonzero(zone & (own == 0) & ~hair_polys & ~alpha & ((pnrm * crad).sum(1) > 0.0)))
        log('  %s: top at %.3fH, collar at %.3fH (shoulders %.3fH): the sheet goes on up to it' % (x['name'], (zt0 - high.lo[2]) / H, (zt - high.lo[2]) / H, (zsh - high.lo[2]) / H))
        # flat pieces under it (the tips of a capelet) are covered by it: they stay in the bake
        for it in list(sheets):
            if zone[it['polys']].mean() > 0.6 and it['out'] < 0.06 * H:
                sheets.remove(it)
                kept[:] = [k_ for k_ in kept if k_ is not it]
                info['dropped'].append({'kind': 'sheet (under the cape top)', 'area_pct': round(100 * it['area'] / total, 2), 'out_by_H': round(it['out'] / H, 3)})
    taken = np.zeros(len(own), dtype=bool)
    for it in kept:
        taken[it['polys']] = True
    for it in sheets:
        c, V = it['centre'], _pca(it['P'])[1]
        it['shape'] = 'grid'
        across = abs(float(c[sk.ax]) - sk.c0) < 0.04 * H and abs(float(V[2] @ sk.lat)) < 0.3
        e3 = V[2] if float(V[2] @ sk.up) >= 0 else -V[2]
        e1 = sk.lat - e3 * float(sk.lat @ e3)
        if np.linalg.norm(e1) > 0.3:
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(e3, e1)
            # 1) a sheet that runs around an axis (hat brim around the crown): one continuous band
            q = it['P'] - c
            o = c + e2 * float((np.percentile(q @ e2, 2) + np.percentile(q @ e2, 98)) / 2)
            if across:
                o[sk.ax] = sk.c0
            else:
                o = o + e1 * float((np.percentile(q @ e1, 2) + np.percentile(q @ e1, 98)) / 2)
            bp = band_profile(it['P'], o, e3, e1, H, mirror=across) if inside(o) else None      # the axis runs through the body
            if bp is not None:
                it['shape'], it['band'] = 'band', bp
                # the rest of the same sheet that was not found as free standing (hidden behind hair, a feather)
                more = np.flatnonzero(thin_body & ~taken)
                more = more[band_contains(bp, cen_all[more], 0.012 * H)] if len(more) else more
                r_in = np.median(bp['R'][:, 0])
                more = more[np.linalg.norm((cen_all[more] - o) - np.outer((cen_all[more] - o) @ e3, e3), axis=1) > 1.03 * r_in] if len(more) else more
                if len(more):
                    it['polys'] = np.concatenate([it['polys'], more])
                    taken[more] = True
                    it['P'] = sample_polys(high, it['polys'], 6000)
                    it['band'] = band_profile(it['P'], o, e3, e1, H, mirror=across) or bp
                continue
        # 2) a long sheet that curls (feather): a strip along its centre line
        rp = ribbon_profile(it['P'], H, cage_dist)
        if rp is not None:
            width = it['area'] / 2 / max(rp['L'], 1e-9)
            if rp['L'] > 2.5 * width and rp['L'] > 0.06 * H:
                it['shape'], it['ribbon'] = 'ribbon', rp
                it['want'] = 6 * float(np.clip(rp['L'] / max(1.3 * width, 0.02 * H), 3, 8))
                continue
        # 3) a flat piece: grid (mirror symmetric when it lies across the mirror plane)
        if across:
            cc = c.copy()
            cc[sk.ax] = sk.c0
            it['frame'] = (cc, e1, np.cross(e3, e1))
    sheet_want = sum(it['want'] for it in sheets)
    # the flat sheets share the cloth budget with the arc sheets (not more than 45% of it)
    left_s = min(sheet_want, 0.45 * cloth_budget) if arcs else min(1.3 * sheet_want, cloth_budget)

    # ---- flat pieces that stand on the body (flanges of a bracer, plates): one piece with the cage (round 6) -
    # the cage quads below them are extruded up to them.  Cards with a transparent texture (feathers) stay sheets
    fins = [it for it in sheets if it['shape'] != 'band' and float(high.poly_area[it['polys']][alpha[it['polys']]].sum()) <= 0.5 * it['area']]
    nfin = 0
    for it in fins:
        sheets[:] = [x_ for x_ in sheets if x_ is not it]
        kept[:] = [x_ for x_ in kept if x_ is not it]
        # round 8: a fin that comes out wrong (a face of it points into the cage: it folded into the limb) is taken
        # back - the piece stays in the bake
        snap_ = (len(B.v), [list(f_) for f_ in B.f], list(B.fg), set(B.seams))
        old_f = {tuple(f_) for f_ in B.f}
        got_ = extrude_fin(B, it, H, log)
        if got_:
            orient_builder(B)
            new_f = [f_ for f_, g_ in zip(B.f, B.fg) if g_ == 'body' and tuple(f_) not in old_f and tuple(f_[::-1]) not in old_f]
            old_any = {frozenset(f_) for f_ in old_f}
            new_f = [f_ for f_ in new_f if frozenset(f_) not in old_any]
            nbad = _parity_bad(B, 'body', new_f, H)
            if nbad:
                del B.v[snap_[0]:]
                B.f[:] = snap_[1]
                B.fg[:] = snap_[2]
                B.seams.clear()
                B.seams.update(snap_[3])
                log('  fin taken back: %d of its %d faces point into the cage' % (nbad, len(new_f)))
                got_ = 0
        nfin += bool(got_)
        info['protrusions'].append({'kind': 'fin (extruded from the cage)' if got_ else 'fin (baked)', 'faces': got_, 'area_pct': round(100 * it['area'] / total, 2),
                                    'out_by_H': round(it['out'] / H, 3), 'centre_H': [round(float(x_), 2) for x_ in (it['centre'] - high.lo) / H],
                                    'extent_H': [round(float(x_), 3) for x_ in it['ext'] / H]})
    if fins:
        log('  fins: %d of %d extruded from the cage' % (nfin, len(fins)))
        bvh = _body_bvh(B)
    sheet_want = sum(it['want'] for it in sheets)
    left_s = min(sheet_want, 0.45 * cloth_budget) if arcs else min(1.3 * sheet_want, cloth_budget)

    def make_sheets(Bx, scale, lg=lambda *a: None):
        """-> triangles per sheet.  All are closed solids (round 5): band, ribbon, relief solid"""
        out = []
        for it in sheets:
            n0 = sum(len(f) - 2 for f in Bx.f)
            if it['shape'] == 'band':
                build_band(Bx, 'sheets', it['band'], it['want'] * scale / 2, H, log=lg)
            elif it['shape'] == 'ribbon':
                build_ribbon(Bx, 'sheets', it['P'], it['ribbon'], it['want'] / 6, H, log=lg)
            else:
                fr = it.get('frame')
                tm = np.isin(high.tri_poly, it['polys'])
                nt = add_relief(Bx, 'sheets', high.co, high.tris[tm], max(10, int(round(it['want'] * scale))), H, res=112, wall=0.007, min_thick=0.005,
                                close=0.004, blur=0.003, frame=None if fr is None else (fr[0], fr[2], fr[1]), symmetric=fr is not None, log=lg)
                if not nt:
                    build_sheet(Bx, 'sheets', it['P'], max(1.0, it['want'] * scale / 4), H, frame=fr, inside=inside, symmetric=fr is not None, log=lg)
            out.append(sum(len(f) - 2 for f in Bx.f) - n0)
        return out

    scale = float(np.clip(left_s / max(sheet_want, 1e-9), 0.2, 2.0))
    got = 0
    for k in range(4):                      # the size of a sheet is not exact: adjust to what is left
        got = sum(make_sheets(Builder(), scale))
        if not got or 0.9 * left_s <= got <= 1.05 * left_s or (got < left_s and scale >= 2.0) or scale <= 0.2:
            break
        scale = float(np.clip(scale * left_s / got, 0.2, 2.0))
    while sheets and got > 1.2 * left_s + 8:           # still too many (every sheet has a minimum): the smallest go
        info['dropped'].append({'kind': 'sheet (budget)', 'area_pct': round(100 * sheets[-1]['area'] / total, 2), 'out_by_H': round(sheets[-1]['out'] / H, 3)})
        sheets.pop()
        got = sum(make_sheets(Builder(), scale))
    kept = [it for it in kept if it['kind'] == 'rod' or any(it is x for x in sheets)]
    # fingers / claws grow out of the cage (the cap cells of a hand are their bases); what finds no quad is a free rod
    # round 6b: a hand has at most 4 finger tubes (the two fingers that lie closest and run alongside are one, like
    # the hand the artists build); what finds no quad stays in the bake of the palm - never a rod of its own
    for l in sk.arms + sk.legs:
        a_, b_ = np.asarray(l.joints[-2], float), np.asarray(l.joints[-1], float)
        def near_end(it):
            t_ = float(np.clip((it['centre'] - a_) @ (b_ - a_) / max(float((b_ - a_) @ (b_ - a_)), 1e-12), 0, 1.5))
            return float(np.linalg.norm(it['centre'] - (a_ + (b_ - a_) * t_))) < 0.09 * H
        grp = [it for it in rods if near_end(it)]
        while len(grp) > 5:
            best = None
            for i1 in range(len(grp)):
                for i2 in range(i1 + 1, len(grp)):
                    ax1, ax2 = _pca(grp[i1]['P'])[1][0], _pca(grp[i2]['P'])[1][0]
                    if abs(float(ax1 @ ax2)) < 0.8:
                        continue
                    d_ = float(np.linalg.norm(grp[i1]['centre'] - grp[i2]['centre']))
                    if best is None or d_ < best[0]:
                        best = (d_, i1, i2)
            if best is None:
                break
            x1, x2 = grp[best[1]], grp[best[2]]
            x1['polys'] = np.concatenate([x1['polys'], x2['polys']])
            x1['P'] = np.concatenate([x1['P'], x2['P']])
            x1['area'] += x2['area']
            x1['centre'], _V, x1['ext'] = _extents(x1['P'])
            grp[:] = [r_ for r_ in grp if r_ is not x2]
            rods[:] = [r_ for r_ in rods if r_ is not x2]
            kept[:] = [r_ for r_ in kept if r_ is not x2]
    # round 8: a finger whose tube comes out wrong (faces that point into the cage: its base ran through a neighbour
    # or through the hand) is taken back and stays in the bake; the others are attached again without it
    rods_try = list(rods)
    for _try in range(8):
        snap_ = (len(B.v), [list(f_) for f_ in B.f], list(B.fg), set(B.seams))
        for it in rods_try:
            it.pop('new_faces', None)
            it.pop('attached', None)
        free = attach_rods(B, rods_try, H, cage_dist, log=log)
        orient_builder(B)
        nb_ = [(_parity_bad(B, 'body', it['new_faces'], H), k_) for k_, it in enumerate(rods_try) if it.get('new_faces')]
        nb_ = sorted((t_ for t_ in nb_ if t_[0]), reverse=True)
        if not nb_ and any(it.get('new_faces') for it in rods_try):
            # round 9: a tube can also bury faces of the hand next to it (its own faces are fine): any face of the
            # cage that points inward near a tube counts against the nearest tube
            cs_ = []
            if _parity_bad(B, 'body', None, H, centres=cs_):
                att_ = [(k_, it) for k_, it in enumerate(rods_try) if it.get('new_faces')]
                hit_ = collections.Counter()
                for c_ in cs_:
                    dd_ = [float(np.linalg.norm(it['P'] - c_, axis=1).min()) for _k, it in att_]
                    j_ = int(np.argmin(dd_))
                    if dd_[j_] < 0.05 * H:
                        hit_[att_[j_][0]] += 1
                nb_ = sorted(((n_, k_) for k_, n_ in hit_.items()), reverse=True)
        if not nb_ or _try == 7:
            break
        # the worst one only (and whatever is as bad: its mirror twin): a broken tube also spoils the test of its neighbours
        bad_rods = [rods_try[k_] for n_, k_ in nb_ if n_ == nb_[0][0]]
        del B.v[snap_[0]:]
        B.f[:] = snap_[1]
        B.fg[:] = snap_[2]
        B.seams.clear()
        B.seams.update(snap_[3])
        log('  %d finger tubes taken back (faces pointing into the cage)' % len(bad_rods))
        rods_try = [it for it in rods_try if not any(it is b_ for b_ in bad_rods)]
        free = None
    free = [it for it in rods if not it.get('attached')]
    for it in free:
        it['faces'] = 0
    if free:
        log('  %d rods found no cage quad: left in the bake' % len(free))
    info['finger_min_gap_by_H'] = DEBUG.get('finger_min_gap_by_H')
    for it, nf in zip(sheets, make_sheets(B, scale, log)):
        it['faces'] = nf
    opacity = None
    for it in kept:
        nf = it['faces']
        g = 'rods' if it['kind'] == 'rod' else 'sheets'
        if nf:
            if not it.get('attached'):            # an attached rod is part of the cage: baked and skinned with the body
                src_polys = it['polys']
                if alpha[src_polys].any():
                    # the bake takes the first surface a ray meets: mostly transparent cards (the fringe of a feather)
                    # in front of opaque ones (its body) would hide them, so they are no bake source then
                    if opacity is None:
                        opacity = poly_opacity(src, high, alpha)
                    solid = opacity[src_polys] >= 0.6
                    if float(high.poly_area[src_polys][solid].sum()) >= 0.35 * float(high.poly_area[src_polys].sum()):
                        src_polys = src_polys[solid]
                group_polys.setdefault(g, []).extend(int(high.poly_index[k]) for k in src_polys)
                assigned[it['polys']] = True
            info['protrusions'].append({'kind': it['kind'] + (' (attached)' if it.get('attached') else '') + (' ' + it['shape'] if it.get('shape') else ''),
                                        'faces': nf, 'area_pct': round(100 * it['area'] / total, 2), 'out_by_H': round(it['out'] / H, 3),
                                        'centre_H': [round(float(x), 2) for x in (it['centre'] - high.lo) / H],
                                        'extent_H': [round(float(x), 3) for x in it['ext'] / H]})
    log('  protrusions: %d sheets, %d rods (%d faces); %d candidates lie on the cage' % (
        sum(p['kind'].startswith('sheet') for p in info['protrusions']), sum(p['kind'].startswith('rod') for p in info['protrusions']),
        sum(p['faces'] for p in info['protrusions']), len(prot) - len(kept)))
    body_tris = sum(len(f) - 2 for f, g in zip(B.f, B.fg) if g == 'body')
    info['body_tris'] = body_tris
    # ---- thick things far outside the cage (pads, pouches): box prisms
    body_own = np.isin(own, body_i) & ~alpha & ~assigned
    pm = body_own[high.pt_poly]
    out_cnt = np.zeros(len(high.poly_area))
    all_cnt = np.zeros(len(high.poly_area))
    far_d = np.zeros(len(high.poly_area))
    for k in np.flatnonzero(pm)[::2]:
        p = Vector(high.pts[k])
        loc, nrm, _i, d = bvh.find_nearest(p)
        pi = high.pt_poly[k]
        all_cnt[pi] += 1
        if loc is not None and d > reach * H and (p - loc).dot(nrm) > 0:
            out_cnt[pi] += 1
            far_d[pi] = max(far_d[pi], d)
    unc = (out_cnt > 0.5 * np.maximum(all_cnt, 1)) & (all_cnt > 0)
    nl = 0
    for ks in sorted(poly_clusters(high, me, unc), key=lambda ks: -high.poly_area[ks].sum()):
        a = float(high.poly_area[ks].sum())
        if a < 0.002 * H * H or far_d[ks].max() < 0.045 * H:
            info['dropped'].append({'kind': 'bulge', 'area_pct': round(100 * a / total, 2), 'out_by_H': round(float(far_d[ks].max()) / H, 3)})
            continue
        prisms.append(dict(name='part%d' % nl, kind='prism', polys=ks, P=high.pts[np.isin(high.pt_poly, ks)], area=a, w=math.sqrt(a)))
        nl += 1
    info['dropped'] = sorted(info['dropped'], key=lambda d: -d['area_pct'])[:12]
    # ---- extras
    body_area = float(high.poly_area[body_own].sum())
    dens = 1.5 * body_tris / max(body_area, 1e-9)          # an extra is not meshed much finer than the body
    wsum = sum(x['w'] for x in arcs) or 1.0
    cloth_left = max(cloth_budget - sum(len(f) - 2 for f, g in zip(B.f, B.fg) if g == 'sheets'), 0)
    built = []
    for x in sorted(arcs, key=lambda x: -x['area']) + prisms:
        ks = x['polys']
        if x['kind'] == 'arc':
            tris = min(cloth_left * x['w'] / wsum, max(dens * x['area'], 16))
            double = x['st']['inward'] > 0.25
            # sheets built before that this one lies on (hair on a cape) are obstacles like the body
            obst = [bvh]
            Ps = x['P'][::max(1, len(x['P']) // 400)]
            for a in built:
                hs_ = []
                for p_ in Ps:
                    loc, nrm, _fi, d_ = a['bvh'].find_nearest(Vector(p_))
                    if loc is not None and d_ < 0.05 * H:
                        hs_.append(float((Vector(p_) - loc).dot(nrm)))
                if len(hs_) >= 10 and np.median(hs_) > 0:
                    obst.append(a['bvh'])
            grid = {}
            snap_a = (len(B.v), len(B.f), set(B.seams))
            nf = build_arc(B, x['name'], x['P'], x['st'], x['axis_p'], sk, tris / 2 / (2 if double else 1), double, log, body=obst,
                           extend=x.get('zone'), margin=0.008 if len(obst) == 1 else 0.011, out=grid)
            if nf and double and x.get('zone') is not None:
                # round 9c: a sheet that folds through itself where it goes on up to the collar (ailisi) is built again
                # without that part
                orient_builder(B)
                nbad_ = _parity_bad(B, x['name'], None, H)
                if nbad_:
                    log('  %s: %d faces of the sheet cross it with the part up to the collar -> built without it' % (x['name'], nbad_))
                    del B.v[snap_a[0]:]
                    del B.f[snap_a[1]:]
                    del B.fg[snap_a[1]:]
                    B.seams.clear()
                    B.seams.update(snap_a[2])
                    x['zone'] = None
                    grid = {}
                    nf = build_arc(B, x['name'], x['P'], x['st'], x['axis_p'], sk, tris / 2 / (2 if double else 1), double, log, body=obst,
                                   extend=None, margin=0.008 if len(obst) == 1 else 0.011, out=grid)
            if nf and double:
                # round 9c: a double sided sheet whose two sides cross (a short closed skirt with one row too many): a
                # coarser grid, as long as that helps
                for _c in range(2):
                    orient_builder(B)
                    nbad_ = _parity_bad(B, x['name'], None, H)
                    if not nbad_:
                        break
                    keep_ = ([np.array(v_) for v_ in B.v[snap_a[0]:]], [list(f_) for f_ in B.f[snap_a[1]:]], list(B.fg[snap_a[1]:]), set(B.seams), grid, nf)
                    del B.v[snap_a[0]:]
                    del B.f[snap_a[1]:]
                    del B.fg[snap_a[1]:]
                    B.seams.clear()
                    B.seams.update(snap_a[2])
                    grid = {}
                    tris *= 0.75
                    nf = build_arc(B, x['name'], x['P'], x['st'], x['axis_p'], sk, tris / 2 / 2, double, log, body=obst,
                                   extend=x.get('zone'), margin=0.008 if len(obst) == 1 else 0.011, out=grid)
                    orient_builder(B)
                    nb2_ = _parity_bad(B, x['name'], None, H) if nf else nbad_ + 1
                    log('  %s: %d faces of the double sided sheet crossed it -> coarser grid: %d' % (x['name'], nbad_, nb2_))
                    if nb2_ >= nbad_:           # no better: the finer one stays
                        del B.v[snap_a[0]:]
                        del B.f[snap_a[1]:]
                        del B.fg[snap_a[1]:]
                        B.v.extend(keep_[0])
                        B.f.extend(keep_[1])
                        B.fg.extend(keep_[2])
                        B.seams.clear()
                        B.seams.update(keep_[3])
                        grid, nf = keep_[4], keep_[5]
                        break
            if nf and 'P' in grid:
                x['bvh'] = BVHTree.FromPolygons([tuple(p_) for p_ in grid['P'].reshape(-1, 3)], grid['quads'])
                built.append(x)
            if nf and x.get('zone') is not None and len(x['zone']['polys']):
                ks = np.unique(np.concatenate([ks, x['zone']['polys']]))      # also baked on the body below it
        else:
            if len(x['P']) < 30:
                continue
            # a compact thing (quiver with its arrows, pouch): a relief solid seen along its thinnest axis
            # (round 5: the 4 sided box was rejected), a box prism when that fails
            nv0 = nv_relief = len(B.v)
            nf_relief = len(B.f)
            nf = add_relief(B, x['name'], high.co, high.tris[np.isin(high.tri_poly, ks)], int(np.clip(4 * dens * x['area'], 40, 110)), H, res=128,
                            wall=0.006, min_thick=0.006, close=0.005, blur=0.003, plate=True, log=log)
            if not nf:
                nv0 = len(B.v)
                nf = build_prism(B, x['name'], x['P'], sk, float(np.clip(dens * x['area'], 12, 60)), N=4, max_rings=4, log=log)
            # it sits on the coat / the body, not in it (round 6): moved as a whole until no vertex is inside
            obst_ = [bvh]
            Ps = x['P'][::max(1, len(x['P']) // 300)]
            for a in built:                 # only what the source lies outside of (a cape may hang over it)
                hs_ = []
                for p_ in Ps:
                    loc, nrm, _fi, d_ = a['bvh'].find_nearest(Vector(p_))
                    if loc is not None and d_ < 0.05 * H:
                        hs_.append(float((Vector(p_) - loc).dot(nrm)))
                if len(hs_) >= 10 and np.percentile(hs_, 35) > 0:
                    obst_.append(a['bvh'])
            def gap_(pt):
                """signed height of a point above the nearest obstacle -> (height, normal, foot) or None"""
                best_ = None
                for bv_ in obst_:
                    loc, nrm, _fi, d_ = bv_.find_nearest(Vector(pt))
                    if loc is None or d_ > 0.08 * H:
                        continue
                    hg_ = float((Vector(pt) - loc).dot(nrm))
                    if best_ is None or hg_ < best_[0]:
                        best_ = (hg_, np.array(nrm), np.array(loc))
                return best_

            new_v = list(range(nv0, len(B.v)))
            relief_ = nv0 == nv_relief and len(new_v) >= 6
            side_in = set(new_v)
            if relief_:
                # round 8: a plate has a side towards the coat and one away from it.  After the rigid move (the whole
                # thing clear of the coat) the inner side is laid ON the coat vertex by vertex, along the thin axis of
                # the plate: no gap (round 7 touched in one point), no crossing, the outline keeps its shape
                Pn = np.array([B.v[k_] for k_ in new_v])
                cq, fe1, fe2, fe3 = B.last_relief['frame']
                Vq = np.array([fe1, fe2, fe3])
                # front / back vertex of the same outline point: the same place in the plane of the plate
                uvw = (Pn - Pn.mean(0)) @ Vq.T
                sd = np.zeros(len(Pn))
                for a_ in range(len(Pn)):
                    d2 = np.hypot(uvw[:, 0] - uvw[a_, 0], uvw[:, 1] - uvw[a_, 1])
                    d2[a_] = np.inf
                    b_ = int(np.argmin(d2))
                    if d2[b_] < 1e-4 * H and abs(uvw[a_, 2] - uvw[b_, 2]) > 0.004 * H:
                        sd[a_] = 1.0 if uvw[a_, 2] > uvw[b_, 2] else -1.0
                g0 = [gap_(p_) for p_ in Pn]
                m_pos = [g_[0] for g_, s_ in zip(g0, sd) if g_ is not None and s_ > 0]
                m_neg = [g_[0] for g_, s_ in zip(g0, sd) if g_ is not None and s_ < 0]
                if m_pos and m_neg:
                    inner_pos = np.mean(m_pos) < np.mean(m_neg)
                    side_in = {k_ for k_, s_ in zip(new_v, sd) if s_ != 0 and (s_ > 0) == inner_pos}
                else:
                    relief_ = False
            rigid = new_v
            clear_ = 0.004 * H
            moved_ = np.zeros(3)
            for _ in range(8):
                if np.linalg.norm(moved_) > 0.04 * H:
                    break
                worst = None
                for k_ in rigid:
                    g_ = gap_(B.v[k_])
                    if g_ is not None and (worst is None or g_[0] < worst[0]):
                        worst = g_
                if worst is None or worst[0] >= clear_:
                    break
                sh_ = worst[1] * (clear_ + 0.001 * H - worst[0])
                moved_ += sh_
                for k_ in new_v:
                    B.v[k_] = B.v[k_] + sh_
            if relief_ and float(np.linalg.norm(moved_)) < 0.002 * H:
                relief_ = False             # it did not touch anything: left where it is
            if relief_:
                seat_ = 0.0015 * H
                twin_ = {}
                for a_ in range(len(Pn)):
                    if sd[a_] != 0:
                        d2 = np.hypot(uvw[:, 0] - uvw[a_, 0], uvw[:, 1] - uvw[a_, 1])
                        d2[a_] = np.inf
                        twin_[new_v[a_]] = new_v[int(np.argmin(d2))]
                thick0 = {k_: abs(float((np.asarray(B.v[t_]) - np.asarray(B.v[k_])) @ Vq[2])) for k_, t_ in twin_.items()}
                e_out = Vq[2] * (-1.0 if inner_pos else 1.0)          # from the coat to the outer side, across the plate
                n_seated = 0
                for k_ in side_in:          # along the thin axis only: the outline of the thing keeps its shape
                    o_ = np.asarray(B.v[k_]) + e_out * 0.05 * H
                    best_ = None
                    for bv_ in obst_:
                        hit = bv_.ray_cast(Vector(o_), Vector(-e_out), 0.08 * H)      # only where it lies against the coat (within 0.015H)
                        if hit[0] is not None and (best_ is None or hit[3] < best_):
                            best_ = hit[3]
                    if best_ is not None and k_ in twin_:
                        # never through its own outer side, never more than twice as thick as it was
                        tw_ = float((np.asarray(B.v[twin_[k_]]) - np.asarray(B.v[k_])) @ e_out)
                        mv_ = best_ - seat_ - 0.05 * H          # > 0: towards the coat
                        if tw_ <= 0 or mv_ > max(2.0 * tw_, 0.025 * H) or -mv_ > 0.6 * tw_:
                            best_ = None
                    if best_ is not None:
                        if os.environ.get('SR_DBG'):
                            log('   seat v%d at %s moved %.4fH along the axis' % (k_, np.round(np.asarray(B.v[k_]) / H, 3), (0.05 * H - best_ + seat_) / H))
                        B.v[k_] = o_ - e_out * (best_ - seat_)
                        n_seated += 1
                fs_new = [f_ for f_ in B.f[nf_relief:] if all(k_ >= nv0 for k_ in f_)]
                for _ in range(10):         # the coat bulges between the vertices: faces and edges are lifted clear of it
                    push = {}
                    for f_ in fs_new:
                        pts_ = [np.asarray(B.v[k_]) for k_ in f_]
                        smp = [np.mean(pts_, axis=0)] + [(pts_[i_] + pts_[(i_ + 1) % len(pts_)]) / 2 for i_ in range(len(pts_))]
                        if len(pts_) == 4:
                            smp += [(pts_[0] + pts_[2]) / 2, (pts_[1] + pts_[3]) / 2]
                        for sp_ in smp:
                            g_ = gap_(sp_)
                            if g_ is not None and g_[0] < 0.0005 * H:
                                for k_ in f_:
                                    if k_ in side_in and (k_ not in push or seat_ - g_[0] > push[k_][0]):
                                        push[k_] = (seat_ - g_[0], g_[1])
                    if not push:
                        break
                    if os.environ.get('SR_DBG'):
                        log('   push %s' % {k_: round(d_ / H, 4) for k_, (d_, n_) in push.items()})
                    moved_any = False
                    for k_, (d_, n_) in push.items():
                        if k_ not in twin_ or float(n_ @ e_out) < 0.3:
                            continue
                        new_ = np.asarray(B.v[k_]) + e_out * d_ / float(n_ @ e_out)
                        if float((np.asarray(B.v[twin_[k_]]) - new_) @ e_out) < 0.4 * thick0[k_]:
                            continue        # it would get too thin / turn inside out there
                        B.v[k_] = new_
                        moved_any = True
                    if not moved_any:
                        break
                gs_ = [g_[0] for g_ in (gap_(B.v[k_]) for k_ in side_in) if g_ is not None and g_[0] < 0.03 * H]
                x['seat_gap_by_H'] = None if not gs_ else round(float(np.median(gs_)) / H, 4)
                x['seat_gap_max_by_H'] = None if not gs_ else round(float(np.max(gs_)) / H, 4)
                x['seated_vertices'] = n_seated
                log('  %s: seated on the coat / body: rigid move %.3fH, %d inner vertices laid on the surface (of %d within 0.03H: gap median %.4fH, max %.4fH)'
                    % (x['name'], np.linalg.norm(moved_) / H, n_seated, len(gs_), (np.median(gs_) if gs_ else 0) / H, (np.max(gs_) if gs_ else 0) / H))
                moved_ = np.zeros(3)
            else:
                # ... and seated on it: moved back as a whole until its nearest vertex is 0.004H above the surface
                worst = None
                for k_ in new_v:
                    g_ = gap_(B.v[k_])
                    if g_ is not None and (worst is None or g_[0] < worst[0]):
                        worst = g_
                if worst is not None and worst[0] > 0.005 * H:
                    for k_ in new_v:
                        B.v[k_] = B.v[k_] - worst[1] * (worst[0] - 0.004 * H)
                x['seat_gap_by_H'] = None if worst is None else round(min(worst[0], 0.004 * H) / H, 4)
            if np.linalg.norm(moved_) > 0:
                log('  %s: moved %.3fH out of the coat / body, then seated on it' % (x['name'], np.linalg.norm(moved_) / H))
        if nf:
            group_polys[x['name']] = [int(high.poly_index[k]) for k in ks]
            assigned[x['polys']] = True
            info['extras'].append({'name': x['name'], 'kind': x['kind'], 'faces': nf, 'source_area_pct': round(100 * x['area'] / total, 1)})
    hf_ = binfo.pop('hair_flap', None)
    if hf_:
        try:
            add_hair_flap(B, hf_, sk, H, log)
            binfo['hair_flap'] = {'faces': hf_['faces'], 'columns': len(hf_['A']), 'z_att_by_H': round((hf_['z_att'] - high.lo[2]) / H, 3)}
        except Exception as ex:             # the step stays as it is
            log('  hair flap failed: %r' % (ex,))
    group_polys['body'] = [int(high.poly_index[k]) for k in np.flatnonzero(~assigned & ~alpha)]
    info['alpha_card_area_pct'] = round(100 * float(high.poly_area[alpha].sum()) / total, 2)
    info['skeleton'] = {'torso': [sk.names[b] for b in sk.torso.core], 'limbs': {l.name: [sk.names[b] for b in l.core] for l in sk.legs + sk.arms},
                        'extra_groups': len(sk.extra)}
    info['tris'] = {'cage': body_tris, 'sheets': sum(len(f) - 2 for f, g in zip(B.f, B.fg) if g == 'sheets'), 'rods': sum(len(f) - 2 for f, g in zip(B.f, B.fg) if g == 'rods'),
                    'cloth': sum(len(f) - 2 for f, g in zip(B.f, B.fg) if g in {x['name'] for x in arcs}),
                    'other': sum(len(f) - 2 for f, g in zip(B.f, B.fg) if g in {x['name'] for x in prisms})}
    return B, group_polys, info, sk


# ====================================================================== relief solids (round 5)
def _trace_loops(mask):
    """boundary loops of a boolean grid [i, j] as closed polylines of grid corners, the inside on the left
    (outlines counter clockwise, holes clockwise).  Cells that touch only at a corner are not connected."""
    pad = np.pad(mask, 1)
    nxt = {}

    def add(a, b):
        nxt.setdefault(a, []).append(b)

    ii, jj = np.nonzero(mask)
    for i, j in zip(ii.tolist(), jj.tolist()):
        if not pad[i + 1, j]:
            add((i, j), (i + 1, j))
        if not pad[i + 2, j + 1]:
            add((i + 1, j), (i + 1, j + 1))
        if not pad[i + 1, j + 2]:
            add((i + 1, j + 1), (i, j + 1))
        if not pad[i, j + 1]:
            add((i, j + 1), (i, j))
    loops = []
    while nxt:
        a = next(iter(nxt))
        b = nxt[a].pop()
        if not nxt[a]:
            del nxt[a]
        loop = [a]
        prev, cur = a, b
        while cur != a:
            loop.append(cur)
            cand = nxt[cur]
            if len(cand) == 1:
                n = cand.pop()
            else:               # pinch: turn left (stay with the same cell)
                d = (cur[0] - prev[0], cur[1] - prev[1])
                left = (cur[0] - d[1], cur[1] + d[0])
                n = left if left in cand else cand[0]
                cand.remove(n)
            if not cand:
                del nxt[cur]
            prev, cur = cur, n
        loops.append(np.array(loop, dtype=float))
    return loops


def _morph(mask, r, grow):
    """dilation (grow) / erosion of a boolean grid by a disk of radius r cells"""
    if r < 0.5:
        return mask.copy()
    R = int(math.ceil(r))
    m = mask if grow else ~mask
    p = np.pad(m, R, constant_values=not grow)
    out = np.zeros_like(m)
    n0, n1 = m.shape
    for di in range(-R, R + 1):
        for dj in range(-R, R + 1):
            if di * di + dj * dj <= r * r + 0.01:
                out |= p[R + di:R + di + n0, R + dj:R + dj + n1]
    return out if grow else ~out


def _nan_blur(z, r):
    """box blur (twice) over the covered cells of a grid (nan = not covered), radius r cells"""
    m = ~np.isnan(z)
    a = np.where(m, z, 0.0)
    w = m.astype(float)
    for _ in range(2):
        pa, pw = np.pad(a, r), np.pad(w, r)
        sa, sw = np.zeros_like(a), np.zeros_like(w)
        for di in range(2 * r + 1):
            for dj in range(2 * r + 1):
                sa += pa[di:di + a.shape[0], dj:dj + a.shape[1]]
                sw += pw[di:di + a.shape[0], dj:dj + a.shape[1]]
        a = np.where(m, sa / np.maximum(sw, 1e-9), 0.0)
    return np.where(m, a, np.nan)


def _loop_area(L):
    x, y = L[:, 0], L[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _dp_significance(L, fixed=()):
    """Douglas-Peucker order of a closed polyline: for every point the tolerance below which it is kept
    (the anchors: inf).  fixed: indices that are anchors too"""
    n = len(L)
    sig = np.zeros(n)
    anchors = sorted(set(fixed))
    if len(anchors) < 2:
        a = anchors[0] if anchors else 0
        anchors = sorted({a, int(np.argmax(((L - L[a]) ** 2).sum(1)))})
    for a in anchors:
        sig[a] = np.inf
    stack = [(a, b, np.inf) for a, b in zip(anchors, anchors[1:] + [anchors[0] + n])]
    while stack:
        i0, i1, cap = stack.pop()
        if i1 - i0 < 2:
            continue
        idx = np.arange(i0 + 1, i1) % n
        p0, p1 = L[i0 % n], L[i1 % n]
        d = p1 - p0
        ln = float(np.hypot(*d))
        q = L[idx] - p0
        if ln < 1e-12:
            dist = np.hypot(q[:, 0], q[:, 1])
        else:
            t = np.clip((q @ d) / (ln * ln), 0, 1)
            dist = np.hypot(*(q - t[:, None] * d).T)
        k = int(np.argmax(dist))
        s = min(float(dist[k]), cap)
        sig[idx[k]] = s
        m = i0 + 1 + k
        stack.append((i0, m, s))
        stack.append((m, i1, s))
    return sig


def _inside_loops(P, loops):
    """even-odd point in polygon test of the points P (n, 2)"""
    cnt = np.zeros(len(P), dtype=np.int64)
    x, y = P[:, 0], P[:, 1]
    for L in loops:
        x0, y0 = L[:, 0], L[:, 1]
        x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
        for k in range(len(L)):
            if y0[k] == y1[k]:
                continue
            cnt += ((y0[k] > y) != (y1[k] > y)) & (x < (x1[k] - x0[k]) * (y - y0[k]) / (y1[k] - y0[k]) + x0[k])
    return cnt % 2 == 1


def _raster_interp(V2, Z, tris, shape):
    """height of the triangles (V2 in grid units, Z per vertex) at the cell centres; nan where no triangle is"""
    out = np.full(shape, np.nan)
    for t in tris:
        X, Y, z = V2[t, 0], V2[t, 1], Z[t]
        i0, i1 = max(int(math.floor(X.min() - 0.5)), 0), min(int(math.ceil(X.max() - 0.5)), shape[0] - 1)
        j0, j1 = max(int(math.floor(Y.min() - 0.5)), 0), min(int(math.ceil(Y.max() - 0.5)), shape[1] - 1)
        if i1 < i0 or j1 < j0:
            continue
        den = (Y[1] - Y[2]) * (X[0] - X[2]) + (X[2] - X[1]) * (Y[0] - Y[2])
        if abs(den) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(i0, i1 + 1) + 0.5, np.arange(j0, j1 + 1) + 0.5, indexing='ij')
        a = ((Y[1] - Y[2]) * (gx - X[2]) + (X[2] - X[1]) * (gy - Y[2])) / den
        b = ((Y[2] - Y[0]) * (gx - X[2]) + (X[0] - X[2]) * (gy - Y[2])) / den
        c = 1 - a - b
        ins = (a >= -1e-9) & (b >= -1e-9) & (c >= -1e-9)
        if ins.any():
            sub = out[i0:i1 + 1, j0:j1 + 1]
            sub[ins] = (a * z[0] + b * z[1] + c * z[2])[ins]
    return out


def relief_solid(co, tris, budget, S, frame=None, res=320, outline_share=0.5, wall=0.012, min_thick=0.006, close=0.006, blur=0.004,
                 symmetric='auto', plate=False, boss=False, tips=False, spine=0, seal=0.0, knife=2.25, spur_close=None, end_boost=0.0, smooth=0.0, limb_strip=0, log=print):
    """A closed solid that follows the outline of a flat-ish object (a blade, the limbs of a bow, a flange, a
    quiver) seen along its thinnest axis.  Along that axis the source is ray cast from both sides on a grid: the
    covered cells are the figure, the two depths its front and back.  The outline of the figure (narrow gaps and
    notches closed) is simplified to as many points as the budget allows (Douglas-Peucker; small holes and
    islands go first); the front and the back are triangulated inside it (constrained Delaunay) with interior
    points added where the depth is furthest from the triangles.  Where the object is thick at its outline the
    front and back outline are joined by a wall, where it is sharp they share the vertices.
    symmetric: the figure is mirror symmetric in its second axis ('auto': tested; True: about v = 0 of `frame`):
    one half is built and mirrored.
    plate (round 6: the fans around interior points were rejected as topology): no interior points at all.  The
    outline is extruded: front and back outline everywhere (at least min_thick apart, each vertex at the depth of
    the object there), a wall of quads all around, the two flat sides triangulated across the outline (a ladder
    for a long shape) and joined to quads.  A mirrored figure gets vertices on its axis where the depth changes
    along it (the ridge of a stock).
    All lengths (wall, min_thick, close, blur) are fractions of the size S.
    -> verts (n, 3), faces (outward), seam edges, info"""
    used = np.unique(tris)
    if frame is None:
        a, b, c_ = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
        ar = 0.5 * np.linalg.norm(np.cross(b - a, c_ - a), axis=1)
        cen = (a + b + c_) / 3
        c = (cen * ar[:, None]).sum(0) / max(ar.sum(), 1e-30)
        q = cen - c
        w, V = np.linalg.eigh((q * ar[:, None]).T @ q / max(ar.sum(), 1e-30))
        V = V[:, np.argsort(-w)].T
        e1, e2 = V[0], V[1]
    else:
        c, e1, e2 = (np.asarray(x, float) for x in frame)
    e3 = np.cross(e1, e2)
    e3 /= np.linalg.norm(e3)
    q = co[used] - c
    U, Vv, W = q @ e1, q @ e2, q @ e3
    ext = max(np.ptp(U), np.ptp(Vv), 1e-9)
    cell = ext / res
    bvh = BVHTree.FromPolygons([tuple(p) for p in co], [tuple(int(i) for i in t) for t in tris])
    wtop, wbot = W.max() + 0.01 * ext, W.min() - 0.01 * ext
    dn, up = Vector(-e3), Vector(e3)

    def cast(u0, v0, nu, nv):
        zf = np.full((nu, nv), np.nan)
        zb = np.full((nu, nv), np.nan)
        for i in range(nu):
            base = c + e1 * (u0 + (i + 0.5) * cell)
            for j in range(nv):
                p = base + e2 * (v0 + (j + 0.5) * cell)
                hit = bvh.ray_cast(Vector(p + e3 * wtop), dn)
                if hit[0] is None:
                    continue
                zf[i, j] = wtop - hit[3]
                hit = bvh.ray_cast(Vector(p + e3 * wbot), up)
                zb[i, j] = wbot + hit[3] if hit[0] is not None else zf[i, j]
        return zf, zb

    u0, v0 = U.min() - 3 * cell, Vv.min() - 3 * cell
    nu, nv = int(math.ceil(np.ptp(U) / cell)) + 6, int(math.ceil(np.ptp(Vv) / cell)) + 6
    zf, zb = cast(u0, v0, nu, nv)
    mask0 = ~np.isnan(zf)
    if not mask0.any():
        return None
    # mirror symmetry in v
    J = None
    if symmetric:
        js = np.flatnonzero(mask0.any(0))
        vc = 0.0 if symmetric is True else v0 + cell * (js[0] + js[-1] + 1) / 2.0
        jc = (vc - v0) / cell                      # axis in grid units
        jm = np.clip(np.round(2 * jc - 1 - np.arange(nv)).astype(int), 0, nv - 1)
        mir = mask0[:, jm]
        iou = float((mask0 & mir).sum()) / max(float((mask0 | mir).sum()), 1.0)
        if symmetric is True or iou > 0.9:
            half = int(math.ceil(max(Vv.max() - vc, vc - Vv.min()) / cell)) + 3
            v0, nv, J = vc - half * cell, 2 * half, half
            zf, zb = cast(u0, v0, nu, nv)
            m = ~np.isnan(zf)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                zf = np.nanmean(np.stack([zf, zf[:, ::-1]]), axis=0)
                zb = np.nanmean(np.stack([zb, zb[:, ::-1]]), axis=0)
            mask0 = m | m[:, ::-1]
    if boss:                                # only where the object is clearly thicker than usual (a hub, a centre cross): a second plate on the first
        T_ = zf - zb
        t50 = float(np.nanmedian(T_[mask0]))
        mb = mask0 & (np.nan_to_num(T_) > max(1.7 * t50, 0.5 * wall * S))
        mb = _morph(_morph(mb, 2.0, False), 2.0, True)
        if not (0.04 * mask0.sum() < mb.sum() < 0.6 * mask0.sum()):
            return None
        mask0 = mb
        zf[~mb] = np.nan
        zb[~mb] = np.nan
    rc = close * S / cell
    mask = _morph(_morph(mask0, rc, True), rc, False)          # narrow gaps and notches closed
    if seal > 0:
        # round 8: a pocket with a narrow mouth (the inside of a scroll, the bay under a hook) is filled - the scroll
        # becomes one solid tooth of the outline, its inside is cut out by the alpha of the bake.  Open notches (the
        # gaps of saw teeth) are NOT touched (a plain closing of this radius would fill them): only what the
        # closing shuts off from the outside counts, plus the seal of its mouth
        rs = seal * S / cell
        big = np.pad(_morph(_morph(np.pad(mask, int(rs) + 2), rs, True), rs, False), 1)
        out_ = np.zeros(big.shape, dtype=bool)
        out_[0, :] = out_[-1, :] = out_[:, 0] = out_[:, -1] = True
        free = ~big
        while True:                         # flood fill of the outside
            g = out_.copy()
            g[1:, :] |= out_[:-1, :]
            g[:-1, :] |= out_[1:, :]
            g[:, 1:] |= out_[:, :-1]
            g[:, :-1] |= out_[:, 1:]
            g &= free
            g |= out_
            if (g == out_).all():
                break
            out_ = g
        o_ = int(rs) + 3
        shut = (free & ~out_)[o_:o_ + mask.shape[0], o_:o_ + mask.shape[1]]
        bigc = big[o_:o_ + mask.shape[0], o_:o_ + mask.shape[1]]
        # ... that was open before (a real hole of the object stays a hole)
        pm = np.pad(mask, 1)
        outm = np.zeros(pm.shape, dtype=bool)
        outm[0, :] = outm[-1, :] = outm[:, 0] = outm[:, -1] = True
        freem = ~pm
        while True:
            g = outm.copy()
            g[1:, :] |= outm[:-1, :]
            g[:-1, :] |= outm[1:, :]
            g[:, 1:] |= outm[:, :-1]
            g[:, :-1] |= outm[:, 1:]
            g &= freem
            g |= outm
            if (g == outm).all():
                break
            outm = g
        pocket = shut & outm[1:-1, 1:-1]
        if os.environ.get('SR_DUMP'):
            DEBUG['_mask_n'] = DEBUG.get('_mask_n', 0) + 1
            np.save(os.path.join(os.environ['SR_DUMP'], 'mask%d.npy' % DEBUG['_mask_n']), np.stack([mask0, mask, bigc, pocket]))
        if os.environ.get('SR_DBG'):
            print('   seal: rs %.1f cells, shut %d, pocket %d, mask %d, grid %s' % (rs, shut.sum(), pocket.sum(), mask.sum(), mask.shape), flush=True)
        if pocket.any() and pocket.sum() < 0.08 * mask.sum():
            mask = mask | pocket | (bigc & ~mask & _morph(pocket, rs + 1.5, True))
    # thin short spurs (ornaments) go: what an opening removes, unless it is a big piece (a long tip, a string)
    rc_close = rc
    if spur_close is not None:              # round 9: a wide closing does not make the spurs that go any larger
        rc = spur_close * S / cell
    op = _morph(_morph(mask, rc, False), rc, True) & mask
    spur = mask & ~op
    if spur.any() and op.any():
        lab = np.zeros(mask.shape, dtype=np.int32)
        nl = 0
        si, sj = np.nonzero(spur)
        for i, j in zip(si.tolist(), sj.tolist()):
            if lab[i, j]:
                continue
            nl += 1
            st_ = [(i, j)]
            lab[i, j] = nl
            cells = []
            while st_:
                a_, b_ = st_.pop()
                cells.append((a_, b_))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x_, y_ = a_ + da, b_ + db
                    if 0 <= x_ < mask.shape[0] and 0 <= y_ < mask.shape[1] and spur[x_, y_] and not lab[x_, y_]:
                        lab[x_, y_] = nl
                        st_.append((x_, y_))
            if len(cells) < (1.2 if plate else 2.5) * rc * rc + 4:
                for a_, b_ in cells:
                    mask[a_, b_] = False
    for z in (zf, zb):                     # cells covered by the closing only: the depth of their neighbours
        z[~mask] = np.nan
        for _ in range(int(rc_close) + 3):
            need = mask & np.isnan(z)
            if not need.any():
                break
            p = np.pad(z, 1, constant_values=np.nan)
            st_ = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
            ok = ~np.isnan(st_)
            mval = np.where(ok, st_, 0).sum(0) / np.maximum(ok.sum(0), 1)
            fill = need & ok.any(0)
            z[fill] = mval[fill]
        z[mask & np.isnan(z)] = np.nanmedian(z) if (~np.isnan(z)).any() else 0.0
    rb = max(1, int(round(blur * S / cell)))
    zf, zb = _nan_blur(zf, rb), _nan_blur(zb, rb)
    swap = zb > zf
    zf[swap], zb[swap] = zb[swap], zf[swap]
    if J is not None:                     # one half
        mask = mask[:, J:]
        zf, zb = zf[:, J:], zb[:, J:]
        v0 += J * cell
        nv -= J
    loops = [L for L in _trace_loops(mask)]
    areas = [_loop_area(L) for L in loops]
    # round 9c ("the limbs are a jagged polygon, the high model is a smooth curved arm"): the outline points are picked
    # on a smoothed copy of the big outlines (the grid steps and the small bumps are gone, the points then spread
    # evenly along the curves); sharp convex corners (the point of a spike, of a wing tip, of a hook) and the points on
    # the mirror axis are not smoothed
    loops_s = [L for L in loops]
    if smooth > 0:
        sg_ = smooth * S / cell
        for li, L in enumerate(loops):
            n_ = len(L)
            if n_ < 12 * sg_ or abs(areas[li]) < 40 * sg_ * sg_:
                continue
            keep_ = (L[:, 1] == 0) if J is not None else np.zeros(n_, dtype=bool)
            Lf = L.astype(float)
            sign_ = 1.0 if areas[li] > 0 else -1.0
            for m_ in (int(max(3, round(1.0 * sg_))), int(max(4, round(1.8 * sg_)))):
                a_ = Lf - np.roll(Lf, m_, axis=0)
                b_ = np.roll(Lf, -m_, axis=0) - Lf
                crs = (a_[:, 0] * b_[:, 1] - a_[:, 1] * b_[:, 0]) * sign_
                cs_ = -(a_ * b_).sum(1) / np.maximum(np.hypot(a_[:, 0], a_[:, 1]) * np.hypot(b_[:, 0], b_[:, 1]), 1e-9)
                tip_ = (crs > 0) & (cs_ > math.cos(math.radians(SMOOTH_TIP)))
                # the sharpest point of every run only
                for k in np.flatnonzero(tip_):
                    if cs_[k] >= cs_[(k - 1) % n_] and cs_[k] >= cs_[(k + 1) % n_]:
                        keep_[k] = True
            kk = np.flatnonzero(keep_)
            idx = np.arange(n_)
            if len(kk):
                dist = np.min(np.minimum(np.abs(idx[:, None] - kk[None, :]), n_ - np.abs(idx[:, None] - kk[None, :])), axis=1).astype(float)
            else:
                dist = np.full(n_, 1e9)
            R_ = int(math.ceil(3 * sg_))
            ww = np.exp(-0.5 * (np.arange(-R_, R_ + 1) / sg_) ** 2)
            ww /= ww.sum()
            Sm = np.zeros_like(Lf)
            for o_, w_ in zip(range(-R_, R_ + 1), ww):
                Sm += w_ * np.roll(Lf, -o_, axis=0)
            wgt = 1.0 - np.exp(-(dist / (1.2 * sg_)) ** 2)
            Ls = Lf + (Sm - Lf) * wgt[:, None]
            if J is not None:
                Ls[:, 1] = np.where(L[:, 1] == 0, 0.0, np.maximum(Ls[:, 1], 0.5))
            loops_s[li] = Ls
    # round 9d ("the limbs are a few big irregular triangles with a squared-off end"): a limb of a mirrored figure (what
    # reaches far out from the stock) is swept along its length - geodesic distance inside the figure from where it
    # leaves the stock; the outline points of the limb are the two ends of `limb_strip` evenly spaced rungs plus the
    # tip (the far end of the sweep: the point of the hook).  The triangulation between such pairs is a strip of
    # quads that follows the curve of the limb
    limb = None
    if limb_strip and J is not None and loops:
        try:
            li_ = int(np.argmax(areas))
            Lb = loops[li_]
            occ = mask.any(1)
            runlen = np.array([int(np.argmin(np.append(mask[i_], False))) for i_ in np.flatnonzero(occ)])
            j0 = int(round(1.5 * float(np.median(runlen)) + 2))
            reg = mask.copy()
            reg[:, :j0] = False
            # largest connected piece beyond the stock
            lab = np.zeros(mask.shape, dtype=np.int32)
            best_c, nl_ = (0, 0), 0
            for i_, j_ in zip(*np.nonzero(reg)):
                if lab[i_, j_]:
                    continue
                nl_ += 1
                st_ = [(int(i_), int(j_))]
                lab[i_, j_] = nl_
                cnt_ = 0
                while st_:
                    a_, b_ = st_.pop()
                    cnt_ += 1
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x_, y_ = a_ + da, b_ + db
                        if 0 <= x_ < mask.shape[0] and 0 <= y_ < mask.shape[1] and reg[x_, y_] and not lab[x_, y_]:
                            lab[x_, y_] = nl_
                            st_.append((x_, y_))
                if cnt_ > best_c[0]:
                    best_c = (cnt_, nl_)
            if best_c[0] > 0.12 * mask.sum():
                reg = lab == best_c[1]
                import heapq
                dist = np.full(mask.shape, np.inf)
                hq = []
                for i_ in np.flatnonzero(reg[:, j0]):
                    dist[i_, j0] = 0.0
                    hq.append((0.0, int(i_), j0))
                heapq.heapify(hq)
                while hq:
                    d_, a_, b_ = heapq.heappop(hq)
                    if d_ > dist[a_, b_]:
                        continue
                    for da, db, w_ in ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)):
                        x_, y_ = a_ + da, b_ + db
                        if 0 <= x_ < mask.shape[0] and 0 <= y_ < mask.shape[1] and reg[x_, y_] and d_ + w_ < dist[x_, y_]:
                            dist[x_, y_] = d_ + w_
                            heapq.heappush(hq, (d_ + w_, x_, y_))
                # distance at the vertices of the outline (grid corners): that of the cells around them
                n_ = len(Lb)
                dv = np.full(n_, -1.0)
                for k in range(n_):
                    ci, cj = int(Lb[k, 0]), int(Lb[k, 1])
                    for a_ in (ci - 1, ci):
                        for b_ in (cj - 1, cj):
                            if 0 <= a_ < mask.shape[0] and 0 <= b_ < mask.shape[1] and np.isfinite(dist[a_, b_]):
                                dv[k] = max(dv[k], dist[a_, b_])
                inr = dv >= 0
                if inr.sum() > 30:
                    kt = int(np.argmax(dv))
                    dmax = float(dv[kt])
                    # the run of the outline through the tip, on both sides
                    ka = kt
                    while inr[(ka - 1) % n_] and (ka - 1) % n_ != kt:
                        ka = (ka - 1) % n_
                    kb_ = kt
                    while inr[(kb_ + 1) % n_] and (kb_ + 1) % n_ != kt:
                        kb_ = (kb_ + 1) % n_
                    side_a = [(ka + q_) % n_ for q_ in range((kt - ka) % n_ + 1)]            # ka ... kt
                    side_b = [(kt + q_) % n_ for q_ in range((kb_ - kt) % n_ + 1)]           # kt ... kb
                    if dmax > 12 and len(side_a) > 8 and len(side_b) > 8:
                        forced = {ka, kb_, kt}
                        Kr = int(limb_strip)
                        # the last rungs lie closer together: the hook
                        lv = [dmax * t_ for t_ in list(np.linspace(0, 0.8, Kr - 1)[1:]) + [0.9, 0.97]]
                        for d_k in lv:
                            ia = next((k for k in side_a if dv[k] >= d_k), None)
                            ib = next((k for k in reversed(side_b) if dv[k] >= d_k), None)
                            if ia is not None and ib is not None:
                                forced.update((ia, ib))
                        # the point of the wing: the vertex of the limb that stands furthest out of the swept band (on
                        # the raw outline: the smoothing rounds it off) is an outline point of its own, unsmoothed
                        Lr = Lb.astype(float)
                        best_w = None
                        for side_ in (side_a, side_b):
                            fs_ = [k for k in side_ if k in forced]
                            for u_, v_ in zip(fs_[:-1], fs_[1:]):
                                seg = side_[side_.index(u_) + 1:side_.index(v_)]
                                if not seg:
                                    continue
                                p0, p1 = Lr[u_], Lr[v_]
                                dd_ = p1 - p0
                                ln_ = max(float(np.hypot(*dd_)), 1e-9)
                                out_ = ((Lr[seg] - p0) @ np.array([dd_[1], -dd_[0]])) / ln_ * (1.0 if areas[li_] > 0 else -1.0)
                                k_ = int(np.argmax(out_))
                                if out_[k_] > 3.0 and (best_w is None or out_[k_] > best_w[0]):
                                    best_w = (float(out_[k_]), seg[k_])
                        if best_w is not None:
                            forced.add(best_w[1])
                            loops_s[li_] = np.array(loops_s[li_], float)
                            loops_s[li_][best_w[1]] = Lr[best_w[1]]
                        runm = np.zeros(n_, dtype=bool)
                        runm[side_a] = True
                        runm[side_b] = True
                        limb = (li_, sorted(forced), runm)
                        if os.environ.get('SR_DBG'):
                            print('   limb strip: stock half width %d cells, limb %d cells, length %.0f cells, %d rungs, %d outline points' % (j0, best_c[0], dmax, len(lv), len(forced)), flush=True)
        except Exception as ex:
            limb = None
            if os.environ.get('SR_DBG'):
                print('   limb strip failed: %r' % (ex,), flush=True)
    # significance of every outline point; a loop counts as much as its size (small holes / islands go first)
    cand, cand_b = [], []
    axis_pt = {}
    for li, (L, A) in enumerate(zip(loops, areas)):
        if abs(A) < 6:
            continue
        on_axis = np.flatnonzero(L[:, 1] == 0) if J is not None else np.zeros(0, dtype=int)
        fixed = []
        if len(on_axis):                  # the ends of the stretches on the mirror axis stay
            s_ = set(on_axis.tolist())
            fixed = [k for k in on_axis.tolist() if (k - 1) % len(L) not in s_ or (k + 1) % len(L) not in s_]
        sig = _dp_significance(loops_s[li], fixed)
        if limb is not None and li == limb[0]:          # in the swept limb only strong features (the point of the wing) count besides the rungs
            fz_ = np.zeros(len(L), dtype=bool)
            fz_[limb[1]] = True
            sig = np.where(limb[2] & ~fz_ & np.isfinite(sig), sig * LIMB_OTHER, sig)
        size = math.sqrt(abs(A)) * 0.5
        fin = np.isfinite(sig)
        third = float(sig[fin].max()) if fin.any() else 0.0
        top = min(size, max(third, 1.0)) * (1.0 if A > 0 else 0.6)
        sig = np.where(fin, np.minimum(sig, top), sig)
        if end_boost and J is not None and len(L) > 24:
            # round 9b: the ends of the limbs (far from the mirror axis) are what the eye reads - the hook, the barb
            # of the wing tip: their outline points count more than those of the stock
            far_ = L[:, 1] > END_FROM * float(L[:, 1].max())
            sig = np.where(fin & far_, np.minimum(sig * end_boost, top), sig)
        k3 = int(np.argmax(np.where(fin, sig, -1)))
        anchors = set(np.flatnonzero(~fin).tolist())
        if len(anchors) < 3:
            anchors.add(k3)
        if tips and len(L) > 24:
            # round 8: outline points go where the teeth are - a sharp turn of the outline (the point of a tooth / spike,
            # the notch between two saw teeth) counts 1.5 times its size against the points of smooth curves
            Lf = L.astype(float)
            sharp_ = np.zeros(len(L), dtype=bool)
            for m_ in (3, 5):
                a_ = np.roll(Lf, m_, axis=0) - Lf
                b_ = np.roll(Lf, -m_, axis=0) - Lf
                cs_ = (a_ * b_).sum(1) / np.maximum(np.hypot(a_[:, 0], a_[:, 1]) * np.hypot(b_[:, 0], b_[:, 1]), 1e-9)
                sharp_ |= cs_ > math.cos(math.radians(75.0))
            sig_b = np.where(fin & sharp_, np.minimum(sig * 1.5, top), sig)
            for k in range(len(L)):
                cand_b.append((top if k in anchors else float(sig_b[k]), li, k))
        else:
            for k in range(len(L)):
                cand_b.append((top if k in anchors else float(sig[k]), li, k))
        for k in range(len(L)):
            cand.append((top if k in anchors else float(sig[k]), li, k))
    if not cand:
        return None
    cand.sort(key=lambda t: -t[0])
    if tips and plate and cand:
        # ... but only when the budget follows the outline closely enough for teeth at all (within 4.5 cells): a
        # coarse outline (the crossbow) needs every point for its main shape
        n_est = max(int(budget * (outline_share / 0.5) / (4.0 if J is None else 8.0)), 3)
        if cand[min(n_est, len(cand) - 1)][0] < 4.5:
            cand = sorted(cand_b, key=lambda t: -t[0])
    if os.environ.get('SR_DBG'):
        print('   outline points by significance (cells):', {t_: sum(1 for c_ in cand if c_[0] > t_) for t_ in (6, 5, 4, 3.5, 3, 2.5, 2, 1.5, 1.2)}, 'cell/S %.5f' % (cell / S), flush=True)
    n_out = max(int(budget * outline_share / (2 if J is None else 4)), 3)
    keep = {}
    if plate:                               # 4 triangles per outline point (2 of the wall, 1 on each side)
        n_out = max(int(budget * (outline_share / 0.5) / (4.0 if J is None else 8.0)), 3)
        if J is not None:                   # vertices on the mirror axis where the depth changes along it
            for li, L in enumerate(loops):
                ax_k = [k for k in range(len(L)) if L[k, 1] == 0]
                if len(ax_k) < 3 or abs(areas[li]) < 6:
                    continue
                def zz(k):
                    i_ = int(min(max(L[k, 0], 0), zf.shape[0] - 1))
                    col_f, col_b = zf[max(i_ - 1, 0):i_ + 1, :2], zb[max(i_ - 1, 0):i_ + 1, :2]
                    if np.isnan(col_f).all():
                        return None
                    return np.array([np.nanmean(col_f), np.nanmean(col_b)])
                run = [k for k in sorted(ax_k, key=lambda k: L[k, 0]) if zz(k) is not None]
                if len(run) < 3:
                    continue
                sel_ = [0, len(run) - 1]
                while len(sel_) < 2 + max(2, n_out // 5):
                    best = None
                    for a_, b_ in zip(sel_[:-1], sel_[1:]):
                        if L[run[b_], 0] - L[run[a_], 0] > len(run) + 5:      # not one stretch
                            continue
                        za, zb_ = zz(run[a_]), zz(run[b_])
                        for m_ in range(a_ + 1, b_):
                            t_ = (L[run[m_], 0] - L[run[a_], 0]) / max(L[run[b_], 0] - L[run[a_], 0], 1e-9)
                            e_ = float(np.abs(zz(run[m_]) - (za * (1 - t_) + zb_ * t_)).max())
                            if best is None or e_ > best[0]:
                                best = (e_, m_)
                    if best is None or best[0] < 0.4 * wall * S:
                        break
                    sel_ = sorted(sel_ + [best[1]])
                keep.setdefault(li, []).extend(run[m_] for m_ in sel_[1:-1])
            n_out = max(n_out - sum(len(v) for v in keep.values()), 3)
    if limb is not None:
        new_ = [k for k in limb[1] if k not in keep.get(limb[0], ())]
        keep.setdefault(limb[0], []).extend(new_)
        n_out = max(n_out - len(new_), 11)
    tol = max(cand[min(n_out, len(cand) - 1)][0], 0.75)
    for s, li, k in cand:
        if s > tol and k not in keep.get(li, ()):
            keep.setdefault(li, []).append(k)
    polys = [loops_s[li][sorted(ks)] for li, ks in keep.items() if len(ks) >= 3]
    if not polys:
        li = int(np.argmax(areas))
        ks = sorted(k for s, l_, k in cand if l_ == li)[:]
        sig_l = sorted(((s, k) for s, l_, k in cand if l_ == li), reverse=True)[:4]
        polys = [loops[li][sorted(k for _s, k in sig_l)]]
    if os.environ.get('SR_DUMP'):
        import pickle
        DEBUG['_rs_n'] = DEBUG.get('_rs_n', 0) + 1
        pickle.dump(dict(mask0=mask0, mask=mask, polys=polys, J=J, loops=loops), open(os.path.join(os.environ['SR_DUMP'], 'relief%d.pkl' % DEBUG['_rs_n']), 'wb'))
    # crossings of the simplified outline are resolved once: both sides then have the same constraint edges
    V2, E = [], []
    for P in polys:
        i0 = len(V2)
        V2 += [tuple(p) for p in P]
        E += [(i0 + k, i0 + (k + 1) % len(P)) for k in range(len(P))]
    r = mgeo.delaunay_2d_cdt([Vector(p) for p in V2], E, [], 0, 1e-6)
    BV = np.array([tuple(p) for p in r[0]])
    BE = [tuple(e) for e, oe in zip(r[1], r[4]) if oe]
    nb = len(BV)
    on_ax = (np.abs(BV[:, 1]) < 1e-6) if J is not None else np.zeros(nb, dtype=bool)
    if J is not None:                       # an outline that only touches the axis in one vertex: the halves stay apart there
        deg0 = np.zeros(nb, dtype=int)
        for a_, b_ in BE:
            if on_ax[a_] and on_ax[b_]:
                deg0[a_] += 1
                deg0[b_] += 1
        lone = on_ax & (deg0 == 0)
        BV[lone, 1] = 0.5
        on_ax &= ~lone

    def grid_z(z, P):
        out = np.zeros(len(P))
        for k, (x, y) in enumerate(P):
            i, j = int(min(max(x, 0), z.shape[0] - 1)), int(min(max(y, 0), z.shape[1] - 1))
            for rad in (0, 1, 2, 3, 5, 9):
                sub = z[max(i - rad, 0):i + rad + 1, max(j - rad, 0):j + rad + 1]
                m = ~np.isnan(sub)
                if m.any():
                    out[k] = np.median(sub[m])
                    break
        return out

    def border_z(z):
        out = np.zeros(nb)
        for k, (x, y) in enumerate(BV):
            i, j = int(round(x)), int(round(y))
            for rad in (1, 2, 4, 8):
                sub = z[max(i - rad, 0):i + rad, max(j - rad, 0):j + rad]
                m = ~np.isnan(sub)
                if m.any():
                    out[k] = np.median(sub[m])
                    break
        return out

    bf, bb = border_z(zf), border_z(zb)
    thick_b = bf - bb
    split = thick_b > wall * S
    if plate:
        split = np.ones(nb, dtype=bool)
        mid_ = (bf + bb) / 2
        raw_th = thick_b.copy()
        sharp = np.zeros(nb, dtype=bool)
        th_ = np.maximum(thick_b, min_thick * S)
        bf, bb = mid_ + th_ / 2, mid_ - th_ / 2
        thick_b = th_
        if tips and os.environ.get('SR_DBG'):
            print('   raw thickness at the outline / S, percentiles 10..90:', np.round(np.percentile(raw_th, [10, 25, 50, 75, 90]) / S, 4), flush=True)
        if tips:
            # round 8: the point of a tooth / spike / horn (a sharp convex corner of the outline) is ONE vertex, the
            # two sides meet there in an edge like the real thing.  Such a point costs 2 triangles instead of 4,
            # which pays for more teeth in the outline
            nbr = collections.defaultdict(list)
            for a_, b_ in BE:
                nbr[a_].append(b_)
                nbr[b_].append(a_)
            for k in range(nb):
                if len(nbr[k]) != 2 or on_ax[k] or th_[k] > 0.03 * S:
                    continue
                if raw_th[k] < knife * min_thick * S:      # a knife edge (the rim of a blade): one vertex as well
                    split[k] = False
                    bf[k] = bb[k] = mid_[k]
                    thick_b[k] = 0.0
                    continue
                d1, d2 = BV[nbr[k][0]] - BV[k], BV[nbr[k][1]] - BV[k]
                l1, l2 = float(np.hypot(*d1)), float(np.hypot(*d2))
                if min(l1, l2) < 1e-9:
                    continue
                cs_ = float(d1 @ d2) / (l1 * l2)
                if cs_ < math.cos(math.radians(62.0)):
                    continue
                bis = d1 / l1 + d2 / l2
                bl = float(np.hypot(*bis))
                if bl < 1e-9:
                    continue
                probe = BV[k] + bis / bl * min(0.3 * min(l1, l2), 1.5)
                if not _inside_loops(probe[None, :], polys)[0]:
                    continue                # a notch, not a point
                sharp[k] = True
                split[k] = False
                bf[k] = bb[k] = mid_[k]
                thick_b[k] = 0.0
    # on the mirror axis the two sides stay apart unless the object really ends there in an edge
    ax_in = on_ax.copy()
    if J is not None:
        deg_ax = np.zeros(nb, dtype=int)
        for a_, b_ in BE:
            if on_ax[a_] and on_ax[b_]:
                deg_ax[a_] += 1
                deg_ax[b_] += 1
        ax_in = on_ax & (deg_ax >= 2)
        split |= on_ax & (thick_b > 0.25 * wall * S)
    mid = (bf + bb) / 2
    bf = np.where(split, np.maximum(bf, mid + 0.5 * min_thick * S * ax_in), mid)
    bb = np.where(split, np.minimum(bb, mid - 0.5 * min_thick * S * ax_in), mid)
    er = _morph(mask, 2.0, False)
    shared_pts = []                         # the same interior points on both sides: the two sides can then never cross
    sides = [{'z': zf, 'bz': bf, 'pts': shared_pts, 'sg': 1.0}, {'z': zb, 'bz': bb, 'pts': shared_pts, 'sg': -1.0}]
    zmid = (zf + zb) / 2

    def triangulate(Sd):
        P = np.concatenate([BV, np.array(Sd['pts'], float).reshape(-1, 2)]) if Sd['pts'] else BV
        r = mgeo.delaunay_2d_cdt([Vector(p) for p in P], BE, [], 0, 1e-6)
        P2 = np.array([tuple(p) for p in r[0]])
        T = np.array([tuple(t) for t in r[2]], dtype=np.int64).reshape(-1, 3)
        if len(T):
            T = T[_inside_loops(P2[T].mean(1), polys)]
            a_, b_, c_ = P2[T[:, 0]], P2[T[:, 1]], P2[T[:, 2]]
            ar2 = (b_[:, 0] - a_[:, 0]) * (c_[:, 1] - a_[:, 1]) - (b_[:, 1] - a_[:, 1]) * (c_[:, 0] - a_[:, 0])
            T = T[np.abs(ar2) > 1e-9]
            a2 = ar2[np.abs(ar2) > 1e-9]
            T[a2 < 0] = T[a2 < 0][:, ::-1]
        if len(P2) > nb:
            zi = grid_z(Sd['z'], P2[nb:])
            zm = grid_z(zmid, P2[nb:])
            zi = zm + Sd['sg'] * np.maximum(Sd['sg'] * (zi - zm), 0.5 * min_thick * S)
            Z = np.concatenate([Sd['bz'], zi])
        else:
            Z = Sd['bz']
        Sd['P'], Sd['T'], Sd['Z'] = P2, T, Z

    def boundary_edges(T):
        ec = set()
        for t in T:
            for k in range(3):
                ec.add((int(t[k]), int(t[(k + 1) % 3])))
        return [e for e in ec if (e[1], e[0]) not in ec]

    def wall_tris(be):
        return sum(int(split[a_]) + int(split[b_]) for a_, b_ in be if not (on_ax[a_] and on_ax[b_]))

    for Sd in sides:
        triangulate(Sd)
    if not len(sides[0]['T']):
        return None
    bedges = boundary_edges(sides[0]['T'])
    # the outline touches itself in a vertex (crossing of the simplified outline): one vertex for both sides there,
    # otherwise 4 wall faces would meet in the edge between its front and its back vertex
    outd = collections.Counter(a_ for a_, _b in bedges)
    for v_, n_ in outd.items():
        if n_ > 1 and v_ < nb and split[v_]:
            split[v_] = False
            m_ = 0.5 * (sides[0]['bz'][v_] + sides[1]['bz'][v_])
            for Sd in sides:
                Sd['bz'][v_] = m_
                Sd['Z'][v_] = m_
    if plate and tips:
        # an inner edge between two single (front = back) vertices would carry 4 faces, a triangle between three of
        # them would be the same face on both sides: one end gets its two vertices back (the thicker one, a sharp
        # point last)
        bset_ = {(min(a_, b_), max(a_, b_)) for a_, b_ in bedges}
        for _ in range(nb):
            bad_ = None
            for t in sides[0]['T']:
                for k in range(3):
                    a_, b_ = int(t[k]), int(t[(k + 1) % 3])
                    if a_ < nb and b_ < nb and not split[a_] and not split[b_] and not on_ax[a_] and not on_ax[b_] and (min(a_, b_), max(a_, b_)) not in bset_:
                        bad_ = (a_, b_)
                        break
                if bad_:
                    break
            if not bad_:
                break
            k_ = max(bad_, key=lambda v_: (not sharp[v_], raw_th[v_]))
            split[k_] = True
            h_ = 0.5 * max(float(raw_th[k_]), min_thick * S)
            sides[0]['bz'][k_] = sides[0]['Z'][k_] = mid_[k_] + h_
            sides[1]['bz'][k_] = sides[1]['Z'][k_] = mid_[k_] - h_
    walls = wall_tris(bedges)
    mult = 1 if J is None else 2

    def count():
        return mult * (len(sides[0]['T']) + len(sides[1]['T']) + walls)

    def add_both(p):
        shared_pts.append((float(p[0]), float(p[1])))

    # no flat double layer: a triangle between three shared outline vertices, or an inner edge between two of
    # them that both sides have, would lie on its twin of the other side -> a point in between on both sides
    for _ in range(0 if plate else 6):
        added = 0
        shared = ~split
        for t in sides[0]['T']:
            if t.max() < nb and shared[t].all():
                add_both(sides[0]['P'][t].mean(0))
                added += 1
        if added:
            for Sd in sides:
                triangulate(Sd)
        es = []
        for Sd in sides:
            s_ = set()
            for t in Sd['T']:
                for k in range(3):
                    a_, b_ = int(t[k]), int(t[(k + 1) % 3])
                    if a_ < nb and b_ < nb and shared[a_] and shared[b_]:
                        s_.add((min(a_, b_), max(a_, b_)))
            es.append(s_)
        bset = {(min(a_, b_), max(a_, b_)) for a_, b_ in bedges}
        both = (es[0] & es[1]) - bset
        for a_, b_ in both:
            add_both((BV[a_] + BV[b_]) / 2)
            added += 1
        if not added:
            break
        for Sd in sides:
            triangulate(Sd)
    sup = max(3, int((0.07 if plate else 0.03) * max(mask.shape)))
    floor = (0.007 if plate else 0.0012) * S
    n_spine = 0
    # plate (round 8): only the few points where the object is much thicker than at its outline (a hub, the stock of a
    # crossbow) - the plate gets its real thickness there instead of a second plate on top (round 7)
    for _it in range((spine if spine else 0) if plate else 80):
        room = budget - count()
        if plate:
            room = 4 * mult if n_spine < spine else 0
        if room < 4 * mult:
            break
        errs = []
        for Sd in sides:
            zi = _raster_interp(Sd['P'], Sd['Z'], Sd['T'], mask.shape)
            e = np.abs(zi - Sd['z'])
            e[~er | np.isnan(e)] = 0
            errs.append(e)
        added = 0
        e = np.maximum(errs[0], errs[1])
        for _k in range(max(1, min(room // (4 * mult), 6))):
            i, j = np.unravel_index(int(np.argmax(e)), e.shape)
            if e[i, j] < floor:
                break
            n_spine += 1
            shared_pts.append((i + 0.5, j + 0.5))
            e[max(i - sup, 0):i + sup + 1, max(j - sup, 0):j + sup + 1] = 0
            added += 1
        if not added:
            break
        for Sd in sides:
            triangulate(Sd)
    # ---- 3D
    def w3(P, Z, sgn=1.0):
        return c + np.outer(u0 + P[:, 0] * cell, e1) + sgn * np.outer(v0 + P[:, 1] * cell, e2) + np.outer(Z, e3)

    F, Bk = sides
    verts, faces, seams = [], [], set()

    def emit(sgn):
        base = len(verts)
        idx_f = list(range(base, base + len(F['P'])))
        verts.extend(w3(F['P'], F['Z'], sgn))
        PB = w3(Bk['P'], Bk['Z'], sgn)
        idx_b = []
        for k in range(len(Bk['P'])):
            if k < nb and not split[k]:
                idx_b.append(idx_f[k])
            else:
                idx_b.append(len(verts))
                verts.append(PB[k])
        return idx_f, idx_b

    def tri(f):
        faces.append(f if cur_sgn > 0 else f[::-1])

    def sm(a_, b_):
        seams.add((min(a_, b_), max(a_, b_)))

    first = None
    for cur_sgn in ((1.0,) if J is None else (1.0, -1.0)):
        idx_f, idx_b = emit(cur_sgn)
        if first is None:
            first = (idx_f, idx_b)
        else:                               # the vertices on the mirror axis are those of the first half
            for k in np.flatnonzero(on_ax):
                idx_f[k] = first[0][k]
                idx_b[k] = first[1][k]
        for t in F['T']:
            tri([idx_f[t[0]], idx_f[t[1]], idx_f[t[2]]])
        for t in Bk['T']:
            tri([idx_b[t[0]], idx_b[t[2]], idx_b[t[1]]])
        run = 0
        for a_, b_ in bedges:
            if on_ax[a_] and on_ax[b_]:
                continue
            fa, fb, ba, bb_ = idx_f[a_], idx_f[b_], idx_b[a_], idx_b[b_]
            sm(fa, fb)
            if split[a_] and split[b_]:
                faces.append([fa, ba, bb_, fb] if cur_sgn > 0 else [fb, bb_, ba, fa])
                sm(ba, bb_)
                run += 1
                if run % 10 == 1:
                    sm(fa, ba)
            elif split[a_]:
                tri([fa, ba, fb])
                sm(ba, fb)
            elif split[b_]:
                tri([fa, bb_, fb])
                sm(fa, bb_)
    used_v = sorted({i for f in faces for i in f})
    remap = {v: k for k, v in enumerate(used_v)}
    info = {'outline_points': nb * mult, 'loops': len(polys), 'interior_points': len(shared_pts) * mult, 'wall_tris': walls * mult,
            'tris': sum(len(f) - 2 for f in faces), 'outline_tol': round(tol * cell / S, 4), 'symmetric': J is not None,
            'frame': (c, e1, e2, e3), 'ext': ext}
    return (np.array([verts[i] for i in used_v]), [[remap[i] for i in f] for f in faces],
            {(remap[a_], remap[b_]) for a_, b_ in seams if a_ in remap and b_ in remap}, info)


def add_relief(B, name, co, tris, budget, S, log=print, **kw):
    """relief_solid() into the builder -> number of triangles (0: nothing built)"""
    share0 = kw.pop('outline_share', 0.5)
    close_auto = kw.pop('close_auto', None)
    r = None
    try:
        def run(**k2):
            share, r_ = share0, None
            for _ in range(5):                  # points added against flat double layers are not in the budget: a coarser outline then
                r_ = relief_solid(co, tris, budget, S, outline_share=share, log=lambda *a: None, **dict(kw, **k2))
                if r_ is None or (r_[3]['tris'] <= 1.06 * budget + 2 and not (kw.get('plate') and r_[3]['tris'] < 0.9 * budget and _ < 3)):
                    break
                share *= float(np.clip(budget / max(r_[3]['tris'], 1), 0.7, 1.5)) if kw.get('plate') else 0.8
            return r_
        r = run()
        if close_auto and r is not None and r[3]['outline_tol'] > close_auto:
            # round 9 ("the ends of the crossbow are deformed"): an outline this coarse cannot follow the small features
            # (feather barbs, slits, the inside of scrolls) - its few points landed on them at random.  They are
            # closed first (radius about the tolerance of the outline; what the closing covers is cut out by the
            # alpha of the bake), the points then follow the main shape
            for cl in (0.004, 0.006, 0.008, 0.010, 0.013, 0.016, 0.02):
                r2 = run(close=cl, spur_close=kw.get('close', 0.006), end_boost=0.0 if LIMB_STRIP else END_BOOST, smooth=SMOOTH_K * cl, limb_strip=LIMB_STRIP)
                if r2 is None:
                    break
                r = r2
                if cl >= CLOSE_K * r2[3]["outline_tol"]:
                    break
            log('    outline too coarse for the small features: closed with radius %.2f%%' % (100 * cl))
        if r is not None:
            log(
                '    relief solid: %d outline points, %d loops (tolerance %.2f%%)%s, interior points %d, wall %d -> %d triangles (budget %d)' % (
                    r[3]['outline_points'], r[3]['loops'], 100 * r[3]['outline_tol'], ', mirrored' if r[3]['symmetric'] else '', r[3]['interior_points'],
                    r[3]['wall_tris'], r[3]['tris'], budget))
    except Exception as ex:                 # a degenerate outline: the caller falls back
        log('    relief solid failed: %s' % ex)
        r = None
    if r is None:
        return 0
    V, F, seams, info = r
    idx = B.add(V)
    for f in F:
        B.face([idx[i] for i in f], name)
    for a, b in seams:
        B.seam(idx[a], idx[b])
    B.keep_winding.add(name)
    B.join_tris.add(name)
    B.last_relief = info
    return info['tris']


class _HS:
    def __init__(self, H):
        self.H = H


def build_prop(src, faces, budget, sides=6, log=print):
    """a held prop (weapon).  Every connected piece of the source is rebuilt on its own, so the result follows
    the outline of the source: a long round piece (shaft, arrow, string: at least 4 times longer than wide,
    round cross-section) becomes an N sided prism with rings where its outline changes, everything else a
    relief solid seen along its thinnest axis (blade, the limbs of a bow, a head with its ornaments).  Small
    pieces go with the big piece they touch.  The budget is shared by the area of the pieces.
    -> Builder, {group: polygons}, info"""
    high = High(src, faces, n_samples=60000)
    D = high.diag
    me = src.data
    cl = poly_clusters(high, me, np.ones(len(high.poly_index), dtype=bool))
    items = []
    for ks in cl:
        a = float(high.poly_area[ks].sum())
        P = sample_polys(high, ks, 3000)
        if len(P) < 10:
            continue
        c, V, L = _extents(P)
        items.append(dict(polys=ks, area=a, P=P, c=c, V=V, L=L))
    items.sort(key=lambda it: -it['area'])
    total = sum(it['area'] for it in items)
    # small pieces join the nearest big one (same relief / prism)
    big = [it for it in items if it['area'] >= 0.015 * total or it['L'][0] > 0.15 * D]
    if not big:
        big = items[:1]
    for it in items:
        if any(it is b for b in big):
            continue
        d = []
        for b in big:
            kd = b.setdefault('kd', None)
            if kd is None:
                kd = KDTree(len(b['P']))
                for i, p in enumerate(b['P']):
                    kd.insert(Vector(p), i)
                kd.balance()
                b['kd'] = kd
            d.append(float(np.median([kd.find(Vector(p))[2] for p in it['P'][::30]])))
        host = big[int(np.argmin(d))]
        host.setdefault('more', []).append(it)
    for b in big:
        if b.get('more'):
            b['polys'] = np.concatenate([b['polys']] + [m['polys'] for m in b['more']])
            b['area'] += sum(m['area'] for m in b['more'])
            b['P'] = sample_polys(high, b['polys'], 4000)
            b['c'], b['V'], b['L'] = _extents(b['P'])
        L = b['L']
        b['rod'] = L[0] > 4 * L[1] and L[2] > 0.45 * L[1]
        b['string'] = b['rod'] and L[1] < 0.012 * D
        b['w'] = b['area'] ** 0.75
    all_pts = [(b, b['P'][::4]) for b in big]

    def touches(b, p):
        """is another piece at the point p (the end of a prism)"""
        for o, P in all_pts:
            if o is not b and float(np.linalg.norm(P - p, axis=1).min()) < 0.03 * D:
                return True
        return False

    B = Builder()
    B.join_shape = 60.0
    info = {'height': D, 'kind': 'extruded outlines + prisms', 'pieces': []}
    wsum = sum(b['w'] for b in big)
    left = float(budget)
    for bi, b in enumerate(big):
        share = budget * b['w'] / wsum
        n0 = sum(len(f) - 2 for f in B.f)
        if b['string']:
            N, tr = 3, 8
        elif b['rod']:
            N = sides if share >= 6 * sides else (4 if share >= 24 else 3)
            tr = max(share, 4 * N + 2)
        if b['rod']:
            q = (b['P'] - b['c']) @ b['V'][0]
            ends = [b['c'] + b['V'][0] * np.percentile(q, 0.5), b['c'] + b['V'][0] * np.percentile(q, 99.5)]
            ext = tuple(0.025 * D if touches(b, e) else 0.0 for e in ends)
            nf = build_prism(B, 'body', b['P'], _HS(D), tr, N=N, tol=0.003, axis=b['V'][0], extend=ext, smooth_axis=0 if b['string'] else 6, log=lambda *a: None)
            kind = 'prism %d' % N
            if not nf:
                b['rod'] = False
        if not b['rod']:
            tm = np.isin(high.tri_poly, b['polys'])
            round_ = b['L'][1] > 0.7 * b['L'][0]
            # round 8: no second plate on the thick middle (round 7: untidy from the front); the plate has its real
            # thickness at a few inner points instead, and the points of its teeth / spikes are single vertices
            got = add_relief(B, 'body', high.co, high.tris[tm], max(int(share), 40 if round_ else 16), D, plate=True, close=0.0015, min_thick=0.004,
                             tips=True, spine=3 if share > 120 else 0, knife=2.25 if share > 120 else 0.0, close_auto=0.009 if share > 120 else None, log=log)
            kind = 'relief'
            if not got:
                build_prism(B, 'body', b['P'], _HS(D), max(share, 12), N=4, log=lambda *a: None)
                kind = 'prism 4 (fallback)'
        n1 = sum(len(f) - 2 for f in B.f)
        info['pieces'].append({'kind': kind, 'tris': n1 - n0, 'area_pct': round(100 * b['area'] / total, 1), 'extent_by_D': [round(float(x) / D, 3) for x in b['L']]})
        log('  prop piece %d: %s, %d triangles (share %.0f), area %.1f%%, extents %s' % (bi, kind, n1 - n0, share, 100 * b['area'] / total, np.round(b['L'] / D, 3).tolist()))
    info['faces'] = len(B.f)
    gp = {'body': [int(i) for i in high.poly_index]}
    return B, gp, info, None


# ====================================================================== UV, bake, skin
def unwrap(low, boost=(), factor=1.7):
    """new UVs from the seams of the builder: one island per tube / sheet side / cap.
    boost: indices of sr_group whose islands get `factor` times the texel density (flat sheets: their outline
    is in the alpha of the texture)"""
    me = low.data
    if not me.uv_layers:
        me.uv_layers.new(name='UVMap')
    retopo._activate(low)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.003)
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.uv.average_islands_scale()
    bpy.ops.object.mode_set(mode='OBJECT')
    if boost:
        g = _group_faces(low)
        uv = me.uv_layers[0]
        a = np.empty(len(me.loops) * 2, dtype=np.float32)
        uv.uv.foreach_get('vector', a)
        a = a.reshape(-1, 2)
        for p in me.polygons:
            if g[p.index] in boost:
                a[p.loop_start:p.loop_start + p.loop_total] *= factor
        uv.uv.foreach_set('vector', a.ravel())
    retopo.atlas(low)


def _group_faces(obj):
    a = obj.data.attributes.get('sr_group')
    g = np.zeros(len(obj.data.polygons), dtype=np.int32)
    a.data.foreach_get('value', g)
    return g


def make_cage(low, pts, H, e_min=0.006, e_max=0.14, margin=0.004):
    """cage of `low` (same topology): every vertex moved along its normal far enough to be outside
    the high surface samples `pts` that lie above its faces.  -> cage object, max ray length"""
    cage = retopo._dup(low, low.name + '_cage')
    mw = low.matrix_world
    bm = bmesh.new()
    bm.from_mesh(low.data)
    bm.transform(mw)
    bm.normal_update()
    bm.faces.ensure_lookup_table()
    bvh = BVHTree.FromBMesh(bm)
    need = np.zeros(len(bm.faces))
    depth = []
    step = max(1, len(pts) // 60000)
    for p in pts[::step]:
        loc, nrm, fi, d = bvh.find_nearest(Vector(p))
        if loc is None:
            continue
        h = float((Vector(p) - loc).dot(nrm))
        if h > need[fi]:
            need[fi] = h
        elif h < 0:
            depth.append(-h)
    need = np.minimum(need, e_max * H)
    inv = mw.inverted()
    emax = 0.0
    for v in bm.verts:
        fs = v.link_faces
        if not fs:
            continue
        e = max(max(need[f.index] for f in fs) + margin * H, e_min * H)
        n = Vector((0, 0, 0))
        for f in fs:
            n += f.normal * f.calc_area()
        n = n.normalized() if n.length > 1e-12 else fs[0].normal
        lean = max(0.4, min(f.normal.dot(n) for f in fs))
        e = e / lean
        emax = max(emax, e)
        cage.data.vertices[v.index].co = inv @ (v.co + n * e)
    bm.free()
    dep = float(np.percentile(depth, 99.5)) if depth else 0.0
    return cage, emax + min(dep, e_max * H) + margin * H


def bake_source_alpha(groups, size):
    """groups: [(low, high, cage, max_ray)] (the lows share one atlas).  The alpha of the base colour textures
    of the high materials (cards: feathers, fringes), baked like the other maps.  -> (size, size) array, top row
    last, or None when no high material is transparent"""
    amats = {m for _lo, hi, _c, _r in groups for m in hi.data.materials if retopo.material_is_alpha(m)}
    if not amats:
        return None
    scene = bpy.context.scene
    saved = [(hi, [sl.material for sl in hi.material_slots]) for _lo, hi, _c, _r in groups]
    tmp = {}

    def emit(m):
        if m not in tmp:
            e = bpy.data.materials.new('sr_alpha_emit')
            e.use_nodes = True
            nt = e.node_tree
            for n in list(nt.nodes):
                nt.nodes.remove(n)
            out = nt.nodes.new('ShaderNodeOutputMaterial')
            em = nt.nodes.new('ShaderNodeEmission')
            nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
            em.inputs['Color'].default_value = (1, 1, 1, 1)
            if m in amats:
                b = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
                src_n = b.inputs['Alpha'].links[0].from_node
                if src_n.type == 'TEX_IMAGE' and src_n.image:
                    # white where the card is opaque, see-through where it is not: a ray that passes a transparent
                    # part of a card goes on to what lies behind it (the solid body of a feather behind its fringe)
                    t = nt.nodes.new('ShaderNodeTexImage')
                    t.image = src_n.image
                    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
                    mix = nt.nodes.new('ShaderNodeMixShader')
                    nt.links.new(t.outputs['Alpha'], mix.inputs[0])
                    nt.links.new(tr.outputs['BSDF'], mix.inputs[1])
                    nt.links.new(em.outputs['Emission'], mix.inputs[2])
                    nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
            tmp[m] = e
        return tmp[m]

    for hi, mats in saved:
        for sl in hi.material_slots:
            sl.material = emit(sl.material)
    img = retopo._new_image('sr_alpha_src', size, True)
    img.pixels.foreach_set(np.zeros(size * size * 4, dtype=np.float32))
    lmat = groups[0][0].data.materials[0]
    node = lmat.node_tree.nodes.new('ShaderNodeTexImage')
    node.image = img
    prev_active = lmat.node_tree.nodes.active
    lmat.node_tree.nodes.active = node
    hidden = {o.name: o.hide_render for o in scene.objects if o.type == 'MESH'}
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 4
    try:
        for gi, (lo, hi, cage, mr) in enumerate(groups):
            for o in scene.objects:
                if o.type == 'MESH':
                    o.select_set(False)
                    o.hide_render = True
            for o in (hi, lo):
                o.hide_set(False)
                o.hide_render = False
                o.select_set(True)
            bpy.context.view_layer.objects.active = lo
            bpy.ops.object.bake(type='EMIT', use_selected_to_active=True, use_cage=True, cage_object=cage.name, max_ray_distance=mr,
                                margin=2, use_clear=gi == 0, target='IMAGE_TEXTURES')
        px = np.empty(size * size * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
        out = px.reshape(size, size, 4)[..., 0].copy()
    finally:
        for n, hr in hidden.items():
            if n in bpy.data.objects:
                bpy.data.objects[n].hide_render = hr
        for hi, mats in saved:
            for sl, m in zip(hi.material_slots, mats):
                sl.material = m
        lmat.node_tree.nodes.remove(node)
        lmat.node_tree.nodes.active = prev_active
        bpy.data.images.remove(img)
        for e in tmp.values():
            bpy.data.materials.remove(e)
    return out


def bake_and_skin(low, raw, src, group_names, group_polys, out_prefix, H, size=2048, ao=True, alpha_groups=(), log=print):
    """low: the whole new mesh (face attribute sr_group), unwrapped.  Every group is baked from its own
    source polygons and takes its skin weights from them.  -> (final low object, high object of all sources)"""
    w2r = retopo.source_faces(src.data)
    g = _group_faces(low)
    order = ['body'] + [n for n in group_names if n != 'body']
    parts, tmp = [], []
    mwr = np.array(raw.matrix_world)
    for name in order:
        gi = group_names.index(name)
        lo_g = retopo._dup(low, low.name + '_' + name)
        retopo._keep_faces(lo_g, set(np.flatnonzero(g == gi).tolist()))
        hi_g = retopo._dup(raw, 'high_' + name)
        retopo._keep_faces(hi_g, {int(w2r[f]) for f in group_polys[name]})
        # bake copy: split along folds (rim of a double sided sheet), cage from the source surface
        blow = retopo._dup(lo_g, lo_g.name + '_bake')
        bm = bmesh.new()
        bm.from_mesh(blow.data)
        bmesh.ops.split_edges(bm, edges=[e for e in bm.edges if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > math.radians(retopo.FOLD_ANGLE)])
        bm.to_mesh(blow.data)
        bm.free()
        hs = High(hi_g, n_samples=60000)
        cage, mr = make_cage(blow, hs.pts, H)
        parts.append((name, lo_g, hi_g, blow, cage, mr))
        tmp += [blow, cage]
    first = parts[0]
    more = [(p[3], [p[2]], p[4], p[5]) for p in parts[1:]]
    imgs = retopo.bake_maps(first[3], [first[2]], first[4], out_prefix, size=size, ao=ao, max_ray=first[5], more=more, gap_shade=1.0, log=log)
    mat = first[3].data.materials[0]
    # cloth sheets: texels no ray reached are holes (torn hems) -> alpha of the base colour
    if alpha_groups and 'baked_mask' in imgs:
        mask = imgs['baked_mask']
        hole = np.zeros_like(mask)
        for p in parts:
            if p[0] in alpha_groups:
                hole |= retopo._uv_mask(p[3], size) & ~mask
        def has_alpha(o):
            used = {pl.material_index for pl in o.data.polygons}
            return any(retopo.material_is_alpha(o.data.materials[i]) for i in used if i < len(o.data.materials))

        ag = [(p[3], p[2], p[4], p[5]) for p in parts if p[0] in alpha_groups and has_alpha(p[2])]
        try:
            a_src = bake_source_alpha(ag, size) if ag else None
        except Exception as ex:
            log('  WARN alpha of the source textures not baked: %s' % ex)
            a_src = None
        if a_src is not None:               # where the source card itself is transparent
            for p in ag:
                hole |= retopo._uv_mask(p[0], size) & mask & (a_src < 0.5)
        if hole.any():
            img = imgs['BC']
            px = np.empty(size * size * 4, dtype=np.float32)
            img.pixels.foreach_get(px)
            px = px.reshape(size, size, 4)
            px[..., 3] = np.where(hole, 0.0, 1.0)
            img.pixels.foreach_set(px.ravel())
            img.save()
            nt = mat.node_tree
            bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
            tex = next(n for n in nt.nodes if n.type == 'TEX_IMAGE' and n.image == img)
            nt.links.new(tex.outputs['Alpha'], bsdf.inputs['Alpha'])
            try:
                mat.surface_render_method = 'DITHERED'
            except Exception:
                pass
            log('  alpha holes: %.1f%% of the atlas' % (100 * hole.mean()))
    for name, lo_g, hi_g, blow, cage, mr in parts:
        retopo.transfer_skin(lo_g, hi_g)
    for o in tmp:
        bpy.data.objects.remove(o)
    # one mesh again
    final = parts[0][1]
    for o in bpy.context.scene.objects:
        o.select_set(False)
    for p in parts:
        p[1].hide_set(False)
        p[1].select_set(True)
    bpy.context.view_layer.objects.active = final
    if len(parts) > 1:
        bpy.ops.object.join()
    final.data.materials.clear()
    final.data.materials.append(mat)
    for p in final.data.polygons:
        p.material_index = 0
    # high model of everything that was a source (every polygon once, also when two groups bake from it)
    hi_all = retopo._dup(raw, 'high_body')
    retopo._keep_faces(hi_all, {int(w2r[f]) for name in order for f in group_polys[name]})
    for p in parts:
        bpy.data.objects.remove(p[2])
    retopo.finish_skin(final, hi_all)
    _fill_unskinned(final, log)
    bpy.data.objects.remove(low)
    return final, hi_all


def _fill_unskinned(obj, log=print):
    """vertices without any weight (the source has unweighted parts: hands without bones) take the weights of the
    nearest weighted vertex, so the whole mesh follows the skeleton"""
    me = obj.data
    has = [any(g.weight > 0 for g in v.groups) for v in me.vertices]
    todo = [i for i, h in enumerate(has) if not h]
    if not todo or not any(has):
        return 0
    kd = KDTree(sum(has))
    for v in me.vertices:
        if has[v.index]:
            kd.insert(v.co, v.index)
    kd.balance()
    for i in todo:
        j = kd.find(me.vertices[i].co)[1]
        for g in me.vertices[j].groups:
            if g.weight > 0:
                obj.vertex_groups[g.group].add([i], g.weight, 'REPLACE')
    log('  skin: %d vertices without weights took those of the nearest weighted vertex' % len(todo))
    return len(todo)


def retopo_part(raw, src, comps, part, budget, arm, out_prefix, exclude_bones=(), size=2048, ao=True, alpha_sheets=True, prop_sides=4, cloth_budget=None, log=print):
    """-> (low object, high object, report)"""
    t0 = time.time()
    faces = [f for c in comps if c.part == part for f in c.faces]
    if part == 'character':
        B, gp, info, sk = build_character(src, faces, arm, budget, exclude_bones, cloth_budget=cloth_budget, log=log)
        hh = info['height']
    else:
        B, gp, info, sk = build_prop(src, faces, budget, log=log)
        hh = info['height']
    low, names = builder_to_object(B, part + '_low')
    low.matrix_world = raw.matrix_world.copy()          # vertices were built in world space
    inv = np.array(raw.matrix_world.inverted())
    co = np.array([v.co[:] for v in low.data.vertices]) @ inv[:3, :3].T + inv[:3, 3]
    low.data.vertices.foreach_set('co', co.ravel())
    low.data.update()
    t1 = time.time()
    unwrap(low, boost=[i for i, n in enumerate(names) if n == 'sheets'] if part == 'character' else ())
    H = hh or float(np.linalg.norm(np.ptp(np.array([(low.matrix_world @ v.co)[:] for v in low.data.vertices]), axis=0)))
    alpha = ([e['name'] for e in info.get('extras', []) if e['kind'] == 'arc'] + ['sheets'] if part == 'character' else ['body', 'sheets']) if alpha_sheets else []
    alpha = [a for a in alpha if a in names]
    final, hi_all = bake_and_skin(low, raw, src, names, gp, out_prefix, H, size=size, ao=ao, alpha_groups=alpha, log=log)
    info['alpha_sheets'] = alpha
    rep = {'part': part, 'build': info, 'groups': list(names), 'seconds': {'geometry': round(t1 - t0, 1), 'uv_bake_skin': round(time.time() - t1, 1)}}
    return final, hi_all, rep
