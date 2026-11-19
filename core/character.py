"""Quad-preserving, loop-aware mesh reduction ("remove edge loops like an artist").

The only operation is the *ring collapse*: an edge ring (poly-chord) is the
set of edges reached from one edge by repeatedly stepping to the opposite edge
of a quad.  Collapsing every edge of a ring deletes one whole strip of quads
(= merges two neighbouring edge loops into one).  Rings end at triangles
(which are deleted with the strip), at mesh borders, or close on themselves.
Because a ring is closed under "opposite edge of a quad", no quad is ever
turned into a triangle and no n-gon is ever produced: quads stay quads, all
remaining edge loops stay complete loops, the triangle count can only go down.
(Two extras around it: a *partial* collapse removes a loop only along a stretch
and ends in triangles - last resort, see Params.tri_penalty; and after a
collapse two triangles that are just the halves of a quad get their diagonal
dissolved again, see Params.heal_quads.  Neither moves away from loop removal.)

Which ring goes first is decided by a cost made of
  * geometric error (quadrics of the ORIGINAL surface, relative to the local
    thickness of the part, so fingers are protected much more than the torso)
  * loop-spacing stretch relative to the artist's ORIGINAL spacing (so dense
    loops at joints are thinned proportionally instead of being wiped first)
  * protection weights (painted vertex group and/or auto detected joints)
UV seams / borders / hard edges are never crossed or merged: vertices on them
only slide along them and all per-corner UVs are interpolated wedge by wedge.
Optional mirror symmetry performs every collapse together with its mirror.
"""
import math
import numpy as np
from .rmesh import RMesh, ekey, BOUNDARY, NONMANIFOLD, SHARP, SEAM0

INF = float('inf')


class Params:
    def __init__(self, **kw):
        self.target_faces = None       # absolute polygon target (wins over ratio)
        self.ratio = 0.5
        self.symmetry = 'AUTO'         # 'AUTO' | 'X' | 'Y' | 'Z' | 'NONE'
        self.preserve_seams = True
        self.seam_layers = 'AUTO'      # 'AUTO' (skip junk layers) | 'ALL' | 'FIRST'
        self.preserve_sharp = True
        self.preserve_boundary = True
        self.protect = None            # per-vertex weights 0..1 (np array) or None
        self.protect_strength = 4.0
        self.auto_joints = True
        self.joint_strength = 5.0
        self.geo_eps = 0.08            # relative error that doubles the cost
        self.stretch_power = 2.0
        self.cost_mode = 'loops'       # 'loops' (characters) | 'qem' (static props)
        self.tube_aspect = 1.5         # limb segment length limit, in limb diameters
        self.texture = None            # (h,w,3) float array of the base colour texture (optional)
        self.tex_weight = 1.0          # weight of texture colour change (0..1) in the relative error
        self.border_weight = 2.5       # open borders are judged this much stricter than surface error
        self.max_warp = 35.0           # quads may not become more twisted than this (degrees; relaxed when stuck)
        self.max_rel_error = 0.3       # hard limit on measured error / feature size (relaxed only when stuck)
        self.max_feature = 0.035       # error is judged relative to min(local thickness, this * bbox diagonal)
        self.profile_protect = True    # silhouette / profile protection: a quad of the original that lies between a
                                       # concave and a convex crease is a step riser (heel block front, ledge).  Edges
                                       # that cross such a step are judged against the step height instead of the
                                       # (much larger) part thickness, so the step is not flattened into a ramp
        self.crease_angle = 60.0       # dihedral angle (degrees) from which an original edge counts as a crease
        self.profile_min = 0.008       # risers narrower than this * bbox diagonal are bevels, not steps: ignored.
                                       # (0.003 also keeps strap bands etc.; costs ~60 faces on Jier, see HANDOFF)
        self.uv_weight = 1.0           # weight of texture sliding vs geometric error
        self.partial = True            # allow partial ring collapses when stuck
        self.max_tri_ratio = None      # max fraction of triangles (None -> original)
        self.tri_penalty = 30.0        # cost factor for a partial collapse (a loop removed only along a stretch, ending
                                       # in a new triangle at each cut): cost *= 1 + tri_penalty * new_tris / faces_removed.
                                       # Every such cut puts a pole / slanted edge into clean quad flow and breaks the
                                       # loops that cross it, so it must be a last resort.  With the old value 2 partial
                                       # collapses competed with whole loops and left fans on the back of the skirt and
                                       # between the shoulder blades (artist review); 30..40 gives identical results on Jier
        self.heal_quads = True         # after a collapse, two triangles that are left as the two halves of a quad
                                       # (planar, convex, shared edge = the diagonal, no seam / hard edge on it) are
                                       # put back together by dissolving that diagonal.  No vertex moves or disappears;
                                       # it only gives the edge loops that ended at the two triangles their way through
                                       # again (artist-made "1 column -> 2 columns" fans degrade into such split quads
                                       # once the doubled columns are gone).  Triangles no collapse touched are left alone
        self.heal_angle = 20.0         # max angle (degrees) between the two triangles of a split quad
        self.stuck_relax_tris = False  # partial collapses may only add triangles up to the triangle share of the source
                                       # (max_tri_ratio).  True: when NOTHING else is left (every whole loop used up or
                                       # blocked: game-ready low poly meshes at 25-50%) that budget is raised step by
                                       # step instead of stopping.  Measured on the validation set: gets only 5-8 points
                                       # closer to the target (the rest is blocked by ring / joint guards and flips) and
                                       # costs quads (范海辛 character 79% -> 74% quads), so it is off
        self.relax_seams = False       # edge-collapse (prop) mode only.  False: UV seams are hard constraints.
                                       # True: a collapse that breaks a seam rule (moves a seam vertex off its seam,
                                       # merges two seams, moves a seam corner) is allowed but pays for the texture
                                       # sliding it causes (the moved distance counts like a geometric error, times
                                       # relax_weight).  Faces / UV triangles still may not flip.  Needed for very
                                       # strong reductions of meshes cut into many small UV islands, which otherwise
                                       # get stuck far above the target.  'STUCK': only once nothing else is possible
        self.relax_weight = 1.0
        self.group_targets = None      # edge-collapse mode: {material index: target face count}; every material is
                                       # reduced to its own budget (None: one global budget, i.e. uniform error)
        self.fit = True
        self.progress = None           # callback(fraction, message)
        for k, v in kw.items():
            if not hasattr(self, k):
                raise TypeError('unknown parameter ' + k)
            setattr(self, k, v)


# ----------------------------------------------------------------------------
class Reducer:
    def __init__(self, mesh: RMesh, params: Params):
        self.m = mesh
        self.p = params
        m = mesh
        nv = len(m.pos)
        self.orig_pos = m.pos.copy()
        self.orig_faces = [list(f) for f in m.faces]
        self.orig_fuv = [list(f) for f in m.fuv]
        self.orig_fcomp = list(m.fcomp)
        self.nfaces0 = m.alive_faces
        ext = m.pos.max(0) - m.pos.min(0)
        self.diag = float(np.linalg.norm(ext))
        self.relax = False             # see Params.relax_seams (switched on by run_edges)
        self.kinds = []
        self.kindmask = NONMANIFOLD
        if params.preserve_boundary:
            self.kindmask |= BOUNDARY
        self.kinds.append(BOUNDARY)       # borders are always a constraint kind
        if params.preserve_sharp and m.sharp:
            self.kinds.append(SHARP)
        self.seam_layers = []
        if params.preserve_seams:
            for k in range(m.n_layers):
                if params.seam_layers == 'ALL' or self._layer_is_sane(k) or \
                        (params.seam_layers == 'FIRST' and k == 0):
                    if params.seam_layers == 'FIRST' and k > 0:
                        continue
                    self.kinds.append(SEAM0 << k)
                    self.seam_layers.append(k)
        self.uv_cost_layer = self.seam_layers[0] if self.seam_layers else (0 if m.n_layers else None)
        self._cbvh = {}
        self.members = [[v] for v in range(nv)]   # original vertices merged into each vertex
        self.cost_override = {}
        self.vacc = np.zeros(nv)       # accumulated corner cutting per vertex
        self.uacc = np.zeros(nv)       # accumulated texture sliding per vertex
        self.vinfo = {}
        self.ecache = {}
        self.orig_boundary = {v for v in range(nv) if any(mk & BOUNDARY for mk in self.info(v).values())}

        self.blocked = set()
        self.l0 = {}
        for vs in m.faces:
            n = len(vs)
            for i in range(n):
                k = ekey(vs[i], vs[(i + 1) % n])
                if k not in self.l0:
                    self.l0[k] = float(np.linalg.norm(m.pos[k[0]] - m.pos[k[1]])) + 1e-12
        self._init_quadrics()
        self._init_thickness()
        self.protect = np.zeros(nv)
        if params.protect is not None:
            self.protect = np.clip(np.asarray(params.protect, dtype=np.float64), 0, 1) * params.protect_strength
        self.mirror = self._init_symmetry()
        self.joint = np.zeros(nv)
        self.ejoint = {}
        self.etube = {}
        from . import joints
        diam, score = joints.analyse(self)
        on_tube = ~np.isnan(diam)
        self.thick[on_tube] = np.clip(diam[on_tube], self.diag * 0.004, self.diag * self.p.max_feature)
        self.eprofile = {}
        if params.profile_protect:
            self.eprofile = self._init_relief()
        if params.auto_joints:
            self.ejoint = {k: v * params.joint_strength for k, v in score.items()}
        from collections import Counter
        self.fail_stats = Counter()
        self.fail_reason = ''
        self.comp_alive = [0] * m.n_comp
        for fid, vs in enumerate(m.faces):
            self.comp_alive[m.fcomp[fid]] += 1
        self.log = []
        self.band_quads = set()

    # ------------------------------------------------------------- set-up
    def _layer_is_sane(self, k):
        """A UV layer with many zero-area faces is left-over junk: its UVs are
        still interpolated but its 'seams' do not constrain the reduction."""
        m = self.m
        zero = 0
        for uv in m.fuv:
            n = len(uv)
            s = 0.0
            for i in range(n):
                x0, y0 = uv[i][k]
                x1, y1 = uv[(i + 1) % n][k]
                s += x0 * y1 - x1 * y0
            if abs(s) < 1e-12:
                zero += 1
        return zero <= 0.05 * len(m.fuv)

    def _init_quadrics(self):
        m = self.m
        nv = len(m.pos)
        self.Q = np.zeros((nv, 4, 4))
        self.w = np.zeros(nv)
        for fid, vs in enumerate(m.faces):
            n = m.face_normal(vs)
            a2 = float(np.linalg.norm(n))
            if a2 < 1e-20:
                continue
            n = n / a2
            area = a2 * 0.5
            c = m.pos[vs].mean(0)
            pl = np.append(n, -float(n @ c))
            K = np.outer(pl, pl) * area
            for v in vs:
                self.Q[v] += K
                self.w[v] += area
        # border / seam / sharp edges: planes through the edge, perpendicular
        # to the adjacent face, keep the feature curve where it is.
        for v in range(nv):
            info = self.info(v)
            for x, mask in info.items():
                if x < v or not (mask & ~NONMANIFOLD):
                    continue
                wgt = 8.0 if mask & BOUNDARY else 2.0
                d = m.pos[x] - m.pos[v]
                L = float(np.linalg.norm(d))
                if L < 1e-12:
                    continue
                for fid in m.vf[v] & m.vf[x]:
                    fn = m.face_normal(m.faces[fid])
                    pn = np.cross(d, fn)
                    l = float(np.linalg.norm(pn))
                    if l < 1e-20:
                        continue
                    pn /= l
                    pl = np.append(pn, -float(pn @ m.pos[v]))
                    K = np.outer(pl, pl) * (wgt * L * L * 0.5)
                    self.Q[v] += K
                    self.Q[x] += K
        self.w = np.maximum(self.w, 1e-12)

    def _init_thickness(self):
        """Local thickness (shape diameter) per vertex: ray casts against the
        original surface along the inward normal."""
        from mathutils.bvhtree import BVHTree
        from mathutils import Vector
        m = self.m
        nv = len(m.pos)
        vn = np.zeros((nv, 3))
        for vs in m.faces:
            n = m.face_normal(vs)
            for v in vs:
                vn[v] += n
        ln = np.linalg.norm(vn, axis=1)
        ln[ln < 1e-20] = 1
        vn /= ln[:, None]
        self.vn0 = vn
        bvh = BVHTree.FromPolygons([tuple(p) for p in m.pos], [tuple(f) for f in m.faces], all_triangles=False)
        self.bvh0 = bvh
        thick = np.full(nv, np.nan)
        eps = self.diag * 1e-4
        # a small cone of directions around the inward normal
        for v in range(nv):
            n = vn[v]
            t1 = np.cross(n, (1.0, 0, 0) if abs(n[0]) < 0.9 else (0, 1.0, 0))
            t1 /= np.linalg.norm(t1) + 1e-20
            t2 = np.cross(n, t1)
            o = Vector(m.pos[v] - n * eps)
            ds = []
            for (a, b) in ((0, 0), (0.35, 0), (-0.35, 0), (0, 0.35), (0, -0.35)):
                d = -n + a * t1 + b * t2
                d /= np.linalg.norm(d)
                hit = bvh.ray_cast(o, Vector(d))
                if hit[0] is not None:
                    # only count hits on the inner side of another face
                    if float(np.dot(np.array(hit[1]), d)) > 0 or True:
                        ds.append(hit[3])
            if ds:
                thick[v] = float(np.median(ds))
        # fallback: component size
        comp_of_v = np.full(nv, -1)
        for fid, vs in enumerate(m.faces):
            for v in vs:
                comp_of_v[v] = m.fcomp[fid]
        for c in range(m.n_comp):
            idx = np.where(comp_of_v == c)[0]
            if len(idx) == 0:
                continue
            ext = m.pos[idx].max(0) - m.pos[idx].min(0)
            size = float(np.sort(ext)[1])          # middle extent of the part
            t = thick[idx]
            t[np.isnan(t)] = size
            thick[idx] = np.minimum(t, max(size, 1e-9))
        thick[np.isnan(thick)] = self.diag * 0.05
        # smooth a little over the 1-ring, clamp
        nb = [set() for _ in range(nv)]
        for vs in m.faces:
            n = len(vs)
            for i in range(n):
                nb[vs[i]].add(vs[(i + 1) % n])
                nb[vs[(i + 1) % n]].add(vs[i])
        for _ in range(2):
            t2 = thick.copy()
            for v in range(nv):
                if nb[v]:
                    t2[v] = 0.5 * thick[v] + 0.5 * float(np.median([thick[x] for x in nb[v]]))
            thick = t2
        self.thick = np.clip(thick, self.diag * 0.004, self.diag * self.p.max_feature)

    def _init_relief(self):
        """Profile feature size per edge (edge key -> size), see Params.profile_protect.

        The shape diameter of a boot is the boot, so the 3 cm step of its heel
        block could be flattened into a ramp with a 'small' relative error.  A
        quad of the original mesh between a concave and an opposite convex
        crease is a step riser; its crease-to-crease width is the size that
        errors of the edges at its vertices are judged against.  Edges running
        along the two creases themselves (they only round the outline of the
        step) keep the part thickness."""
        m = self.m
        pos = m.pos
        lo = self.diag * self.p.profile_min
        hi = self.diag * self.p.max_feature
        cos_c = math.cos(math.radians(self.p.crease_angle))
        fn = {}
        fc = {}
        ef = {}
        for fid, vs in enumerate(m.faces):
            if vs is None:
                continue
            n = m.face_normal(vs)
            l = float(np.linalg.norm(n))
            fn[fid] = n / l if l > 1e-20 else None
            fc[fid] = pos[vs].mean(0)
            for i in range(len(vs)):
                ef.setdefault(ekey(vs[i], vs[(i + 1) % len(vs)]), []).append(fid)
        crease = {}                    # edge -> +1 convex, -1 concave
        for k, fs in ef.items():
            if len(fs) != 2 or fn[fs[0]] is None or fn[fs[1]] is None:
                continue
            if float(fn[fs[0]] @ fn[fs[1]]) < cos_c:
                crease[k] = -1 if float((fc[fs[1]] - fc[fs[0]]) @ fn[fs[0]]) > 0 else 1
        relief = {}                    # vertex -> smallest riser width
        along = set()                  # the creases that bound a riser
        cross = set()                  # riser edges from crease to crease
        for fid, vs in enumerate(m.faces):
            if vs is None or len(vs) != 4:
                continue
            for i in (0, 1):
                a, b, c, d = vs[i], vs[(i + 1) % 4], vs[(i + 2) % 4], vs[(i + 3) % 4]
                s1, s2 = crease.get(ekey(a, b)), crease.get(ekey(c, d))
                if s1 is None or s2 is None or s1 == s2:
                    continue
                wdt = 0.5 * (float(np.linalg.norm(pos[b] - pos[c])) + float(np.linalg.norm(pos[d] - pos[a])))
                if wdt < lo or wdt >= hi:
                    continue
                wdt = max(wdt, self.diag * 0.004)      # same floor as the thickness
                along.add(ekey(a, b))
                along.add(ekey(c, d))
                cross.add(ekey(b, c))
                cross.add(ekey(d, a))
                for v in vs:
                    if wdt < relief.get(v, hi):
                        relief[v] = wdt
        ep = {}
        for k in ef:
            if k in along and k not in cross:
                continue
            r = min(relief.get(k[0], hi), relief.get(k[1], hi))
            if r < hi:
                ep[k] = float(r)
        return ep

    def _init_symmetry(self):
        nv = len(self.m.pos)
        mirror = np.full(nv, -1, dtype=np.int64)
        self.sym_axis = None
        mode = self.p.symmetry
        if mode in (None, 'NONE'):
            return mirror
        from mathutils import kdtree
        pos = self.m.pos
        kd = kdtree.KDTree(nv)
        for i in range(nv):
            kd.insert(pos[i], i)
        kd.balance()
        tol = self.diag * 2e-4
        best = None
        for ax, name in enumerate('XYZ'):
            if mode not in ('AUTO', name):
                continue
            mir = np.full(nv, -1, dtype=np.int64)
            q = pos.copy()
            q[:, ax] *= -1
            for i in range(nv):
                co, j, d = kd.find(q[i])
                if d < tol:
                    mir[i] = j
            # must be an involution
            for i in range(nv):
                j = mir[i]
                if j >= 0 and mir[j] != i:
                    mir[i] = -1
            n = int((mir >= 0).sum())
            if best is None or n > best[0]:
                best = (n, ax, mir)
        if best is None:
            return mirror
        n, ax, mir = best
        if mode == 'AUTO' and n < 0.3 * nv:
            return mirror
        self.sym_axis = ax
        return mir

    # ----------------------------------------------------------- topology
    def info(self, v):
        """neighbour vertex -> feature mask of the edge (v, neighbour)."""
        r = self.vinfo.get(v)
        if r is not None:
            return r
        m = self.m
        acc = {}
        for fid in m.vf[v]:
            vs = m.faces[fid]
            n = len(vs)
            i = vs.index(v)
            uv = m.fuv[fid]
            for j in ((i + 1) % n, (i - 1) % n):
                acc.setdefault(vs[j], []).append((uv[i], uv[j]))
        r = {}
        for x, lst in acc.items():
            if len(lst) == 1:
                mask = BOUNDARY
            elif len(lst) > 2:
                mask = NONMANIFOLD
            else:
                mask = 0
                (a0, b0), (a1, b1) = lst
                if a0 != a1 or b0 != b1:
                    for k in range(m.n_layers):
                        if a0[k] != a1[k] or b0[k] != b1[k]:
                            mask |= SEAM0 << k
            if ekey(v, x) in m.sharp:
                mask |= SHARP
            r[x] = mask
        self.vinfo[v] = r
        return r

    def rings(self):
        """Edge rings as lists of edge keys (union of opposite edges in quads)."""
        m = self.m
        par = {}

        def find(x):
            r = x
            while par[r] != r:
                r = par[r]
            while par[x] != r:
                par[x], x = r, par[x]
            return r
        for vs in m.faces:
            if vs is None:
                continue
            n = len(vs)
            ks = [ekey(vs[i], vs[(i + 1) % n]) for i in range(n)]
            for k in ks:
                if k not in par:
                    par[k] = k
            if n == 4:
                a, b = find(ks[0]), find(ks[2])
                if a != b:
                    par[a] = b
                a, b = find(ks[1]), find(ks[3])
                if a != b:
                    par[a] = b
        out = {}
        for k in par:
            out.setdefault(find(k), []).append(k)
        return list(out.values())

    # --------------------------------------------------------------- cost
    def eval_edge(self, a, b):
        """-> (cost, t, p) for collapsing b into a at a + t*(b-a), or None."""
        key = (a, b)
        r = self.ecache.get(key, 0)
        if r != 0:
            return r
        r = self._eval_edge(a, b)
        self.ecache[key] = r
        return r

    def _constraint(self, a, b, ia, ib):
        """allowed t: None (forbidden), 'free', 0.0 or 1.0"""
        emask = ia[b]
        if emask & NONMANIFOLD:
            return None
        t = 'free'
        relax = self.relax
        self._relaxed = False
        for kbit in self.kinds:
            na = nb = 0
            for msk in ia.values():
                if msk & kbit:
                    na += 1
            for msk in ib.values():
                if msk & kbit:
                    nb += 1
            if na == 0 and nb == 0:
                continue
            ca = 0 if na == 0 else (1 if na == 2 else 2)
            cb = 0 if nb == 0 else (1 if nb == 2 else 2)
            soft = relax and kbit >= SHARP        # hard edges and UV seams (never borders)
            pay = kbit >= SEAM0                   # only a broken seam rule slides texture
            if emask & kbit:
                if ca == 2 and cb == 2:
                    if not soft:
                        return None
                    self._relaxed |= pay
                    continue
                want = 0.0 if ca == 2 else (1.0 if cb == 2 else 'free')
            else:
                if ca > 0 and cb > 0:
                    if not soft:
                        return None
                    self._relaxed |= pay
                    want = 0.0 if ca > cb else (1.0 if cb > ca else 'free')
                else:
                    want = 0.0 if ca > 0 else 1.0
            if want != 'free':
                if t != 'free' and t != want:
                    if not soft:
                        return None
                    self._relaxed |= pay
                    continue
                t = want
        # triangles on the edge: their two other edges fuse -> must not both
        # be features of the same kind; other common neighbours break manifoldness
        m = self.m
        common = [c for c in ia if c in ib]
        if common:
            ntri = 0
            for fid in m.vf[a] & m.vf[b]:
                vs = m.faces[fid]
                if len(vs) == 3:
                    ntri += 1
            if len(common) != ntri:
                return None
            for c in common:
                if ia[c] & ib[c] & ~0:
                    for kbit in self.kinds:
                        if ia[c] & ib[c] & kbit:
                            if relax and kbit >= SHARP:
                                self._relaxed |= kbit >= SEAM0
                                continue
                            return None
                if (ia[c] | ib[c]) & NONMANIFOLD:
                    return None
        return t

    def _eval_edge(self, a, b):
        m = self.m
        ia, ib = self.info(a), self.info(b)
        if b not in ia:
            return None
        t = self._constraint(a, b, ia, ib)
        if t is None:
            return None
        relaxed = self._relaxed
        if self.relax and t == 'free' and (relaxed or any(mk & ~(BOUNDARY | NONMANIFOLD) for mk in ia.values())
                                           or any(mk & ~(BOUNDARY | NONMANIFOLD) for mk in ib.values())):
            # a relaxed collapse keeps one end point: wedges without a partner
            # across the edge keep their UV, so only one side slides
            ha, hb = np.append(m.pos[a], 1.0), np.append(m.pos[b], 1.0)
            Qs = self.Q[a] + self.Q[b]
            t = 0.0 if float(ha @ Qs @ ha) <= float(hb @ Qs @ hb) else 1.0
        ma, mb = self.mirror[a], self.mirror[b]
        if ma == b and mb == a:           # edge crosses the mirror plane
            if t == 'free':
                t = 0.5
            else:
                return None
        pa, pb = m.pos[a], m.pos[b]
        d = pb - pa
        Q = self.Q[a] + self.Q[b]
        if t == 'free':
            A = Q[:3, :3]
            bv = Q[:3, 3]
            dAd = float(d @ A @ d)
            L2 = float(d @ d)
            wsum = self.w[a] + self.w[b]
            reg = 0.01 * wsum * L2
            # minimise quadric + reg*(t-0.5)^2 along the edge
            num = -(float(d @ A @ pa) + float(bv @ d)) + reg * 0.5
            den = dAd + reg
            t = 0.5 if den < 1e-30 else min(1.0, max(0.0, num / den))
            # keep it away from tiny slivers of movement: snap
            if t < 0.08:
                t = 0.0
            elif t > 0.92:
                t = 1.0
        p = pa + t * d
        ph = np.append(p, 1.0)
        err = float(ph @ Q @ ph)
        dist = math.sqrt(max(err, 0.0) / (self.w[a] + self.w[b]))
        tau = min(self.thick[a], self.thick[b])
        if self.eprofile:
            tau = min(tau, self.eprofile.get(ekey(a, b), tau))
        rel = dist / tau
        if self.p.cost_mode == 'qem':
            # static props: plain quadric error + texture sliding, no loop logic
            dev, shift = self._slide_error(a, b, t, p, ia, ib)
            wsum = self.w[a] + self.w[b]
            cost = max(err, 0.0) + wsum * ((self.p.uv_weight * shift) ** 2 + 1e-4 * float(d @ d))
            if relaxed:
                cost += wsum * self.p.relax_weight ** 2 * float(d @ d)
            return (cost, t, p, dist, shift, 1.0, tau, 0.0)
        # stretch of surviving edges relative to the original loop spacing
        S = 1.0
        l0 = self.l0
        pos = m.pos
        px, py, pz = p
        for x in ia:
            if x != b and x not in ib:
                q = pos[x]
                s = math.sqrt((px - q[0]) ** 2 + (py - q[1]) ** 2 + (pz - q[2]) ** 2) / l0[ekey(a, x)]
                if s > S:
                    S = s
        for x in ib:
            if x != a and x not in ia:
                q = pos[x]
                s = math.sqrt((px - q[0]) ** 2 + (py - q[1]) ** 2 + (pz - q[2]) ** 2) / l0[ekey(b, x)]
                if s > S:
                    S = s
        # limb segments must not get much longer than the limb is thick,
        # otherwise there is nothing left to bend
        dt = self.etube.get(ekey(a, b))
        if dt is not None:
            et = self.etube
            Lmax = 0.0
            for x in ia:
                if x != b and x not in ib and ekey(a, x) in et:
                    q = pos[x]
                    Lmax = max(Lmax, math.sqrt((px - q[0]) ** 2 + (py - q[1]) ** 2 + (pz - q[2]) ** 2))
            for x in ib:
                if x != a and x not in ia and ekey(b, x) in et:
                    q = pos[x]
                    Lmax = max(Lmax, math.sqrt((px - q[0]) ** 2 + (py - q[1]) ** 2 + (pz - q[2]) ** 2))
            S *= (1.0 + (Lmax / (self.p.tube_aspect * dt)) ** 6) ** (1.0 / self.p.stretch_power)
        # corner cutting + texture sliding of this step: where does the old
        # vertex end up relative to the new edge (opposite neighbour -> p)?
        dev, shift = self._slide_error(a, b, t, p, ia, ib)
        dacc = max(self.vacc[a], self.vacc[b]) + dev
        sacc = max(self.uacc[a], self.uacc[b]) + shift
        rel = max(dist, dacc) / tau + self.p.uv_weight * sacc / tau
        prot = 1.0 + max(self.protect[a], self.protect[b]) + self.ejoint.get(ekey(a, b), 0.0)
        base = S ** self.p.stretch_power * prot
        cost = base * (1.0 + (rel / self.p.geo_eps) ** 2)
        return (cost, t, p, dacc, sacc, base, tau, self.p.uv_weight * sacc / tau)

    def _slide_error(self, a, b, t, p, ia, ib):
        m = self.m
        pos = m.pos
        kuv = self.uv_cost_layer
        dev_max = 0.0
        shift_max = 0.0
        for v, o, tt, iv, io in ((a, b, t, ia, ib), (b, a, 1.0 - t, ib, ia)):
            if tt <= 0.0:
                continue
            pv = pos[v]
            dv = pv - pos[o]
            ldv = math.sqrt(float(dv @ dv)) or 1e-30
            best = None
            bc = 0.3
            for x in iv:
                if x == o or x in io:
                    continue
                e = pos[x] - pv
                le = math.sqrt(float(e @ e)) or 1e-30
                c = float(e @ dv) / (le * ldv)
                if c > bc:
                    bc = c
                    best = x
            if best is None:
                continue
            x = best
            px = pos[x]
            seg = p - px
            l2 = float(seg @ seg)
            if l2 < 1e-30:
                continue
            sp = float((pv - px) @ seg) / l2
            sp = 0.0 if sp < 0 else (1.0 if sp > 1 else sp)
            q = px + sp * seg - pv
            d = math.sqrt(float(q @ q))
            if d > dev_max:
                dev_max = d
            if kuv is None:
                continue
            # uv predicted at the old vertex by the new edge x -> p
            fe = m.vf[v] & m.vf[o]
            for f in m.vf[v] & m.vf[x]:
                vs = m.faces[f]
                uv = m.fuv[f]
                uv_v = uv[vs.index(v)][kuv]
                uv_x = uv[vs.index(x)][kuv]
                part = None
                for g in fe:
                    gvs = m.faces[g]
                    guv = m.fuv[g]
                    if guv[gvs.index(v)][kuv] == uv_v:
                        part = guv[gvs.index(o)][kuv]
                        break
                if part is None:
                    continue
                unx = uv_v[0] + tt * (part[0] - uv_v[0])
                uny = uv_v[1] + tt * (part[1] - uv_v[1])
                ex = uv_x[0] + sp * (unx - uv_x[0]) - uv_v[0]
                ey = uv_x[1] + sp * (uny - uv_x[1]) - uv_v[1]
                du = math.sqrt((uv_x[0] - uv_v[0]) ** 2 + (uv_x[1] - uv_v[1]) ** 2)
                lx = math.sqrt(float((px - pv) @ (px - pv)))
                if du < 1e-9 or lx < 1e-12:
                    continue
                sh = math.sqrt(ex * ex + ey * ey) / (du / lx)
                if sh > shift_max:
                    shift_max = sh
        return dev_max, shift_max

    # -------------------------------------------------------------- apply
    def collapse_edges(self, edges):
        """Collapse a set of vertex-disjoint edges [(a, b, t, p)], b into a.
        Returns an undo record, or None (nothing changed) if invalid."""
        m = self.m
        nl = m.n_layers
        affected = set()
        for a, b, t, p in edges:
            affected |= m.vf[a]
            affected |= m.vf[b]
        touched = set()
        for fid in affected:
            touched.update(m.faces[fid])
        # ---- undo record
        undo = {
            'faces': {fid: (m.faces[fid], m.fuv[fid]) for fid in affected},
            'vf': {v: set(m.vf[v]) for v in touched},
            'pos': {}, 'Q': {}, 'w': {}, 'thick': {}, 'prot': {}, 'joint': {}, 'mirror': {}, 'vacc': {}, 'uacc': {},
            'l0': {}, 'sharp': None, 'alive': m.alive_faces, 'members': {},
        }
        old_normals = {fid: m.face_normal(m.faces[fid]) for fid in affected}
        old_warp = {fid: self._warp(m.faces[fid]) for fid in affected if len(m.faces[fid]) == 4}
        old_uvarea = {fid: self._uv_areas(m.fuv[fid]) for fid in affected}
        fan_before = {v: self._fan_components(v) for v in touched}
        rep = {}
        newuv = {}        # (fid, corner index) -> uv tuple
        for a, b, t, p in edges:
            rep[b] = a
            # wedge maps along the edge, per layer
            Ma = [dict() for _ in range(nl)]
            Mb = [dict() for _ in range(nl)]
            for fid in m.vf[a] & m.vf[b]:
                vs = m.faces[fid]
                uv = m.fuv[fid]
                ua, ub = uv[vs.index(a)], uv[vs.index(b)]
                for k in range(nl):
                    Ma[k][ua[k]] = ub[k]
                    Mb[k][ub[k]] = ua[k]
            for v, M, tt in ((a, Ma, t), (b, Mb, 1.0 - t)):
                # corner at v keeps (1-tt) of its own uv and tt of its partner
                for fid in m.vf[v]:
                    vs = m.faces[fid]
                    i = vs.index(v)
                    cur = m.fuv[fid][i]
                    if tt == 0.0:
                        continue
                    out = []
                    for k in range(nl):
                        x = cur[k]
                        y = M[k].get(x)
                        if y is None:
                            out.append(x)
                        elif tt == 1.0:
                            out.append(y)
                        else:
                            out.append((x[0] + tt * (y[0] - x[0]), x[1] + tt * (y[1] - x[1])))
                    newuv[(fid, i)] = tuple(out)
        # ---- original loop spacing / sharp flags of edges that get re-keyed
        groups = {}
        for a, b, t, p in edges:
            for v in (a, b):
                for x in self.info(v):
                    k = ekey(v, x)
                    self.ecache.pop((v, x), None)
                    self.ecache.pop((x, v), None)
                    r0, r1 = rep.get(k[0], k[0]), rep.get(k[1], k[1])
                    if r0 == r1:
                        continue
                    groups.setdefault(ekey(r0, r1), set()).add(k)
        sharp_new = None
        for nk, olds in groups.items():
            if len(olds) == 1 and nk in olds:
                continue
            undo['l0'][nk] = self.l0.get(nk)
            if m.sharp and any(k in m.sharp for k in olds):
                if sharp_new is None:
                    sharp_new = set(m.sharp)
                sharp_new.add(nk)
        l0new = {nk: sum(self.l0[k] for k in olds) / len(olds) for nk, olds in groups.items() if nk in undo['l0']}
        et = self.etube
        if et:
            etnew = {}
            for nk in l0new:
                ds = [et[k] for k in groups[nk] if k in et]
                if ds and et.get(nk) != min(ds):
                    etnew[nk] = min(ds)
            undo['et'] = {nk: et.get(nk) for nk in etnew}
            et.update(etnew)
        ep = self.eprofile
        if ep:
            epnew = {}
            for nk in l0new:
                ds = [ep[k] for k in groups[nk] if k in ep]
                if ds and ep.get(nk) != min(ds):
                    epnew[nk] = min(ds)
            undo['ep'] = {nk: ep.get(nk) for nk in epnew}
            ep.update(epnew)
        ej = self.ejoint
        ejnew = {}
        if ej:
            for nk in l0new:
                j = max(ej.get(k, 0.0) for k in groups[nk])
                if j != ej.get(nk, 0.0):
                    ejnew[nk] = j
            undo['ej'] = {nk: ej.get(nk) for nk in ejnew}
        # ---- rebuild faces
        removed = []
        for fid in affected:
            vs = m.faces[fid]
            uv = m.fuv[fid]
            nvs = []
            nuv = []
            for i, v in enumerate(vs):
                r = rep.get(v, v)
                u = newuv.get((fid, i), uv[i])
                if nvs and nvs[-1] == r:
                    continue
                nvs.append(r)
                nuv.append(u)
            if len(nvs) > 1 and nvs[0] == nvs[-1]:
                nvs.pop()
                nuv.pop()
            if len(nvs) < 3:
                removed.append(fid)
                m.faces[fid] = None
                m.fuv[fid] = None
            else:
                m.faces[fid] = nvs
                m.fuv[fid] = nuv
        for fid in removed:
            for v in undo['faces'][fid][0]:
                m.vf[v].discard(fid)
            self.comp_alive[m.fcomp[fid]] -= 1
        undo['removed'] = removed
        m.alive_faces -= len(removed)
        for a, b, t, p in edges:
            for arr, name in ((m.pos, 'pos'), (self.Q, 'Q'), (self.w, 'w'), (self.thick, 'thick'),
                              (self.protect, 'prot'), (self.joint, 'joint'), (self.mirror, 'mirror'), (self.vacc, 'vacc'), (self.uacc, 'uacc')):
                undo[name][a] = arr[a].copy()
                undo[name][b] = arr[b].copy()
            m.vf[a] |= m.vf[b]
            m.vf[b] = set()
            undo['members'][a] = self.members[a]
            self.members[a] = self.members[a] + self.members[b]
            m.pos[a] = p
            self.Q[a] += self.Q[b]
            self.w[a] += self.w[b]
            self.thick[a] = (1 - t) * self.thick[a] + t * self.thick[b]
            self.protect[a] = max(self.protect[a], self.protect[b])
            self.joint[a] = max(self.joint[a], self.joint[b])
        self.l0.update(l0new)
        if ejnew:
            ej.update(ejnew)
        if sharp_new is not None:
            undo['sharp'] = m.sharp
            m.sharp = sharp_new
        undo['touched'] = touched
        # invalidate caches
        for v in touched:
            self.vinfo.pop(v, None)
        # ---- validation
        ok = True
        seen = set()
        for fid in removed:
            if self.comp_alive[m.fcomp[fid]] < 2:
                ok = False; self.fail_reason = 'part'        # never delete (or reduce to one face) a whole part
        for fid in (affected if ok else ()):
            vs = m.faces[fid]
            if vs is None:
                continue
            if len(set(vs)) != len(vs):
                ok = False; self.fail_reason = 'dupvert'
                break
            fs = frozenset(vs)
            if fs in seen:
                ok = False; self.fail_reason = 'dupface'
                break
            seen.add(fs)
            n0 = old_normals[fid]
            n1 = m.face_normal(vs)
            l0 = float(np.linalg.norm(n0))
            l1 = float(np.linalg.norm(n1))
            if l1 < 1e-14 or (l0 > 1e-14 and float(n0 @ n1) < 0.2 * l0 * l1):
                ok = False; self.fail_reason = 'flip'
                break
            if len(vs) == 4 and not self._quad_ok(vs, n1 / l1):
                ok = False; self.fail_reason = 'concave'
                break
            ua0 = old_uvarea[fid]
            ua1 = self._uv_areas(m.fuv[fid])
            for k in self.seam_layers:
                if abs(ua0[k]) > 1e-12 and ua0[k] * ua1[k] <= 0:
                    ok = False; self.fail_reason = 'uvflip'
                    break
            if not ok:
                break
        if ok:
            for v in touched:
                if not m.vf[v]:
                    continue
                info = self.info(v)
                nf = len(m.vf[v])
                for x, mask in info.items():
                    if mask & NONMANIFOLD:
                        ok = False; self.fail_reason = 'nonmanifold'
                        break
                if not ok:
                    break
                if self._fan_components(v) > max(1, fan_before[v]):
                    ok = False; self.fail_reason = 'fan'
                    break
                if nf < 3 and not any(mk & BOUNDARY for mk in info.values()):
                    ok = False; self.fail_reason = 'doublet'        # doublet
                    break
                if nf == 1 and len(info) == 2 and self.p.preserve_boundary and False:
                    ok = False; self.fail_reason = 'x'
                    break
        if not ok:
            self.undo(undo)
            return None
        # how much worse did the flatness of the quads get (degrees)
        w = 0.0
        for fid, w0 in old_warp.items():
            vs = m.faces[fid]
            if vs is not None and len(vs) == 4:
                w1 = self._warp(vs)
                if w1 > w0 + 8.0 and w1 > w:
                    w = w1
        self.last_warp = w
        return undo

    def _warp(self, vs):
        """non-planarity of a quad: angle between the two triangles (degrees)"""
        p = self.m.pos
        a, b, c, d = p[vs[0]], p[vs[1]], p[vs[2]], p[vs[3]]
        best = 0.0
        for (n1, n2) in ((np.cross(b - a, c - a), np.cross(c - a, d - a)),
                         (np.cross(b - a, d - a), np.cross(c - b, d - b))):
            l = float(np.linalg.norm(n1)) * float(np.linalg.norm(n2))
            if l < 1e-20:
                continue
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(n1 @ n2) / l))))
            if ang > best:
                best = ang
        return best

    def true_error(self, a):
        """Measured two-sided distance between the faces around vertex a and
        the part of the original surface they replace."""
        from mathutils.geometry import closest_point_on_tri
        from mathutils import Vector
        m = self.m
        pos = m.pos
        tris = []
        e2 = 0.0
        nearest = self.bvh0.find_nearest
        for fid in m.vf[a]:
            vs = m.faces[fid]
            pts = [Vector(pos[v]) for v in vs]
            c = Vector(pos[vs].mean(0))
            hit = nearest(c)
            if hit[3] is not None and hit[3] > e2:
                e2 = hit[3]
            for i in range(1, len(vs) - 1):
                tris.append((pts[0], pts[i], pts[i + 1]))
        hit = nearest(Vector(pos[a]))
        if hit[3] is not None and hit[3] > e2:
            e2 = hit[3]
        e1 = 0.0
        op = self.orig_pos
        for ov in self.members[a]:
            q = Vector(op[ov])
            best = 1e30
            for t0, t1, t2 in tris:
                d = (closest_point_on_tri(q, t0, t1, t2) - q).length_squared
                if d < best:
                    best = d
            if best < 1e29 and best > e1:
                e1 = best
        err = max(math.sqrt(e1), e2)
        # open borders (skirt hem, sleeve, hair tips) are silhouette: the border
        # curve itself must stay close, judged more strictly than the surface
        segs = [x for x, mask in self.info(a).items() if mask & BOUNDARY]
        if segs:
            pa = Vector(pos[a])
            ends = [Vector(pos[x]) for x in segs]
            ob = self.orig_boundary
            eb = 0.0
            for ov in self.members[a]:
                if ov not in ob:
                    continue
                q = Vector(op[ov])
                best = 1e30
                for e in ends:
                    d = e - pa
                    l2 = d.length_squared
                    t = 0.0 if l2 < 1e-30 else max(0.0, min(1.0, (q - pa).dot(d) / l2))
                    dd = (pa + d * t - q).length_squared
                    if dd < best:
                        best = dd
                if best < 1e29 and best > eb:
                    eb = best
            err = max(err, self.p.border_weight * math.sqrt(eb))
        return err

    def _comp_bvh(self, c):
        ent = self._cbvh.get(c)
        if ent is None:
            from mathutils.bvhtree import BVHTree
            fids = [f for f, cc in enumerate(self.orig_fcomp) if cc == c]
            vids = sorted({v for f in fids for v in self.orig_faces[f]})
            loc = {v: i for i, v in enumerate(vids)}
            bvh = BVHTree.FromPolygons([tuple(self.orig_pos[v]) for v in vids],
                                       [tuple(loc[v] for v in self.orig_faces[f]) for f in fids], all_triangles=False)
            ent = (bvh, fids)
            self._cbvh[c] = ent
        return ent

    def tex_error(self, a):
        """Mean colour difference (0..1) between the texture as mapped by the
        faces around vertex a and the texture the original surface shows at
        the same places."""
        tex = self.p.texture
        k = self.uv_cost_layer
        if tex is None or k is None:
            return 0.0
        from . import texture as texmod
        m = self.m
        pos = m.pos
        op = self.orig_pos
        uv_new = []
        uv_old = []
        for fid in m.vf[a]:
            vs = m.faces[fid]
            n = len(vs)
            fuv = m.fuv[fid]
            P = pos[vs]
            U = np.array([c[k] for c in fuv])
            pc = P.mean(0)
            uc = U.mean(0)
            bvh, fids = self._comp_bvh(m.fcomp[fid])
            pts = [(pc, uc)]
            for i in range(n):
                pts.append((0.5 * (pc + P[i]), 0.5 * (uc + U[i])))
                pts.append((0.25 * pc + 0.375 * (P[i] + P[(i + 1) % n]), 0.25 * uc + 0.375 * (U[i] + U[(i + 1) % n])))
            for pp, uu in pts:
                hit = bvh.find_nearest(pp)
                if hit[0] is None:
                    continue
                of = fids[hit[2]]
                ovs = self.orig_faces[of]
                ouv = self.orig_fuv[of]
                q = np.array(hit[0])
                best = None
                for i in range(1, len(ovs) - 1):
                    A, B, C = op[ovs[0]], op[ovs[i]], op[ovs[i + 1]]
                    v0, v1, v2 = B - A, C - A, q - A
                    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
                    d20, d21 = v2 @ v0, v2 @ v1
                    den = d00 * d11 - d01 * d01
                    if abs(den) < 1e-30:
                        continue
                    bv = (d11 * d20 - d01 * d21) / den
                    bw = (d00 * d21 - d01 * d20) / den
                    bu = 1.0 - bv - bw
                    neg = min(bu, bv, bw)
                    if best is None or neg > best[0]:
                        best = (neg, bu, bv, bw, i)
                if best is None:
                    continue
                _, bu, bv, bw, i = best
                u0, u1, u2 = ouv[0][k], ouv[i][k], ouv[i + 1][k]
                uv_old.append((bu * u0[0] + bv * u1[0] + bw * u2[0], bu * u0[1] + bv * u1[1] + bw * u2[1]))
                uv_new.append(uu)
        if not uv_new:
            return 0.0
        c1 = texmod.sample(tex, np.array(uv_new))
        c0 = texmod.sample(tex, np.array(uv_old))
        return float(np.abs(c1 - c0).mean())

    def undo(self, u):
        m = self.m
        for fid, (vs, uv) in u['faces'].items():
            m.faces[fid] = vs
            m.fuv[fid] = uv
        for v, s in u['vf'].items():
            m.vf[v] = s
        for arr, name in ((m.pos, 'pos'), (self.Q, 'Q'), (self.w, 'w'), (self.thick, 'thick'),
                          (self.protect, 'prot'), (self.joint, 'joint'), (self.mirror, 'mirror'), (self.vacc, 'vacc'), (self.uacc, 'uacc')):
            for v, val in u[name].items():
                arr[v] = val
        for v, lst in u['members'].items():
            self.members[v] = lst
        for k, val in u['l0'].items():
            if val is None:
                self.l0.pop(k, None)
            else:
                self.l0[k] = val
        for k, val in u.get('et', {}).items():
            if val is None:
                self.etube.pop(k, None)
            else:
                self.etube[k] = val
        for k, val in u.get('ep', {}).items():
            if val is None:
                self.eprofile.pop(k, None)
            else:
                self.eprofile[k] = val
        for k, val in u.get('ej', {}).items():
            if val is None:
                self.ejoint.pop(k, None)
            else:
                self.ejoint[k] = val
        if u['sharp'] is not None:
            m.sharp = u['sharp']
        m.alive_faces = u['alive']
        for fid in u['removed']:
            self.comp_alive[m.fcomp[fid]] += 1
        for v in u['touched']:
            self.vinfo.pop(v, None)

    def _uv_areas(self, uv):
        out = []
        n = len(uv)
        for k in range(self.m.n_layers):
            s = 0.0
            for i in range(n):
                x0, y0 = uv[i][k]
                x1, y1 = uv[(i + 1) % n][k]
                s += x0 * y1 - x1 * y0
            out.append(s)
        return out

    def _quad_ok(self, vs, n):
        """reject strongly concave / folded quads"""
        p = self.m.pos
        neg = 0
        for i in range(4):
            a, b, c = p[vs[i - 1]], p[vs[i]], p[vs[(i + 1) % 4]]
            cr = np.cross(b - a, c - b)
            if float(cr @ n) < -1e-9 * (1 + float(np.linalg.norm(cr))):
                neg += 1
        return neg == 0 or self._quad_was_concave(vs)

    def _quad_was_concave(self, vs):
        return False

    def _fan_components(self, v):
        """number of edge-connected groups of faces around v"""
        m = self.m
        fs = list(m.vf[v])
        if len(fs) <= 1:
            return len(fs)
        par = {f: f for f in fs}

        def find(x):
            while par[x] != x:
                x = par[x]
            return x
        byn = {}
        for f in fs:
            vs = m.faces[f]
            n = len(vs)
            i = vs.index(v)
            for x in (vs[(i + 1) % n], vs[(i - 1) % n]):
                if x in byn:
                    par[find(f)] = find(byn[x])
                else:
                    byn[x] = f
        return len({find(f) for f in fs})

    def invalidate_after(self, touched):
        """drop cached edge costs around changed vertices (2-ring)"""
        ec = self.ecache
        ring1 = set(touched)
        for v in touched:
            if self.m.vf[v]:
                ring1.update(self.info(v).keys())
        for v in ring1:
            if not self.m.vf[v]:
                continue
            for x in self.info(v):
                ec.pop((v, x), None)
                ec.pop((x, v), None)
        return ring1

    # --------------------------------------------------------- operations
    def ring_op(self, ring, ring_of=None, ring_len=None, full=True):
        """Build the collapse list for one ring (plus its mirror image).
        -> (cost, [(a,b,t,p)], n_faces_removed_estimate) or None"""
        mir = self.mirror
        self._acc = {}
        cost = 0.0
        ssum = 0.0
        ax = self.sym_axis
        todo = list(ring)
        ringset = set(ring)
        # mirror ring
        extra = []
        for (a, b) in ring:
            ma, mb = mir[a], mir[b]
            if ma >= 0 and mb >= 0:
                k = ekey(ma, mb)
                if k not in ringset:
                    extra.append(k)
            else:
                extra = None
                break
        sym = extra is not None
        if sym and extra and not full:
            for (a, b) in extra:
                if b not in self.info(a):
                    sym = False
                    break
        elif sym and extra:
            # the mirrored edges must exist and form exactly another ring
            rid = None
            for k in extra:
                r = ring_of.get(k) if ring_of is not None else None
                if r is None or (rid is not None and r != rid):
                    sym = False
                    break
                rid = r
            if sym and (ring_len[rid] != len(extra) or len(extra) != len(ring)):
                sym = False
        used = set()
        out = []
        for (a, b) in ring:
            if a in used or b in used:
                self.fail_stats['shared-vertex'] += 1
                return None
            r = self.eval_edge(a, b)
            if r is None:
                return None
            c, t, p, da, sa = r[:5]
            used.add(a)
            used.add(b)
            out.append((a, b, t, p))
            self._acc[a] = (da, sa, r[5], r[6], r[7])
            if c > cost:
                cost = c
            ssum += c
            if sym and extra:
                ma, mb = mir[a], mir[b]
                if ekey(ma, mb) in ringset:
                    continue
                r2 = self.eval_edge(ma, mb)
                if r2 is None:
                    return None
                if ma in used or mb in used:
                    return None
                used.add(ma)
                used.add(mb)
                q = p.copy()
                q[ax] = -q[ax]
                out.append((ma, mb, t, q))
                self._acc[ma] = r2[3:8]
                if r2[0] > cost:
                    cost = r2[0]
        if sym and not extra and ax is not None:
            # self-mirrored ring: make parameters symmetric
            byedge = {ekey(a, b): i for i, (a, b, t, p) in enumerate(out)}
            for i, (a, b, t, p) in enumerate(out):
                ma, mb = mir[a], mir[b]
                j = byedge.get(ekey(ma, mb))
                if j is None or j <= i:
                    if j == i and ma == a and mb == b:
                        p = p.copy()
                        p[ax] = 0.0 if abs(self.m.pos[a][ax]) < 1e-9 and abs(self.m.pos[b][ax]) < 1e-9 else p[ax]
                        out[i] = (a, b, t, p)
                    continue
                a2, b2, t2, p2 = out[j]
                # orientation of the mirrored edge
                tt = t if a2 == ma else 1.0 - t
                r2 = self.eval_edge(a2, b2)
                # the mirrored edge may have a hard constraint: then use it for both
                if abs(r2[1] - tt) > 1e-6:
                    c2 = self._constraint(a2, b2, self.info(a2), self.info(b2))
                    c1 = self._constraint(a, b, self.info(a), self.info(b))
                    if c2 != 'free' and c1 != 'free':
                        return None
                    if c2 != 'free':
                        tt = c2
                        t = tt if a2 == ma else 1.0 - tt
                        p = self.m.pos[a] + t * (self.m.pos[b] - self.m.pos[a])
                        out[i] = (a, b, t, p)
                q = p.copy()
                q[ax] = -q[ax]
                out[j] = (a2, b2, tt, q)
        return (cost + 1e-3 * ssum / max(1, len(ring)), out, sym, self._acc)

    def chord_sequences(self, ring):
        """order the edges of a ring along the chord -> list of sequences
        (a closed chord gives one sequence that starts anywhere)"""
        m = self.m
        ringset = set(ring)
        link = {k: [] for k in ring}
        for (a, b) in ring:
            for f in m.vf[a] & m.vf[b]:
                vs = m.faces[f]
                if len(vs) != 4:
                    continue
                i = vs.index(a)
                # the opposite edge
                if vs[(i + 1) % 4] == b:
                    o = ekey(vs[(i + 2) % 4], vs[(i + 3) % 4])
                else:
                    o = ekey(vs[(i + 1) % 4], vs[(i + 2) % 4])
                if o in ringset:
                    link[(a, b)].append(o)
        seqs = []
        seen = set()
        starts = [k for k in ring if len(link[k]) < 2] + list(ring)
        for k0 in starts:
            if k0 in seen:
                continue
            seq = [k0]
            seen.add(k0)
            cur = k0
            while True:
                nxt = None
                for o in link[cur]:
                    if o not in seen:
                        nxt = o
                        break
                if nxt is None:
                    break
                seq.append(nxt)
                seen.add(nxt)
                cur = nxt
            seqs.append(seq)
        return seqs

    def op_stats(self, out):
        """-> (faces removed, new triangles, touches protected/joint zone at a cut)"""
        m = self.m
        es = {ekey(a, b) for a, b, t, p in out}
        removed = set()
        newtri = set()
        guarded = False
        for a, b, t, p in out:
            for f in m.vf[a] & m.vf[b]:
                vs = m.faces[f]
                if len(vs) == 3:
                    removed.add(f)
                elif len(vs) == 4:
                    i = vs.index(a)
                    if vs[(i + 1) % 4] == b:
                        o = ekey(vs[(i + 2) % 4], vs[(i + 3) % 4])
                    else:
                        o = ekey(vs[(i + 1) % 4], vs[(i + 2) % 4])
                    if o in es:
                        removed.add(f)
                    else:
                        newtri.add(f)
                        if f in self.band_quads:
                            guarded = True       # never break a closed ring of quads
                        if self.ejoint.get(o, 0.0) > 0 or self.ejoint.get(ekey(a, b), 0.0) > 0 or \
                                max(self.protect[v] for v in vs) > 0:
                            guarded = True
        return len(removed), len(newtri), guarded

    def partial_ops(self, ring, level):
        """Sub-chords of a ring that is blocked or expensive somewhere: the
        loop is removed only along a stretch and ends in a triangle at each cut
        (what an artist does when a loop cannot be removed all the way)."""
        res = []
        mir = self.mirror
        for seq in self.chord_sequences(ring):
            n = len(seq)
            if n < 3:
                continue
            cs = []
            for (a, b) in seq:
                r = self.eval_edge(a, b)
                cs.append(INF if r is None else r[0])
            fin = sorted(c for c in cs if c < INF)
            if len(fin) < 3:
                continue
            ths = sorted({fin[len(fin) // 4], fin[len(fin) // 2], fin[(3 * len(fin)) // 4], fin[-1]})
            done = set()
            for th in ths:
                if fin[0] > level * 1.0 and level < INF and th > fin[0]:
                    pass
                i = 0
                while i < n:
                    if cs[i] > th:
                        i += 1
                        continue
                    j = i
                    while j + 1 < n and cs[j + 1] <= th:
                        j += 1
                    if j - i + 1 >= 3 and (i, j) not in done and not (i == 0 and j == n - 1 and len(fin) == n):
                        done.add((i, j))
                        run = seq[i:j + 1]
                        # symmetric completion when the run touches its own mirror image
                        rs = set(run)
                        ext = []
                        overlap = False
                        okm = True
                        for (a, b) in run:
                            ma, mb = mir[a], mir[b]
                            if ma < 0 or mb < 0:
                                okm = False
                                break
                            k = ekey(ma, mb)
                            if k in rs:
                                overlap = True
                            else:
                                ext.append(k)
                        if okm and overlap and ext:
                            if any(b2 not in self.info(a2) for a2, b2 in ext):
                                i = j + 1
                                continue
                            run = run + ext
                        r = self.ring_op(run, full=False)
                        if r is not None:
                            rem, nt, guarded = self.op_stats(r[1])
                            if not guarded and rem >= 2 and rem >= nt:
                                cost = r[0] * (1.0 + self.p.tri_penalty * nt / rem)
                                res.append((cost, run, r[1], r[2], r[3], nt))
                    i = j + 1
        return res

    def heal_split_quads(self, fids):
        """Dissolve the diagonal of split quads among the given (just modified)
        faces: a pair of triangles that together are a planar convex quad whose
        diagonal is their shared edge.  -> set of vertices whose faces changed"""
        m = self.m
        pos = m.pos
        cos_max = math.cos(math.radians(self.p.heal_angle))
        cands = []
        seen = set()
        for f1 in fids:
            vs = m.faces[f1]
            if vs is None or len(vs) != 3:
                continue
            for i in range(3):
                a, b, c = vs[i], vs[(i + 1) % 3], vs[(i + 2) % 3]
                if self.info(a).get(b, 1) != 0:
                    continue                      # border / seam / hard edge / non-manifold
                fs = m.vf[a] & m.vf[b]
                if len(fs) != 2:
                    continue
                f2 = next(f for f in fs if f != f1)
                v2 = m.faces[f2]
                if len(v2) != 3 or m.fcomp[f2] != m.fcomp[f1] or (min(f1, f2), max(f1, f2)) in seen:
                    continue
                seen.add((min(f1, f2), max(f1, f2)))
                d = next(v for v in v2 if v != a and v != b)
                if d == c or v2[(v2.index(b) + 1) % 3] != a:
                    continue                      # inconsistent winding
                lab = float(np.linalg.norm(pos[a] - pos[b]))
                if any(float(np.linalg.norm(pos[x] - pos[y])) > lab * (1 + 1e-9)
                       for x, y in ((a, c), (b, c), (a, d), (b, d))):
                    continue                      # the shared edge is not the diagonal
                n1 = m.face_normal(vs)
                n2 = m.face_normal(v2)
                l = float(np.linalg.norm(n1)) * float(np.linalg.norm(n2))
                if l < 1e-30:
                    continue
                cs = float(n1 @ n2) / l
                if cs < cos_max:
                    continue
                quad = [c, a, d, b]
                # corners must look like quad corners
                ok = True
                for j in range(4):
                    e1 = pos[quad[j - 1]] - pos[quad[j]]
                    e2 = pos[quad[(j + 1) % 4]] - pos[quad[j]]
                    ca = float(e1 @ e2) / ((float(np.linalg.norm(e1)) * float(np.linalg.norm(e2))) or 1e-30)
                    if abs(ca) > 0.82:            # outside 35..145 degrees
                        ok = False
                        break
                if not ok or not self._quad_ok(quad, n1 + n2):
                    continue
                cands.append((-cs, f1, f2, a, b, c, d))
        cands.sort()
        touched = set()
        used = set()
        for _, f1, f2, a, b, c, d in cands:
            if f1 in used or f2 in used:
                continue
            # a and b lose an edge: they must keep at least 3 (interior) / 2 (border) faces
            bad = False
            for v in (a, b):
                border = any(mk & (BOUNDARY | NONMANIFOLD) for mk in self.info(v).values())
                if len(m.vf[v]) - 1 < (2 if border else 3):
                    bad = True
            if bad:
                continue
            vs = m.faces[f1]
            v2 = m.faces[f2]
            uv1 = m.fuv[f1]
            uv2 = m.fuv[f2]
            m.faces[f1] = [c, a, d, b]
            m.fuv[f1] = [uv1[vs.index(c)], uv1[vs.index(a)], uv2[v2.index(d)], uv1[vs.index(b)]]
            m.faces[f2] = None
            for v in v2:
                m.vf[v].discard(f2)
            m.vf[d].add(f1)
            m.alive_faces -= 1
            self.comp_alive[m.fcomp[f2]] -= 1
            self.n_tris -= 2
            self.n_healed += 1
            used.add(f1)
            used.add(f2)
            for v in (a, b, c, d):
                self.vinfo.pop(v, None)
                touched.add(v)
        return touched

    def _count_tris_around(self, out):
        m = self.m
        fs = set()
        for a, b, t, p in out:
            fs |= m.vf[a]
            fs |= m.vf[b]
        return sum(1 for f in fs if m.faces[f] is not None and len(m.faces[f]) == 3)

    def apply_op(self, out, sym, acc=None):
        """apply, keep mirror map consistent. -> touched set or None"""
        mir = self.mirror
        u = self.collapse_edges(out)
        if u is None:
            return None
        if acc:
            eps = self.p.geo_eps
            exact = 0.0
            self.last_rel = 0.0
            for a, b, t, p in out:
                if a in acc:
                    da, sa, base, tau, uvrel = acc[a]
                    err = self.true_error(a)
                    self.vacc[a], self.uacc[a] = max(err, 0.5 * da), sa
                    trel = self.p.tex_weight * self.tex_error(a) if self.p.texture is not None else 0.0
                    c = base * (1.0 + ((err / tau + uvrel + trel) / eps) ** 2)
                    if err / tau > self.last_rel:
                        self.last_rel = err / tau
                    if c > exact:
                        exact = c
            self.last_exact = exact
        if sym:
            for a, b, t, p in out:
                ma, mb = u['mirror'][a], u['mirror'][b]
                # partner edge was collapsed in the same op: survivors pair up
                if ma == b:
                    mir[a] = a
                else:
                    # survivor of mirrored edge
                    sa = int(ma)
                    if not self.m.vf[sa]:
                        sa = int(mb)
                    mir[a] = sa
        else:
            for a, b, t, p in out:
                for v in (a, b):
                    mv = u['mirror'][v]
                    if mv >= 0:
                        mir[mv] = -1
                mir[a] = -1
        return u

    # ---------------------------------------------------------------- run
    def run(self):
        m = self.m
        p = self.p
        target = p.target_faces if p.target_faces else int(round(self.nfaces0 * p.ratio))
        target = max(4, target)
        start = m.alive_faces
        it = 0
        stuck = False
        q0, t0, _o = m.count()
        self.n_tris = t0
        self.max_tris_frac = p.max_tri_ratio if p.max_tri_ratio is not None else max(t0 / max(1, self.nfaces0), 0.10)
        self.n_requeued = 0
        self.n_healed = 0
        self.last_exact = 0.0
        from . import joints
        rcache = {}          # ring key -> [vertex set, full op, band quads, partial ops]
        tol = max(2, int(0.004 * self.nfaces0))
        tol_stop = max(3, int(0.006 * self.nfaces0))
        finished = False
        rel_cap = p.max_rel_error
        warp_cap = p.max_warp
        self.last_warp = 0.0
        capped = set()
        tri_blocked = set()
        self.n_tri_relax = 0
        self.last_rel = 0.0
        while m.alive_faces > target and not finished:
            it += 1
            rings = self.rings()
            ring_of = {}
            ring_len = []
            for i, ring in enumerate(rings):
                ring_len.append(len(ring))
                for k in ring:
                    ring_of[k] = i
            ents = []
            self.band_quads = set()
            for ring in rings:
                fk = frozenset(ring)
                ent = rcache.get(fk)
                if ent is None:
                    r = self.ring_op(ring, ring_of, ring_len)
                    vs = set()
                    for a, b in ring:
                        vs.add(a)
                        vs.add(b)
                    if r is not None:
                        for a, b, t, pp in r[1]:
                            vs.add(a)
                            vs.add(b)
                    bq = None
                    if len(ring) >= 3:
                        bl = joints.closed_bands(self, [ring])
                        if bl:
                            bq = bl[0]['quads']
                            if r is not None and len(ring) >= 6:
                                # a limb segment must not get much longer than the limb is
                                # thick, or nothing is left to bend (works on the current
                                # mesh, so it also covers rings that only became closed later)
                                dia = bl[0]['perim'] / math.pi
                                lmax = 0.0
                                pos = m.pos
                                for a, b, t, pp in r[1]:
                                    for v, o in ((a, b), (b, a)):
                                        dv = pos[v] - pos[o]
                                        best = None
                                        bc = 0.3
                                        for x in self.info(v):
                                            if x == o:
                                                continue
                                            e = pos[x] - pos[v]
                                            c = float(e @ dv) / ((math.sqrt(float(e @ e)) * math.sqrt(float(dv @ dv))) or 1e-30)
                                            if c > bc:
                                                bc = c
                                                best = x
                                        if best is not None:
                                            lmax = max(lmax, float(np.linalg.norm(pp - pos[best])))
                                fac = 1.0 + (lmax / (p.tube_aspect * dia)) ** 6
                                r = (r[0] * fac,) + tuple(r[1:])
                    ent = [vs, r, bq, None]
                    rcache[fk] = ent
                if ent[2]:
                    self.band_quads |= ent[2]
                ents.append((ring, fk, ent))
            ops = []
            for ring, fk, ent in ents:
                r = ent[1]
                if r is not None and fk not in self.blocked:
                    ops.append((max(r[0], self.cost_override.get(fk, 0.0)), len(ops), ring, r[1], r[2], fk, r[3]))
                if p.partial and len(ring) >= 3:
                    if ent[3] is None:
                        ent[3] = self.partial_ops(ring, INF)
                        for po in ent[3]:
                            for a, b, t, pp in po[2]:
                                ent[0].add(a)
                                ent[0].add(b)
                    for cost, run, out, sym, acc, nt in ent[3]:
                        fk2 = frozenset(run)
                        if fk2 in self.blocked:
                            continue
                        ops.append((max(cost, self.cost_override.get(fk2, 0.0)), len(ops), run, out, sym, fk2, acc))
            ops = [o for o in ops if o[5] not in capped]
            if not ops:
                if capped:
                    rel_cap *= 1.4
                    warp_cap *= 1.25
                    capped.clear()
                    continue
                if p.stuck_relax_tris and tri_blocked and self.max_tris_frac < 1.0:
                    self.max_tris_frac = min(1.0, self.max_tris_frac * 1.5 + 0.05)
                    self.blocked -= tri_blocked
                    tri_blocked.clear()
                    self.n_tri_relax += 1
                    continue
                stuck = True
                break
            ops.sort(key=lambda o: o[0])
            best = None
            dirty = set()
            applied = 0
            for cost, _, ring, out, sym, fk, acc in ops:
                if best is not None and cost > best * 1.03 + 1e-9:
                    break
                vs = set()
                for a, b, t, pp in out:
                    vs.add(a)
                    vs.add(b)
                if vs & dirty:
                    continue
                need = m.alive_faces - target
                before = m.alive_faces
                if need <= tol_stop and len(self.log) >= 5:
                    ref = sorted(l[0] for l in self.log[-9:])[len(self.log[-9:]) // 2]
                    if cost > 1.5 * ref:
                        finished = True      # close enough: do not force a bad last step
                        break
                rem, nt, guarded = self.op_stats(out)
                if nt and (guarded or (self.n_tris + nt) > self.max_tris_frac * (m.alive_faces - rem)):
                    self.blocked.add(fk)      # triangle budget / would break a ring
                    if not guarded:
                        tri_blocked.add(fk)
                    continue
                tri_before = self._count_tris_around(out)
                u = self.apply_op(out, sym, acc)
                if u is None:
                    self.blocked.add(fk)
                    self.fail_stats[self.fail_reason] += 1
                    continue
                if self.last_rel > rel_cap or self.last_warp > warp_cap:
                    # too much shape error for now; only allowed if nothing else is left
                    self.undo(u)
                    capped.add(fk)
                    continue
                if self.last_exact > 1.25 * cost:
                    # the estimate was too optimistic: put it back with the measured cost
                    self.undo(u)
                    self.cost_override[fk] = self.last_exact
                    self.n_requeued += 1
                    continue
                removed = before - m.alive_faces
                if removed > need + tol and removed > 2:
                    # overshoots the target too much: undo, try a smaller one
                    self.undo(u)
                    self.blocked.add(fk)
                    continue
                self.n_tris += self._count_tris_around(out) - tri_before
                dirty |= self.invalidate_after(u['touched'])
                if p.heal_quads:
                    ht = self.heal_split_quads(list(u['faces']))
                    if ht:
                        dirty |= self.invalidate_after(ht)
                applied += 1
                if best is None:
                    best = cost
                self.log.append((cost, len(ring), removed, sym))
                if m.alive_faces <= target:
                    break
            if dirty:
                for fk in [k for k, e in rcache.items() if not e[0].isdisjoint(dirty)]:
                    del rcache[fk]
            if p.progress and it % 3 == 0:
                p.progress(min(1.0, (start - m.alive_faces) / max(1, start - target)), 'collapsing loops')
        self.stuck = stuck
        self.rel_cap_final = rel_cap
        return m.alive_faces

    # ------------------------------------------------- single edge collapses
    def run_edges(self):
        """Plain (triangle friendly) edge-collapse decimation with the same
        seam / border / flip safeguards.  Used for static props when Blender's
        own decimator damages UVs."""
        import heapq
        m = self.m
        p = self.p
        target = p.target_faces if p.target_faces else int(round(self.nfaces0 * p.ratio))
        target = max(4, target)
        start = m.alive_faces
        self.relax = p.relax_seams is True
        # per material budgets
        gt = p.group_targets
        fmat = getattr(m, 'fmat', None)
        galive = {}
        if gt and fmat is not None:
            for fid, vs in enumerate(m.faces):
                if vs is not None:
                    galive[fmat[fid]] = galive.get(fmat[fid], 0) + 1
            gt = {g: max(1, int(gt.get(g, galive[g]))) for g in galive}
            target = sum(min(gt[g], galive[g]) for g in galive)
        else:
            gt = None

        def fill():
            hp = []
            c = 0
            for (a, b) in list(self.l0.keys()):
                if not m.vf[a] or not m.vf[b]:
                    continue
                r = self.eval_edge(a, b)
                if r is not None:
                    hp.append((r[0], c, a, b, r))
                    c += 1
            heapq.heapify(hp)
            return hp, c
        heap, cnt = fill()
        last_refill = m.alive_faces
        n = 0
        ecache = self.ecache
        while m.alive_faces > target:
            if not heap:
                # collapses that failed earlier may have become possible; then relax
                if m.alive_faces < last_refill:
                    last_refill = m.alive_faces
                    ecache.clear()
                    heap, cnt = fill()
                    continue
                if p.relax_seams == 'STUCK' and not self.relax:
                    self.relax = True
                    ecache.clear()
                    heap, cnt = fill()
                    continue
                break
            cost, _, a, b, r = heapq.heappop(heap)
            if ecache.get((a, b)) is not r:
                continue
            if gt is not None:
                g = fmat[next(iter(m.vf[a]))] if m.vf[a] else None
                if g is None or galive[g] <= gt[g]:
                    continue
            ia = self.info(a)
            ib = self.info(b)
            apex = [c for c in ia if c in ib]
            u = self.collapse_edges([(a, b, r[1], r[2])])
            if u is None:
                ecache[(a, b)] = None
                continue
            if gt is not None:
                for fid in u['removed']:
                    galive[fmat[fid]] -= 1
            self.vacc[a], self.uacc[a] = r[3], r[4]
            for v in (a, b):
                mv = u['mirror'][v]
                if mv >= 0:
                    self.mirror[mv] = -1
            self.mirror[a] = -1
            # only edges at the merged vertex and at the triangle apexes change
            for v in [a] + apex:
                if not m.vf[v]:
                    continue
                for x in self.info(v):
                    k = (v, x) if v < x else (x, v)
                    ecache.pop(k, None)
                    r2 = self.eval_edge(k[0], k[1])
                    if r2 is not None:
                        heapq.heappush(heap, (r2[0], cnt, k[0], k[1], r2))
                        cnt += 1
            n += 1
            if p.progress and n % 200 == 0:
                p.progress(min(1.0, (start - m.alive_faces) / max(1, start - target)), 'collapsing edges')
        self.stuck = m.alive_faces > target
        return m.alive_faces

    def self_check(self):
        """exact checks against each face's own source polygon:
        -> (faces whose normal turned by more than 90 degrees, faces with flipped UVs)"""
        m = self.m
        nflip = uvflip = 0
        for fid, vs in enumerate(m.faces):
            if vs is None:
                continue
            n0 = m.face_normal(self.orig_faces[fid], self.orig_pos)
            n1 = m.face_normal(vs)
            if float(n0 @ n1) < 0:
                nflip += 1
            a0 = self._uv_areas(self.orig_fuv[fid])
            a1 = self._uv_areas(m.fuv[fid])
            for k in self.seam_layers:
                if abs(a0[k]) > 1e-12 and a0[k] * a1[k] <= 0:
                    uvflip += 1
                    break
        return nflip, uvflip
