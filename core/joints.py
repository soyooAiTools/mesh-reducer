"""Tube / joint analysis without an armature.

A *band* is a closed edge ring: a strip of quads that goes all the way round
a limb, finger, torso ...  Stacks of bands are tubes.  From them we get
  * a reliable local diameter (perimeter / pi) for every vertex on a tube
  * joint candidates: bands that are clearly narrower than the typical band of
    their tube are the places where the artist put extra loops for bending
    (knee, elbow, wrist).  Very narrow bands (bevels, hems) are not joints.
"""
import math
import numpy as np
from .rmesh import ekey


def closed_bands(red, rings=None):
    """-> list of dict(edges, quads, width, perim, sides=(frozenset, frozenset))"""
    m = red.m
    out = []
    for ring in (rings if rings is not None else red.rings()):
        if len(ring) < 3:
            continue
        quads = set()
        ok = True
        for a, b in ring:
            fs = m.vf[a] & m.vf[b]
            if len(fs) != 2:
                ok = False
                break
            for f in fs:
                if len(m.faces[f]) != 4:
                    ok = False
                    break
            if not ok:
                break
            quads |= fs
        if not ok or len(quads) != len(ring):
            continue
        ringset = set(ring)
        # the two side loops: connect vertices through the non-ring edges
        par = {}

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        perim = 0.0
        bad = False
        for f in quads:
            vs = m.faces[f]
            n_ring = 0
            for i in range(4):
                a, b = vs[i], vs[(i + 1) % 4]
                par.setdefault(a, a)
                par.setdefault(b, b)
                if ekey(a, b) in ringset:
                    n_ring += 1
                else:
                    par[find(a)] = find(b)
                    perim += 0.5 * float(np.linalg.norm(m.pos[a] - m.pos[b]))
            if n_ring != 2:
                bad = True
        if bad:
            continue
        groups = {}
        for v in par:
            groups.setdefault(find(v), set()).add(v)
        if len(groups) != 2:
            continue
        s1, s2 = (frozenset(g) for g in groups.values())
        width = float(np.mean([np.linalg.norm(m.pos[a] - m.pos[b]) for a, b in ring]))
        out.append({'edges': list(ring), 'quads': quads, 'width': width, 'perim': perim, 'sides': (s1, s2)})
    return out


def analyse(red):
    """-> (band diameter per vertex (nan if not on a tube), {edge key: joint score 0..1})"""
    m = red.m
    nv = len(m.pos)
    diam = np.full(nv, np.nan)
    bands = closed_bands(red)
    for b in bands:
        d = b['perim'] / math.pi
        for s in b['sides']:
            for v in s:
                if np.isnan(diam[v]) or d < diam[v]:
                    diam[v] = d
    # chains of bands sharing a side loop
    par = list(range(len(bands)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    by_vert = {}
    for i, b in enumerate(bands):
        for si, sd in enumerate(b['sides']):
            for v in sd:
                by_vert.setdefault(v, set()).add((i, si))
    for i, b in enumerate(bands):
        for si, sd in enumerate(b['sides']):
            cnt = {}
            for v in sd:
                for (j, sj) in by_vert[v]:
                    if j != i:
                        cnt[(j, sj)] = cnt.get((j, sj), 0) + 1
            for (j, sj), n in cnt.items():
                if 2 * n >= min(len(sd), len(bands[j]['sides'][sj])):
                    par[find(i)] = find(j)
    chains = {}
    for i in range(len(bands)):
        chains.setdefault(find(i), []).append(i)
    score = {}
    red.tubes = []
    for ch in chains.values():
        if len(ch) < 4:
            continue
        ws = np.array([bands[i]['width'] for i in ch])
        ref = float(np.median(ws))
        red.tubes.append([bands[i] for i in ch])
        for i in ch:
            r = bands[i]['width'] / ref
            if r <= 0.3 or r >= 0.95:
                s = 0.0
            elif r < 0.45:
                s = (r - 0.3) / 0.15
            elif r <= 0.8:
                s = 1.0
            else:
                s = (0.95 - r) / 0.15
            bands[i]['joint'] = s
            if s > 0:
                for k in bands[i]['edges']:
                    score[k] = max(score.get(k, 0.0), s)
    red.bands0 = bands
    # every edge that crosses a band of a real tube remembers the tube diameter
    etube = {}
    for ch in chains.values():
        if len(ch) < 3:
            continue
        for i in ch:
            d = bands[i]['perim'] / math.pi
            for k in bands[i]['edges']:
                etube[k] = min(etube.get(k, d), d)
    red.etube = etube
    return diam, score
