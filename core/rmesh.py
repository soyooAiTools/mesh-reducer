"""Light-weight editable polygon mesh used by the reducers.

Pure python + numpy; filled from / written back to a Blender mesh datablock.
Faces keep per-corner UVs (all layers), so UV seams are implicit: an edge is a
seam in layer k when the two faces around it disagree on the UV of one of its
end points.
"""
import numpy as np

BOUNDARY = 1
NONMANIFOLD = 2
SHARP = 4
SEAM0 = 8          # bit for uv layer 0, layer k -> SEAM0 << k


def ekey(a, b):
    return (a, b) if a < b else (b, a)


class RMesh:
    def __init__(self):
        self.pos = None            # (V,3) float64
        self.faces = []            # fid -> [vid,...] or None
        self.fuv = []              # fid -> [corner -> ((u,v),(u,v),...)]
        self.fsrc = []             # fid -> source polygon index
        self.fcomp = []            # fid -> connected component id
        self.vf = []               # vid -> set(fid)
        self.n_layers = 0
        self.uv_names = []
        self.sharp = set()         # set of edge keys with split normals
        self.alive_faces = 0
        self.flat_shaded = False

    # ------------------------------------------------------------------ io
    @classmethod
    def from_blender(cls, me, use_sharp=True):
        m = cls()
        nv = len(me.vertices)
        pos = np.empty(nv * 3, dtype=np.float32)
        me.vertices.foreach_get('co', pos)
        m.pos = pos.reshape(nv, 3).astype(np.float64)
        nl = len(me.loops)
        lv = np.empty(nl, dtype=np.int32)
        me.loops.foreach_get('vertex_index', lv)
        uvs = []
        for layer in me.uv_layers:
            a = np.empty(nl * 2, dtype=np.float32)
            layer.uv.foreach_get('vector', a)
            uvs.append(a.reshape(nl, 2).astype(np.float64))
            m.uv_names.append(layer.name)
        m.n_layers = len(uvs)
        cn = np.empty(nl * 3, dtype=np.float32)
        me.corner_normals.foreach_get('vector', cn)
        cn = cn.reshape(nl, 3)
        m.vf = [set() for _ in range(nv)]
        face_loops = []
        for p in me.polygons:
            ls = p.loop_start
            n = p.loop_total
            vs = [int(lv[ls + i]) for i in range(n)]
            fid = len(m.faces)
            m.faces.append(vs)
            m.fuv.append([tuple((float(u[ls + i, 0]), float(u[ls + i, 1])) for u in uvs) for i in range(n)])
            m.fsrc.append(p.index)
            face_loops.append((ls, n))
            for v in vs:
                m.vf[v].add(fid)
        m.alive_faces = len(m.faces)
        # shading style / sharp edges from split normals
        nsharp = 0
        nint = 0
        edge_seen = {}
        for fid, vs in enumerate(m.faces):
            ls, n = face_loops[fid]
            for i in range(n):
                a, b = vs[i], vs[(i + 1) % n]
                k = ekey(a, b)
                na, nb = cn[ls + i], cn[ls + (i + 1) % n]
                if k in edge_seen:
                    pa, pb = edge_seen[k]
                    if a > b:
                        na, nb = nb, na
                    nint += 1
                    if float(np.dot(na, pa)) < 0.9995 or float(np.dot(nb, pb)) < 0.9995:
                        nsharp += 1
                        m.sharp.add(k)
                else:
                    edge_seen[k] = (na, nb) if a < b else (nb, na)
        m.flat_shaded = nint > 0 and nsharp > 0.9 * nint
        if m.flat_shaded or not use_sharp:
            m.sharp = set()
        m._components()
        return m

    def _components(self):
        nv = len(self.pos)
        par = list(range(nv))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for vs in self.faces:
            if vs is None:
                continue
            r = find(vs[0])
            for v in vs[1:]:
                par[find(v)] = r
        ids = {}
        self.fcomp = []
        for vs in self.faces:
            if vs is None:
                self.fcomp.append(-1)
                continue
            r = find(vs[0])
            self.fcomp.append(ids.setdefault(r, len(ids)))
        self.n_comp = len(ids)

    def compact(self):
        """-> verts (n,3), faces [[...]], uvs per layer per face-corner, fsrc, old->new vertex map"""
        used = {}
        faces = []
        fuv = []
        fsrc = []
        fcomp = []
        for fid, vs in enumerate(self.faces):
            if vs is None:
                continue
            faces.append([used.setdefault(v, len(used)) for v in vs])
            fuv.append(self.fuv[fid])
            fsrc.append(self.fsrc[fid])
            fcomp.append(self.fcomp[fid])
        verts = np.zeros((len(used), 3))
        for old, new in used.items():
            verts[new] = self.pos[old]
        return verts, faces, fuv, fsrc, fcomp, used

    def to_blender(self, name, src_me, bpy):
        verts, faces, fuv, fsrc, fcomp, used = self.compact()
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts.tolist(), [], faces)
        for k, lname in enumerate(self.uv_names):
            layer = me.uv_layers.new(name=lname)
            flat = np.array([c[k] for f in fuv for c in f], dtype=np.float32).ravel()
            layer.uv.foreach_set('vector', flat)
        for i, layer in enumerate(src_me.uv_layers):
            if layer.active_render and i < len(me.uv_layers):
                me.uv_layers[i].active_render = True
            if layer.active and i < len(me.uv_layers):
                me.uv_layers.active_index = i
        for mat in src_me.materials:
            me.materials.append(mat)
        mi = np.empty(len(src_me.polygons), dtype=np.int32)
        src_me.polygons.foreach_get('material_index', mi)
        sm = np.empty(len(src_me.polygons), dtype=bool)
        src_me.polygons.foreach_get('use_smooth', sm)
        fs = np.array(fsrc, dtype=np.int64)
        me.polygons.foreach_set('material_index', mi[fs].astype(np.int32))
        me.polygons.foreach_set('use_smooth', sm[fs])
        me.update()
        me.validate()
        return me, used

    # ------------------------------------------------------------ helpers
    def face_normal(self, vs, pos=None):
        p = self.pos if pos is None else pos
        n = len(vs)
        if n == 3:
            a, b, c = p[vs[0]], p[vs[1]], p[vs[2]]
            return np.cross(b - a, c - a)
        if n == 4:
            return np.cross(p[vs[2]] - p[vs[0]], p[vs[3]] - p[vs[1]])
        nrm = np.zeros(3)
        for i in range(n):
            a, b = p[vs[i]], p[vs[(i + 1) % n]]
            nrm += np.cross(a, b)
        return nrm

    def count(self):
        q = t = o = 0
        for vs in self.faces:
            if vs is None:
                continue
            if len(vs) == 4:
                q += 1
            elif len(vs) == 3:
                t += 1
            else:
                o += 1
        return q, t, o
