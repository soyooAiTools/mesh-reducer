"""ASCII FBX -> binary FBX (Blender's importer only reads binary files).

The node tree of the text file is parsed and written again with
io_scene_fbx.encode_bin.  The text format does not carry property types, the
binary one does and the importer asserts some of them, so the types are chosen
by the rules the FBX SDK itself follows:
  object ids / connection ends / pose "Node"    INT64
  "Class::Name"                                 b"Name\\x00\\x01Class"
  P: name, type, ... of an integer type          INT32 (KTime / ULongLong: INT64)
  arrays (*N { a: ... })                        by element name (indices INT32,
                                                key times INT64, the rest FLOAT64)
  other numbers                                 INT32 / INT64 if written without
                                                a fraction, else FLOAT64
Embedded media ("Content") is dropped.
"""
import re
import numpy as np

_TOK = re.compile(rb'''\s*(?:
    (?P<comment>;[^\n]*) |
    (?P<str>"[^"]*") |
    (?P<open>\{) | (?P<close>\}) | (?P<comma>,) |
    (?P<key>[A-Za-z_][A-Za-z0-9_\-|]*)\s*: |
    (?P<arr>\*\d+) |
    (?P<num>[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|1\.\#INF|1\.\#IND|1\.\#QNAN|nan|inf)) |
    (?P<word>[A-Za-z_][A-Za-z0-9_]*)
)''', re.X)

INT_ARRAYS = {b'PolygonVertexIndex', b'Edges', b'Materials', b'Indexes', b'Smoothing', b'KeyAttrFlags', b'KeyAttrRefCount',
              b'TextureId', b'PolygonGroup', b'Visibility'}
INT64_ARRAYS = {b'KeyTime'}
FLOAT32_ARRAYS = {b'KeyValueFloat', b'KeyAttrDataFloat'}
INT_PTYPES = {b'int', b'Integer', b'enum', b'bool', b'Bool', b'Enum', b'Short', b'Byte'}
INT64_PTYPES = {b'KTime', b'ULongLong', b'LongLong'}
INT64_ELEMS = {b'Node'}


def is_ascii(path):
    with open(path, 'rb') as f:
        return not f.read(20).startswith(b'Kaydara FBX Binary')


class _Node:
    __slots__ = ('id', 'props', 'elems')

    def __init__(self, id_):
        self.id, self.props, self.elems = id_, [], []


def _parse(data):
    """-> root _Node; props are ('s', bytes) | ('n', bytes) | ('w', bytes) | ('a', ndarray-text)"""
    root = _Node(b'')
    stack = [root]
    cur = None                      # node whose properties are being read
    pos = 0
    n = len(data)
    match = _TOK.match
    while pos < n:
        m = match(data, pos)
        if m is None:
            if not data[pos:].strip():
                break
            raise ValueError('ASCII FBX: cannot parse at byte %d: %r' % (pos, data[pos:pos + 40]))
        pos = m.end()
        k = m.lastgroup
        if k == 'comment' or k == 'comma':
            continue
        if k == 'key':
            cur = _Node(m.group('key'))
            stack[-1].elems.append(cur)
        elif k == 'str':
            cur.props.append(('s', m.group('str')[1:-1]))
        elif k == 'num':
            cur.props.append(('n', m.group('num')))
        elif k == 'word':
            cur.props.append(('w', m.group('word')))
        elif k == 'arr':
            # *N { a: v,v,v }
            o = data.index(b'{', pos)
            a = data.index(b'a:', o)
            c = data.index(b'}', a)
            cur.props.append(('a', data[a + 2:c]))
            pos = c + 1
        elif k == 'open':
            stack.append(cur)
        elif k == 'close':
            stack.pop()
            cur = None
    return root


def _num(tok):
    if tok.isdigit() or (tok[:1] in b'+-' and tok[1:].isdigit()):
        return int(tok)
    try:
        return float(tok)
    except ValueError:
        return 0.0


def _array(text, id_):
    txt = text.decode('ascii', 'replace').replace('\n', ' ').replace('\r', ' ')
    if id_ in INT_ARRAYS or id_.endswith(b'Index'):
        return np.array(np.fromstring(txt, dtype=np.float64, sep=','), dtype=np.int32), 'i'
    if id_ in INT64_ARRAYS:
        return np.array([int(x) for x in txt.split(',') if x.strip()], dtype=np.int64), 'l'
    if id_ in FLOAT32_ARRAYS:
        return np.fromstring(txt, dtype=np.float64, sep=',').astype(np.float32), 'f'
    return np.fromstring(txt, dtype=np.float64, sep=','), 'd'


def _name(s):
    """"Class::Name" -> Name \\x00\\x01 Class"""
    i = s.find(b'::')
    return s if i < 0 else s[i + 2:] + b'\x00\x01' + s[:i]


def _emit(node, parent, ctx, FBXElem):
    if node.id == b'Content':        # embedded media (base64 text)
        return
    e = FBXElem(node.id)
    parent.elems.append(e)
    props = node.props
    ptype = props[1][1] if node.id == b'P' and len(props) > 1 and props[1][0] == 's' else None
    for i, (k, v) in enumerate(props):
        if k == 's':
            if (ctx == b'Objects' and i == 1) or (node.id in (b'SceneInfo', b'Take') and i == 0 and b'::' in v):
                v = _name(v)
            e.add_string(v.replace(b'&quot;', b'"'))
        elif k == 'w':
            e.add_string(v)
        elif k == 'a':
            arr, t = _array(v, node.id)
            {'i': e.add_int32_array, 'l': e.add_int64_array, 'f': e.add_float32_array, 'd': e.add_float64_array}[t](arr)
        else:
            x = _num(v)
            if (ctx == b'Objects' and i == 0) or (ctx == b'Connections' and i in (1, 2)) or node.id in INT64_ELEMS:
                e.add_int64(int(x))
            elif ptype is not None and i >= 4:
                if ptype in INT_PTYPES:
                    e.add_int32(int(x))
                elif ptype in INT64_PTYPES:
                    e.add_int64(int(x))
                else:
                    e.add_float64(float(x))
            elif isinstance(x, int):
                if -2 ** 31 <= x < 2 ** 31:
                    e.add_int32(x)
                else:
                    e.add_int64(x)
            else:
                e.add_float64(x)
    for c in node.elems:
        _emit(c, e, None, FBXElem)


def to_binary(src, dst, version=7400):
    """write the ASCII FBX file `src` as binary FBX `dst`"""
    import addon_utils
    addon_utils.enable('io_scene_fbx', default_set=False)
    from io_scene_fbx import encode_bin
    with open(src, 'rb') as f:
        data = f.read()
    root = _parse(data)
    out = encode_bin.FBXElem(b'')
    for top in root.elems:
        e = encode_bin.FBXElem(top.id)
        out.elems.append(e)
        for k, v in top.props:
            e.add_string(v) if k in 'sw' else None
        for c in top.elems:
            _emit(c, e, top.id, encode_bin.FBXElem)
    encode_bin.write(dst, out, version)
    return dst
