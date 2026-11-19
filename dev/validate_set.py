"""dev/validate_set.py SRC_ROOT OUT_ROOT [--tag NAME] [--jobs 3] [--only 01,05] [--ratios 0.5,0.25]
                         [--big 150000:20000,5000] [--retopo 0.25] [--retopo-only 01,11] [--notes notes.json] [--collect]

Batch validation over a set of models.  SRC_ROOT/<id>/model.<ext> (+ optional
SOURCE.txt with the original name, + textures next to it).  Every model is
reduced with dev/validate_one.py at each ratio of its triangle count (models
above --big triangles also to those absolute triangle counts), in parallel
processes.  Written to OUT_ROOT/<tag>/:

    <id>_<target>/result.fbx, result.json, wire.png, tex.png, log.txt
    summary.md / summary.json     one row per run: metrics + verdict
    contact_wire.png, contact_tex.png   all previews (left result, right source)
OUT_ROOT/history.tsv gets one line per run and tag, so iterations can be compared
(summary.md shows the change against the previous tag).

Verdict (automatic, from the numbers only; --notes adds what a human saw):
    bad         run failed | more than 0.5% flipped faces | new non-manifold edges
                | Hausdorff > 3% | silhouette difference > 12% (worst of 4 views)
                | missed the triangle target by > 25%
                | loop modes: more than 10% of the UV islands lost
                | prop mode: texture render difference > 20 (of 255)
                | skinned vertices lost
    acceptable  Hausdorff > 1.5% | mean distance > 0.15% | silhouette difference > 6%
                | texture render difference > 12 (not for retopo: new textures)
                | target missed by > 10% | slivers +5 points over the source
                | quad share -15 points | prop mode: more than 10% of the UV islands lost
    good        everything else
(silhouette / shade / texture differences: see image_proxies in validate_one.py)
--notes: JSON {"<id>_<target>": {"verdict": "...", "problem": "..."}} overrides / explains.
--collect: only rebuild the summary from the result.json files already there."""
import sys, os, json, glob, subprocess, time, concurrent.futures as cf

HERE = os.path.dirname(os.path.abspath(__file__))


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def verdict(r, retopo=False):
    if not r.get('ok'):
        return 'bad', 'run failed: ' + str(r.get('error', '?'))[:120]
    m = r['metrics']
    bad, acc = [], []
    miss = m['tri_ratio'] / max(r['ratio'], 1e-9) - 1.0
    if m['flipped'] > 0.005 * m['faces']:
        bad.append('%d flipped faces' % m['flipped'])
    if m['nonmanifold_edges'] > m['nonmanifold_edges_source'] + max(2, 0.002 * m['faces']):
        bad.append('non-manifold edges %d (source %d)' % (m['nonmanifold_edges'], m['nonmanifold_edges_source']))
    if m['hausdorff_pct'] > 3.0:
        bad.append('Hausdorff %.2f%%' % m['hausdorff_pct'])
    elif m['hausdorff_pct'] > 1.5:
        acc.append('Hausdorff %.2f%%' % m['hausdorff_pct'])
    if m['mean_pct'] > 0.15:
        acc.append('mean distance %.3f%%' % m['mean_pct'])
    if miss > 0.25:
        bad.append('stuck at %.0f%% of the triangles (asked %.0f%%)' % (100 * m['tri_ratio'], 100 * r['ratio']))
    elif abs(miss) > 0.10:
        acc.append('target missed: %.0f%% instead of %.0f%%' % (100 * m['tri_ratio'], 100 * r['ratio']))
    sil, texd = m.get('silhouette_diff_pct'), m.get('tex_diff')
    if sil is not None and sil > 12:
        bad.append('silhouette differs by %.0f%%' % sil)
    elif sil is not None and sil > 6:
        acc.append('silhouette differs by %.0f%%' % sil)
    if not retopo:
        is_prop = max(r.get('modes', {'x': 0}), key=lambda k: r['modes'][k]) == 'prop' if r.get('modes') else False
        try:
            a, b = [int(x) for x in m['uv_islands'].split('/')]
            if a < 0.9 * b:
                (acc if is_prop else bad).append('UV islands %s' % m['uv_islands'])
        except ValueError:
            pass
        if texd is not None and texd > 20 and is_prop:
            bad.append('texture render differs by %.0f/255' % texd)
        elif texd is not None and texd > 12:
            acc.append('texture render differs by %.0f/255' % texd)
    if m.get('unskinned_new_vertices', 0) > 0 or (retopo and m['skinned_vertex_pct'] < m['source_skinned_vertex_pct'] - 0.5):
        bad.append('%d vertices lost their skin weights (skinned %.1f%%, source %.1f%%)' % (
            m.get('unskinned_new_vertices', 0), m['skinned_vertex_pct'], m['source_skinned_vertex_pct']))
    if m['sliver_pct'] > m['sliver_pct_source'] + 5:
        acc.append('slivers %.1f%% (source %.1f%%)' % (m['sliver_pct'], m['sliver_pct_source']))
    if m['quad_pct'] < m['source_quad_pct'] - 15:
        acc.append('quads %.0f%% (source %.0f%%)' % (m['quad_pct'], m['source_quad_pct']))
    if bad:
        return 'bad', '; '.join(bad + acc)
    if acc:
        return 'acceptable', '; '.join(acc)
    return 'good', ''


def run_one(py, src, out, extra, timeout):
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    with open(os.path.join(out, 'log.txt'), 'w') as lf:
        try:
            subprocess.run([py, os.path.join(HERE, 'validate_one.py'), src, out] + extra, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            json.dump({'ok': False, 'error': 'timeout after %d s' % timeout}, open(os.path.join(out, 'result.json'), 'w'))
    if not os.path.isfile(os.path.join(out, 'result.json')):
        json.dump({'ok': False, 'error': 'crashed, see log.txt'}, open(os.path.join(out, 'result.json'), 'w'))
    return out, time.time() - t0


# ---------------------------------------------------------------- contact sheet
FONT = {'0': '111101101101111', '1': '010110010010111', '2': '111001111100111', '3': '111001111001111', '4': '101101111001001',
        '5': '111100111001111', '6': '111100111101111', '7': '111001001001001', '8': '111101111101111', '9': '111101111001111',
        'r': '000000110100100', 't': '010111010010011', 'k': '100101110101101', 'G': '111100101101111', 'A': '010101111101101',
        'B': '110101110101110', 'R': '110101110101101', '_': '000000000000111', ' ': '000000000000000', '-': '000000111000000',
        '%': '101001010100101', 'x': '000101010101000'}


def _text(np, s, scale=4, color=(1, 1, 0.2)):
    h, w = 5 * scale, (4 * len(s)) * scale
    a = np.zeros((h + 2 * scale, w + scale, 4), np.float32)
    a[..., 3] = 1
    a[..., :3] = 0.08
    for i, ch in enumerate(s):
        g = FONT.get(ch, FONT['-'])
        for y in range(5):
            for x in range(3):
                if g[y * 3 + x] == '1':
                    a[scale + y * scale:scale + (y + 1) * scale, scale + (i * 4 + x) * scale:scale + (i * 4 + x + 1) * scale, :3] = color
    return a


def contact_sheet(rows, tagdir, name, out_png, thumb_h=420, cols=4):
    """rows: [(label, run dir)] -> grid of the preview images `name` with a label strip"""
    import bpy
    import numpy as np
    sys.path.insert(0, os.path.dirname(HERE))
    from core import preview
    cells = []
    for label, d, col in rows:
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            img = np.zeros((thumb_h, thumb_h, 4), np.float32)
            img[..., 0] = 0.5
            img[..., 3] = 1
        else:
            a = preview._read_png(p)
            f = max(1, int(np.ceil(a.shape[0] / thumb_h)))
            hh, ww = a.shape[0] // f * f, a.shape[1] // f * f
            img = a[:hh, :ww].reshape(hh // f, f, ww // f, f, 4).mean((1, 3))
        t = _text(np, label, color=col)
        w = max(img.shape[1], t.shape[1])
        cell = np.zeros((img.shape[0] + t.shape[0], w, 4), np.float32)
        cell[..., :3] = 0.15
        cell[..., 3] = 1
        cell[:t.shape[0], :t.shape[1]] = t
        cell[t.shape[0]:, :img.shape[1]] = img
        cells.append(cell)
    lines = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        h = max(c.shape[0] for c in row)
        padded = []
        for c in row:
            q = np.zeros((h, c.shape[1] + 8, 4), np.float32)
            q[..., :3] = 0.15
            q[..., 3] = 1
            q[:c.shape[0], :c.shape[1]] = c
            padded.append(q)
        lines.append(np.concatenate(padded, axis=1))
    W = max(l.shape[1] for l in lines)
    out = []
    for l in lines:
        q = np.zeros((l.shape[0] + 8, W, 4), np.float32)
        q[..., :3] = 0.15
        q[..., 3] = 1
        q[:l.shape[0], :l.shape[1]] = l
        out.append(q)
    preview.save_png(np.concatenate(out, axis=0), out_png)


def main():
    flags = {'--tag', '--jobs', '--only', '--ratios', '--big', '--retopo', '--retopo-only', '--notes', '--timeout', '--python'}
    pos = [a for i, a in enumerate(sys.argv[1:]) if not a.startswith('--') and sys.argv[i] not in flags]
    src_root, out_root = pos[0], pos[1]
    tag = arg('--tag', time.strftime('%Y%m%d_%H%M'))
    jobs = int(arg('--jobs', '3'))
    only = arg('--only')
    ratios = [float(x) for x in arg('--ratios', '0.5,0.25').split(',') if x]
    big = arg('--big', '150000:20000,5000')
    big_min, big_t = int(big.split(':')[0]), [int(x) for x in big.split(':')[1].split(',') if x]
    retopo = [float(x) for x in arg('--retopo', '').split(',') if x]
    retopo_only = arg('--retopo-only')
    notes = json.load(open(arg('--notes'), encoding='utf-8')) if arg('--notes') else {}
    timeout = int(arg('--timeout', '2400'))
    py = arg('--python', sys.executable)
    tagdir = os.path.join(out_root, tag)
    os.makedirs(tagdir, exist_ok=True)
    models = []
    for d in sorted(glob.glob(os.path.join(src_root, '*', ''))):
        fs = [f for f in glob.glob(os.path.join(d, 'model.*')) if f.lower().endswith(('.fbx', '.obj', '.glb', '.gltf'))]
        if not fs:
            continue
        mid = os.path.basename(os.path.dirname(d))
        if only and mid not in only.split(','):
            continue
        name = mid
        st = os.path.join(d, 'SOURCE.txt')
        if os.path.isfile(st):
            name = os.path.splitext(os.path.basename(open(st, encoding='utf-8').read().strip()))[0]
        models.append((mid, name, fs[0]))
    todo = []
    for mid, name, f in models:
        for r in ratios:
            todo.append((mid, name, 'r%02d' % round(100 * r), f, ['--ratio', str(r)]))
        for t in big_t:
            todo.append((mid, name, 't%dk' % (t // 1000), f, ['--tris', str(t), '--min-source-tris', str(big_min)]))
        if not retopo_only or mid in retopo_only.split(','):
            for r in retopo:
                todo.append((mid, name, 'bake%02d' % round(100 * r), f, ['--ratio', str(r), '--mode', 'retopo']))
    if '--collect' not in sys.argv:
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(run_one, py, f, os.path.join(tagdir, '%s_%s' % (mid, tg)), extra, timeout): (mid, tg) for mid, name, tg, f, extra in todo}
            for fu in cf.as_completed(futs):
                out, sec = fu.result()
                print('done %-12s %6.0fs' % (os.path.basename(out), sec), flush=True)
        print('all runs: %.0f s' % (time.time() - t0))
    # ---- summary
    prev = {}
    hist = os.path.join(out_root, 'history.tsv')
    if os.path.isfile(hist):
        for line in open(hist, encoding='utf-8'):
            p = line.rstrip('\n').split('\t')
            if len(p) >= 4 and p[0] != tag:
                prev[p[1]] = (p[0], p[2], p[3])        # the latest other tag wins
    rows = []
    for mid, name, tg, f, extra in todo:
        d = os.path.join(tagdir, '%s_%s' % (mid, tg))
        try:
            r = json.load(open(os.path.join(d, 'result.json'), encoding='utf-8'))
        except Exception:
            r = {'ok': False, 'error': 'no result.json'}
        if r.get('skipped'):
            continue
        key = '%s_%s' % (mid, tg)
        v, why = verdict(r, retopo=tg.startswith('bake'))
        n = notes.get(key, {})
        rows.append({'key': key, 'id': mid, 'name': name, 'target': tg, 'dir': d, 'result': r, 'auto_verdict': v, 'auto_problem': why,
                     'verdict': n.get('verdict', v), 'problem': n.get('problem', why), 'seen': 'verdict' in n})
    json.dump([{k: v for k, v in r.items() if k != 'dir'} for r in rows], open(os.path.join(tagdir, 'summary.json'), 'w'), ensure_ascii=False, indent=1)
    md = ['# validation set, tag `%s`' % tag, '',
          'Distances in % of the bounding box diagonal. tri% = triangles of the result / source. '
          'prev = verdict and Hausdorff of the previous tag. verdict* = set by a human (--notes), else automatic.', '',
          '| run | model | mode | src tris | faces | quad% | tris | tri% | non-manif (src) | flipped | sliver% (src) | Hausd% | mean% | silh% | shade | tex | UV islands | skin% | max infl | s | verdict | problem | prev |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    hl = []
    for r in rows:
        res = r['result']
        pv = prev.get(r['key'])
        pvs = '%s %s (%s)' % (pv[1], pv[2], pv[0]) if pv else ''
        if not res.get('ok'):
            md.append('| %s | %s | - | | | | | | | | | | | | | | | | | | **%s**%s | %s | %s |' % (r['key'], r['name'], r['verdict'], '*' if r['seen'] else '', r['problem'], pvs))
            hl.append('\t'.join([tag, r['key'], r['verdict'], '-', r['problem']]))
            continue
        m = res['metrics']
        modes = '+'.join('%s' % k for k, _ in sorted(res.get('modes', {}).items(), key=lambda kv: -kv[1]))
        md.append('| %s | %s | %s | %d | %d | %.0f | %d | %.0f | %d (%d) | %d | %.1f (%.1f) | %.2f | %.3f | %s | %s | %s | %s | %.0f | %d | %.0f | **%s**%s | %s | %s |' % (
            r['key'], r['name'], modes, res['source_tris'], m['faces'], m['quad_pct'], m['tri_equiv'], 100 * m['tri_ratio'],
            m['nonmanifold_edges'], m['nonmanifold_edges_source'], m['flipped'], m['sliver_pct'], m['sliver_pct_source'],
            m['hausdorff_pct'], m['mean_pct'], m.get('silhouette_diff_pct'), m.get('shade_diff'), m.get('tex_diff'), m['uv_islands'], m['skinned_vertex_pct'], m['max_influences'], res.get('seconds', 0),
            r['verdict'], '*' if r['seen'] else '', r['problem'], pvs))
        hl.append('\t'.join([tag, r['key'], r['verdict'], '%.2f' % m['hausdorff_pct'], r['problem']]))
    cnt = {}
    for r in rows:
        cnt[r['verdict']] = cnt.get(r['verdict'], 0) + 1
    md += ['', 'totals: ' + ', '.join('%s %d' % kv for kv in sorted(cnt.items()))]
    open(os.path.join(tagdir, 'summary.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    old = [l for l in open(hist, encoding='utf-8')] if os.path.isfile(hist) else []
    old = [l for l in old if not l.startswith(tag + '\t')]
    open(hist, 'w', encoding='utf-8').write(''.join(old) + '\n'.join(hl) + '\n')
    colors = {'good': (0.3, 1, 0.3), 'acceptable': (1, 0.9, 0.2), 'bad': (1, 0.3, 0.3)}
    lab = [('%s %s' % (r['key'].replace('bake', 'B'), {'good': 'G', 'acceptable': 'A', 'bad': 'BAR'[0]}.get(r['verdict'], '-')), r['dir'], colors.get(r['verdict'], (1, 1, 1))) for r in rows]
    try:
        contact_sheet(lab, tagdir, 'wire.png', os.path.join(tagdir, 'contact_wire.png'))
        contact_sheet(lab, tagdir, 'tex.png', os.path.join(tagdir, 'contact_tex.png'))
    except Exception as ex:
        print('contact sheet failed:', ex)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
