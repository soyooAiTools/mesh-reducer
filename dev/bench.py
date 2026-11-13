"""dev harness: reduce Jier to N faces, evaluate, render"""
import sys,time,json; import os; ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
import bpy
from core import io_util, preview, api, metrics
TEX=os.path.join(ROOT,'testdata','jier_diffuse.png')
REG=json.load(open(os.path.join(ROOT,'testdata','jier_regions.json'))) if __import__('os').path.exists(os.path.join(ROOT,'testdata','jier_regions.json')) else None
def run(target, tag='', render=True, **kw):
    io_util.reset_scene()
    o=io_util.import_file(os.path.join(ROOT,'testdata','jier_high.fbx'))[0]
    r,info=api.reduce_character(o,target_faces=target,**kw)
    m=metrics.evaluate(o.data,r.data,texture=TEX,regions=REG)
    row=metrics.summary_row(m)
    print(info); print(row); print('geo',m['geometry'])
    if REG: print({k:(v['ring_loops_result'],v['ring_loops_original'],v['tris_result'],v['tris_original']) for k,v in m['loops']['regions'].items()})
    json.dump(m,open(os.path.join(ROOT,'dev','tmp','')+'m_%d%s.json'%(target,tag),'w'),indent=1)
    if render:
        preview.render_views([r,o],os.path.join(ROOT,'dev','tmp','')+'b_%d%s.png'%(target,tag),views=('front','back'),res=800,texture=TEX)
    return o,r,m
if __name__=='__main__':
    for t in sys.argv[1:]:
        run(int(t))
