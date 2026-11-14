"""dev/split_parts.py SRC.FBX OUTDIR 'part:target,...'  -> character + axe + crossbow as separate skinned FBX"""
import sys,os,time,json
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bpy, numpy as np
from core import io_util, materials, preview, metrics, prop, api
def load(src):
    io_util.reset_scene()
    objs=io_util.import_file(src)
    materials.link_textures(objs, os.path.dirname(os.path.abspath(src)))
    return objs[0]
def mat_counts(me):
    mi=np.empty(len(me.polygons),np.int32); me.polygons.foreach_get('material_index',mi)
    return {me.materials[i].name:int((mi==i).sum()) for i in range(len(me.materials))}
def texmap(o):
    return {i:materials.base_image_path(s.material) for i,s in enumerate(o.material_slots)}
# closeups: name -> (view, center (world m), half size)
CLOSE={'face':('front',(0,0,1.78),0.22),'hands':('front',(0.72,0,1.25),0.2),'torso':('front',(0,0,1.35),0.4),
       'axe':('front',(-1.75,0,1.55),0.55),'crossbow':('front',(-1.05,0,1.28),0.38),'feather':('left',(0,0.15,1.9),0.25),
       'legs':('front',(0,0,0.45),0.5),'back':('back',(0,0,1.2),0.6)}
def closeups(r,o,outdir,tag,names=None,res=800):
    os.makedirs(outdir,exist_ok=True)
    for n,(v,c,h) in CLOSE.items():
        if names and n not in names: continue
        preview.render_views([r,o],os.path.join(outdir,'%s_%s.png'%(tag,n)),views=(v,),res=res,zoom_box=(c,h))

import bmesh, collections, mathutils
src,outdir=sys.argv[1],sys.argv[2]
jobs=[(p.split(':')[0],int(p.split(':')[1])) for p in sys.argv[3].split(',')]
stem=os.path.splitext(os.path.basename(src))[0]; axes=io_util.fbx_axes(src)
o=load(src); me=o.data; name=o.name
# parts = connected pieces whose weights use only these bones (范海辛: weapons on the hand prop bones)
PART_BONES=json.loads(os.environ['PART_BONES']) if 'PART_BONES' in os.environ else {'axe':['bone20'],'crossbow':['bone17','bone18','bone19']}
PART_BONES={k:set(v) for k,v in PART_BONES.items()}
# classify faces: connected components (by position) of each material, dominant bones
bm=bmesh.new(); bm.from_mesh(me)
dl=bm.verts.layers.deform.active
key=lambda v:tuple(round(c,6) for c in v.co)
adj=collections.defaultdict(list)
for f in bm.faces:
    for v in f.verts: adj[key(v)].append(f.index)
part=['character']*len(bm.faces); seen=set()
bm.faces.ensure_lookup_table()
for f in bm.faces:
    if f.index in seen: continue
    st=[f.index]; seen.add(f.index); comp=[]
    while st:
        i=st.pop(); comp.append(i)
        for v in bm.faces[i].verts:
            for j in adj[key(v)]:
                if j not in seen and bm.faces[j].material_index==f.material_index: seen.add(j); st.append(j)
    names={o.vertex_groups[g].name for i in comp for v in bm.faces[i].verts for g,w in v[dl].items() if w>1e-4}
    for p,bs in PART_BONES.items():
        if names and names<=bs:
            for i in comp: part[i]=p
print('PARTS',collections.Counter(part))
objs={}
for p in ['character']+list(PART_BONES):
    c=o.copy(); c.data=me.copy(); bpy.context.scene.collection.objects.link(c)
    b2=bmesh.new(); b2.from_mesh(c.data); b2.faces.ensure_lookup_table()
    bmesh.ops.delete(b2,geom=[b2.faces[i] for i in range(len(part)) if part[i]!=p],context='FACES')
    b2.to_mesh(c.data); b2.free(); c.name=name+'_'+p
    # drop material slots and vertex groups this part does not use
    used=sorted({pp.material_index for pp in c.data.polygons})
    if len(used)<len(c.data.materials):
        mats=[c.data.materials[i] for i in used]; remap={o_:n_ for n_,o_ in enumerate(used)}
        idx=[remap[pp.material_index] for pp in c.data.polygons]
        c.data.materials.clear()
        for m_ in mats: c.data.materials.append(m_)
        for pp,i_ in zip(c.data.polygons,idx): pp.material_index=i_
    objs[p]=c
os.makedirs(os.path.join(outdir,'previews'),exist_ok=True)
for p,c in objs.items():
    print('SPLIT',p,len(c.data.polygons),mat_counts(c.data))
    io_util.export_file(os.path.join(outdir,'%s_%s_原始.FBX'%(stem,p)),[c],copy_textures=True,axes=axes)
for p,t in jobs:
    c=objs[p]; mt=None
    if p=='character':
        mc=mat_counts(c.data); fixed={'眼睛':40,'羽毛':60}
        rest=t-sum(v for k,v in fixed.items() if any(n.endswith(k) for n in mc))
        big={n:v for n,v in mc.items() if not any(n.endswith(k) for k in fixed)}
        tot=sum(big.values()); mt={}
        for i,s in enumerate(c.material_slots):
            n=s.material.name; f=next((v for k,v in fixed.items() if n.endswith(k)),None)
            mt[i]=f if f is not None else int(round(rest*big[n]/tot))
    r,info=prop.reduce_prop(c,target_faces=t,method='FAST',material_targets=mt)
    r.name='%s_%s'%(name,p); r.data.name=r.name
    ni=[sum(1 for g in v.groups if g.weight>0) for v in r.data.vertices]
    print('RESULT',p,t,len(r.data.polygons),mat_counts(r.data),'maxinf',max(ni),'unweighted',sum(1 for x in ni if x==0))
    out=os.path.join(outdir,'%s_%s_%d.FBX'%(stem,p,t))
    io_util.export_file(out,[r],copy_textures=True,axes=axes)
    m=metrics.evaluate(c.data,r.data,texture=texmap(c)); print('ROW',p,t,metrics.summary_row(m))
    tag='%s_%s_%d'%(stem,p,t); pv=os.path.join(outdir,'previews')
    if p=='character':
        preview.render_views([r,c],os.path.join(pv,tag+'_front.png'),views=('front',),res=1000,wire=False)
        preview.render_views([r,c],os.path.join(pv,tag+'_back.png'),views=('back',),res=1000,wire=False)
        closeups(r,c,pv,tag,names=['face','hands','torso','legs'])
    else:
        co=np.array([c.matrix_world@v.co for v in c.data.vertices]); lo,hi=co.min(0),co.max(0)
        ctr=tuple(((lo+hi)/2).tolist()); h=float((hi-lo).max()/2*1.1)
        preview.render_views([r,c],os.path.join(pv,tag+'.png'),views=('front',),res=900,zoom_box=(ctr,h))
        preview.render_views([r,c],os.path.join(pv,tag+'_wire.png'),views=('front',),res=900,zoom_box=(ctr,h),wire=True)
    r.hide_set(True)
