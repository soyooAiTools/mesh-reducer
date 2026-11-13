import sys; import os; ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
from core import preview
TEX=os.path.join(ROOT,'testdata','jier_diffuse.png')
CU={ # name: (center m, half, view)
 'hand':  ((0.40,-0.03,0.52),0.12,'front'),
 'handback':  ((-0.40,-0.03,0.52),0.12,'back'),
 'arm':   ((0.25,0.0,0.70),0.17,'front'),
 'armback':   ((-0.25,0.0,0.70),0.17,'back'),
 'legs':  ((0.0,0.0,0.33),0.22,'front'),
 'boots': ((0.0,0.0,0.13),0.20,'front34'),
 'bootside': ((0.1,0.0,0.13),0.18,'left'),
 'skirt': ((0.0,0.0,0.56),0.22,'front'),
 'back':  ((0.0,0.0,0.72),0.24,'back'),
 'torso': ((0.0,0.0,0.74),0.22,'front'),
 'head':  ((0.0,0.0,1.17),0.36,'front'),
 'skirtback': ((0.0,0.0,0.56),0.22,'back'),
 'skirtside': ((0.0,0.0,0.56),0.22,'left'),
 'face34': ((0.0,0.0,1.12),0.34,'front34'),
 'side':  ((0.0,0.0,0.75),0.80,'left'),
}
def render(objs,prefix,names=None,res=560,**kw):
    out=[]
    for n,(c,h,v) in CU.items():
        if names and n not in names: continue
        p='%s_%s.png'%(prefix,n)
        preview.render_views(objs,p,views=(v,),res=res,texture=TEX,zoom_box=(c,h),**kw); out.append(p)
    return out
