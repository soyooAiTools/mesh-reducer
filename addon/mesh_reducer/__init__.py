# -*- coding: utf-8 -*-
"""Mesh Reducer add-on (optional front end; the same core/ is used by the
stand-alone app).  3D View > N panel > 减面."""
import os
import sys
import time

import bpy
from bpy.props import IntProperty, FloatProperty, EnumProperty, BoolProperty, StringProperty, PointerProperty

bl_info = {                      # legacy add-on installs still read this
    "name": "Mesh Reducer 减面",
    "author": "MeshReducer",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "3D View > Sidebar > 减面",
    "description": "Loop-aware, quad-preserving mesh reduction",
    "category": "Mesh",
}

try:
    from . import core                      # packaged add-on: core/ is inside
except ImportError:                         # running from the source tree
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    import core
import importlib
api = importlib.import_module(core.__name__ + '.api')
prop = importlib.import_module(core.__name__ + '.prop')


class MR_Settings(bpy.types.PropertyGroup):
    target_type: EnumProperty(
        name="目标", description="Target by face count or by ratio",
        items=[('FACES', "面数", "Target polygon count"), ('RATIO', "比例", "Ratio of the original face count")],
        default='FACES')
    faces: IntProperty(name="目标面数", description="Target polygon count (a quad is one face)",
                       default=1500, min=4, soft_max=200000)
    ratio: FloatProperty(name="目标比例", description="Target face count as a fraction of the original",
                         default=0.5, min=0.01, max=1.0, subtype='FACTOR')
    mode: EnumProperty(
        name="模式", description="Reduction mode",
        items=[('CHARACTER', "角色（四边形环线）", "Remove whole edge loops, keep quads, joint loops and UV seams"),
               ('PROP', "静态道具（三角面）", "Attribute aware triangle decimation for static props")],
        default='CHARACTER')
    symmetry: EnumProperty(
        name="对称", description="Mirror axis: symmetric loops are removed together",
        items=[('AUTO', "自动", "Detect"), ('X', "X", ""), ('Y', "Y", ""), ('Z', "Z", ""), ('NONE', "关闭", "Off")],
        default='AUTO')
    protect_group: StringProperty(name="保护顶点组",
                                  description="Vertex group; loops through weighted vertices are removed last")
    preserve_seams: BoolProperty(name="保持 UV 接缝", description="Never move or merge UV seams", default=True)
    preserve_sharp: BoolProperty(name="保持硬边", description="Keep hard (split normal) edges", default=True)
    auto_joints: BoolProperty(name="自动保护关节环线",
                              description="Detect dense loops on limbs (knee, elbow, wrist) and keep them", default=True)
    hide_original: BoolProperty(name="隐藏原模型", description="Hide the original object after reducing (it is never modified)",
                                default=True)


class MR_OT_reduce(bpy.types.Operator):
    bl_idname = "mesh_reducer.reduce"
    bl_label = "减面"
    bl_description = "Reduce the active mesh into a new object (the original is kept)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == 'MESH'

    def execute(self, context):
        s = context.scene.mesh_reducer
        obj = context.active_object
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        n = len(obj.data.polygons)
        kw = {'target_faces': s.faces} if s.target_type == 'FACES' else {'ratio': s.ratio}
        if s.target_type == 'FACES' and s.faces >= n:
            self.report({'WARNING'}, "目标面数不小于当前面数 (%d)" % n)
            return {'CANCELLED'}
        t0 = time.time()
        wm = context.window_manager
        wm.progress_begin(0, 100)
        try:
            cb = lambda f, msg='': wm.progress_update(int(f * 100))
            if s.mode == 'CHARACTER':
                new, info = api.reduce_character(
                    obj, symmetry=s.symmetry, protect_group=s.protect_group or None,
                    preserve_seams=s.preserve_seams, preserve_sharp=s.preserve_sharp,
                    auto_joints=s.auto_joints, progress=cb, **kw)
            else:
                new, info = prop.reduce_prop(obj, symmetry=s.symmetry, preserve_seams=s.preserve_seams,
                                             preserve_sharp=s.preserve_sharp, progress=cb, **kw)
        except Exception as ex:
            self.report({'ERROR'}, "减面失败: %s" % ex)
            return {'CANCELLED'}
        finally:
            wm.progress_end()
        for o in context.selected_objects:
            o.select_set(False)
        new.select_set(True)
        context.view_layer.objects.active = new
        if s.hide_original:
            obj.hide_set(True)
        self.report({'INFO'}, "%d → %d 面 (四边形 %d, 三角形 %d)，用时 %.1f 秒" % (
            info['faces_before'], info['faces_after'], info.get('quads', 0), info.get('tris', 0), time.time() - t0))
        return {'FINISHED'}


class MR_PT_panel(bpy.types.Panel):
    bl_label = "模型减面"
    bl_idname = "MR_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "减面"

    def draw(self, context):
        s = context.scene.mesh_reducer
        lay = self.layout
        o = context.active_object
        if o is not None and o.type == 'MESH':
            lay.label(text="%s：%d 面" % (o.name, len(o.data.polygons)), icon='MESH_DATA')
        else:
            lay.label(text="请选择一个网格物体", icon='INFO')
        lay.prop(s, "mode")
        row = lay.row()
        row.prop(s, "target_type", expand=True)
        lay.prop(s, "faces" if s.target_type == 'FACES' else "ratio")
        lay.prop(s, "symmetry")
        if o is not None and o.type == 'MESH':
            lay.prop_search(s, "protect_group", o, "vertex_groups")
        col = lay.column(align=True)
        col.prop(s, "preserve_seams")
        col.prop(s, "preserve_sharp")
        if s.mode == 'CHARACTER':
            col.prop(s, "auto_joints")
        col.prop(s, "hide_original")
        lay.operator(MR_OT_reduce.bl_idname, icon='MOD_DECIM')


classes = (MR_Settings, MR_OT_reduce, MR_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.mesh_reducer = PointerProperty(type=MR_Settings)


def unregister():
    del bpy.types.Scene.mesh_reducer
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
