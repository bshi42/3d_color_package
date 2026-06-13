import bpy, sys, os
argv = sys.argv
argv = argv[argv.index("--")+1:] if "--" in argv else []
in_path  = os.path.abspath(argv[0])
out_path = os.path.abspath(argv[1])
angle    = float(argv[2])
island_m = float(argv[3])
pack_m   = float(argv[4])
avg      = int(argv[5])

bpy.ops.wm.read_homefile(use_empty=True)

# --- Import OBJ (new API first, then fallback) ---
try:
    bpy.ops.wm.obj_import(filepath=in_path)
except AttributeError:
    # Older Blender (<4.2) path
    bpy.ops.import_scene.obj(filepath=in_path, use_split_objects=False, use_split_groups=False)

# Get imported mesh object
objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']
if not objs:
    raise RuntimeError("No mesh imported from OBJ.")
obj = objs[0]
bpy.context.view_layer.objects.active = obj

# --- Unwrap ---
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(angle_limit=angle, island_margin=island_m,
                        correct_aspect=True, scale_to_bounds=False)
bpy.ops.uv.select_all(action='SELECT')

if avg:
    try:
        bpy.ops.uv.average_islands_scale()
    except Exception:
        pass
bpy.ops.uv.pack_islands(margin=pack_m)
bpy.ops.uv.select_all(action='DESELECT')
bpy.ops.object.mode_set(mode='OBJECT')

# --- Export OBJ (new API first, then fallbacks) ---
try:
    # 4.2+ operator. Some builds have 'export_keep_vertex_order' – try it first.
    bpy.ops.wm.obj_export(filepath=out_path,
                            export_selected_objects=True,
                            export_keep_vertex_order=True)
except TypeError:
    # Same operator but without that keyword
    bpy.ops.wm.obj_export(filepath=out_path, export_selected_objects=True)
except AttributeError:
    # Older operator
    try:
        bpy.ops.export_scene.obj(filepath=out_path, use_selection=True, keep_vertex_order=True)
    except TypeError:
        bpy.ops.export_scene.obj(filepath=out_path, use_selection=True)