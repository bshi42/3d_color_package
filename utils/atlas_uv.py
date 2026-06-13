import bpy, sys

# ---- args ----
argv = sys.argv
argv = argv[argv.index("--")+1:] if "--" in argv else []
in_path  = argv[0]
out_path = argv[1]
merge_d  = float(argv[2])
ang      = float(argv[3])
island_m = float(argv[4])

bpy.ops.wm.read_homefile(use_empty=True)

# ---- Import OBJ in RAS (no axis rotation) ----
try:
    bpy.ops.wm.obj_import(filepath=in_path, forward_axis='Y', up_axis='Z')
except AttributeError:
    # Older fallbacks (not expected on 4.5, but harmless)
    bpy.ops.import_scene.obj(filepath=in_path,
                             use_split_objects=False, use_split_groups=False,
                             axis_forward='Y', axis_up='Z')

obj = [o for o in bpy.context.selected_objects if o.type == 'MESH'][0]
bpy.context.view_layer.objects.active = obj

# ---- Clean & UV ----
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')

# Merge by distance (cover multiple Blender versions)
for op in (
    lambda: bpy.ops.mesh.remove_doubles(threshold=merge_d),
    lambda: bpy.ops.mesh.merge_by_distance(distance=merge_d),
    lambda: bpy.ops.mesh.merge(type='DISTANCE', distance=merge_d),
):
    try:
        op(); break
    except Exception:
        pass

# Triangulate
try:
    bpy.ops.mesh.quads_convert_to_tris()
except Exception:
    pass

# Smart UV Project
bpy.ops.uv.smart_project(angle_limit=ang, island_margin=island_m,
                         correct_aspect=True, scale_to_bounds=False)

bpy.ops.object.mode_set(mode='OBJECT')

# ---- Export OBJ in RAS (no axis rotation) ----
# NOTE: don't pass non-existent args; do force axes.
bpy.ops.wm.obj_export(
    filepath=out_path,
    export_selected_objects=True,
    export_triangulated_mesh=True,
    forward_axis='Y', up_axis='Z',
    export_materials=False,
)