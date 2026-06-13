# Blender Batch Re-Texturing Script (Version 7.0 - Direct Re-Projection)
#
# This script performs a definitive re-texturing workflow in a single stage.
# It uses the original textures as the source, applying them to the aligned models,
# and then re-projects them onto new, optimized UVs for the resampled models.
#
# MUST BE RUN WITH AN OFFICIAL PORTABLE BLENDER DISTRIBUTION (4.2+).

import bpy
import os
import re

# --- CONFIGURATION ---
# Please carefully edit these FOUR directory paths.

# 1. Directory of the rigidly aligned models produced by Slicer DeCA.
#    This is the SOURCE GEOMETRY for the bake. (e.g., contains 'Mchenga_m1_align.obj')
ALIGNED_MODELS_DIR = "/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/deca_out/2025_09-06_16_04_26/alignedModels"

# 2. Directory of the resampled models produced by Slicer DeCA.
#    This is the TARGET GEOMETRY for the bake. (e.g., contains 'Mchenga_m1_resampled.obj')
RESAMPLED_MODELS_DIR = "/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/deca_out/2025_09-06_16_04_26/resampledModels"

# 3. Directory containing your ORIGINAL, un-padded .png texture files.
#    This is the SOURCE COLOR for the bake.
ORIGINAL_TEXTURES_DIR = "/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/Texture"

# 4. Directory where the FINAL, correctly re-textured models and their new textures will be saved.
FINAL_OUTPUT_DIR = "/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/deca_out/2025_09-06_16_04_26/Final_Textured_Models"

MARGIN_PIXELS = 16
# --- END CONFIGURATION ---

# Optional: set to True if you want GPU (requires proper device setup in Blender prefs)
USE_GPU = False

# ------------------------------
# HELPER UTILITIES
# ------------------------------

def _clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Purge orphans (Blender 3.2+)
    try:
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except Exception:
        pass

def _import_single_mesh(filepath: str) -> bpy.types.Object:
    before = set(bpy.data.objects.keys())
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=filepath)
        else:
            bpy.ops.import_scene.obj(filepath=filepath)
    elif ext == ".ply":
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=filepath)
        else:
            bpy.ops.import_mesh.ply(filepath=filepath)
    else:
        raise ValueError(f"Unsupported extension: {ext} ({filepath})")

    imported = [bpy.data.objects[n] for n in set(bpy.data.objects.keys()) - before]
    meshes = [o for o in imported if o.type == 'MESH']
    if not meshes:
        raise RuntimeError(f"No mesh data imported from: {filepath}")

    bpy.ops.object.select_all(action='DESELECT')
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()

    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    return obj

def _guess_texture_path(base: str) -> str | None:
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(ORIGINAL_TEXTURES_DIR, base + ext)
        if os.path.exists(p):
            return p
    return None

def _make_source_material_with_texture(obj: bpy.types.Object, image_path: str):
    """Clear materials and build: Image Texture -> Emission -> Output."""
    img = bpy.data.images.load(image_path, check_existing=True)
    obj.data.materials.clear()

    mat = bpy.data.materials.new(name=f"{obj.name}_SourceMat")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    n_tex  = nt.nodes.new("ShaderNodeTexImage")
    n_emit = nt.nodes.new("ShaderNodeEmission")
    n_out  = nt.nodes.new("ShaderNodeOutputMaterial")

    n_tex.image = img
    n_tex.image.colorspace_settings.name = "sRGB"
    nt.links.new(n_tex.outputs["Color"], n_emit.inputs["Color"])
    nt.links.new(n_emit.outputs["Emission"], n_out.inputs["Surface"])

    obj.data.materials.append(mat)
    return mat, n_tex

def _setup_target_material_with_bake_image(obj: bpy.types.Object, w: int, h: int, img_name: str):
    if img_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[img_name])
    bpy.ops.image.new(name=img_name, width=w, height=h, alpha=False, float=False)
    bake_img = bpy.data.images[img_name]

    obj.data.materials.clear()
    mat = bpy.data.materials.new(name=f"{obj.name}_BakedMat")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    n_img  = nt.nodes.new("ShaderNodeTexImage")
    n_bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    n_out  = nt.nodes.new("ShaderNodeOutputMaterial")

    n_img.image = bake_img
    nt.links.new(n_img.outputs["Color"], n_bsdf.inputs["Base Color"])
    nt.links.new(n_bsdf.outputs["BSDF"], n_out.inputs["Surface"])
    obj.data.materials.append(mat)

    # ACTIVE image node is required for baking
    nt.nodes.active = n_img
    return bake_img, n_img

def _smart_unwrap(obj: bpy.types.Object):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')

def _recalc_normals(*objs: bpy.types.Object):
    for obj in objs:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')

def _ensure_cycles(use_gpu=False):
    scn = bpy.context.scene
    scn.render.engine = 'CYCLES'
    if use_gpu:
        try:
            prefs = bpy.context.preferences.addons['cycles'].preferences
            # Pick one that matches your setup: 'CUDA', 'OPTIX', 'HIP', 'METAL'
            prefs.compute_device_type = 'CUDA'
            scn.cycles.device = 'GPU'
        except Exception:
            scn.cycles.device = 'CPU'

def _has_uvs(obj: bpy.types.Object) -> bool:
    me = obj.data
    return bool(me.uv_layers and me.uv_layers.active)

# ------------------------------
# CORE
# ------------------------------

def process_single(base: str, resampled_path: str, aligned_path: str, tex_path: str):
    print(f"\n--- Processing base='{base}' ---")
    _clean_scene()

    target = _import_single_mesh(resampled_path)
    target.name = "Target_Resampled"

    source = _import_single_mesh(aligned_path)
    source.name = "Source_Aligned"

    # Source must have UVs to sample the original texture
    if not _has_uvs(source):
        raise RuntimeError(
            f"Aligned source has no UVs: {aligned_path}\n"
            f"Cannot project original texture. Ensure UVs are present in the aligned OBJ."
        )

    # Build fresh emission material on source with the original texture
    src_mat, src_tex_node = _make_source_material_with_texture(source, tex_path)
    src_img = src_tex_node.image
    bake_w, bake_h = int(src_img.size[0]), int(src_img.size[1])

    # Target unwrap + bake target material
    _smart_unwrap(target)
    bake_img, bake_node = _setup_target_material_with_bake_image(
        target, bake_w, bake_h, img_name=f"{base}_final_texture"
    )

    # Normals help ray hits
    _recalc_normals(source, target)

    # Bake setup
    _ensure_cycles(USE_GPU)
    scn = bpy.context.scene
    b = scn.render.bake
    b.use_selected_to_active = True
    b.use_cage = False
    b.margin = MARGIN_PIXELS
    b.use_clear = True  # <-- Blender 4.5 name
    # Reasonable ray distance for Sel->Active
    try:
        b.max_ray_distance = max(target.dimensions) * 0.05
    except Exception:
        pass

    # Select: target active; source selected
    bpy.ops.object.select_all(action='DESELECT')
    source.select_set(True)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target

    print("Baking (Selected→Active, type=EMIT)...")
    bpy.ops.object.bake(type='EMIT')

    # Save baked texture
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    tex_out = os.path.join(FINAL_OUTPUT_DIR, f"{base}_final.png")
    bake_img.filepath_raw = tex_out
    bake_img.file_format = 'PNG'
    scn.render.image_settings.color_mode = 'RGB'
    bake_img.save()
    print(f"Saved baked texture: {tex_out}")

    # Point material to the saved path for better external references
    bake_img.filepath = tex_out

    # Export target as OBJ (+MTL) referencing the baked texture
    mdl_out = os.path.join(FINAL_OUTPUT_DIR, f"{base}_final.obj")
    bpy.ops.object.select_all(action='DESELECT')
    target.select_set(True)
    bpy.context.view_layer.objects.active = target

    try:
        bpy.ops.wm.obj_export(
            filepath=mdl_out,
            export_selected_objects=True,
            export_materials=True,
            export_normals=True,
            export_uv=True,
            path_mode='AUTO',
        )
    except TypeError:
        bpy.ops.export_scene.obj(
            filepath=mdl_out,
            use_selection=True,
            use_materials=True,
            use_normals=True,
            use_uvs=True,
            path_mode='AUTO',
        )

    print(f"Saved model: {mdl_out}")

def run_batch():
    print("--- Starting Batch Re-Texturing (v8.1) ---")
    for p in (ALIGNED_MODELS_DIR, RESAMPLED_MODELS_DIR, ORIGINAL_TEXTURES_DIR):
        if not os.path.isdir(p):
            print(f"ERROR: Missing directory: {p}")
            return

    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

    resampled_files = [
        f for f in os.listdir(RESAMPLED_MODELS_DIR)
        if f.lower().endswith((".obj", ".ply")) and "_resampled" in f.lower()
    ]
    if not resampled_files:
        print(f"ERROR: No resampled models found in {RESAMPLED_MODELS_DIR}")
        return

    for resampled_file in sorted(resampled_files):
        # Derive base (strip *_resampled.* plus any trailing token)
        base = os.path.splitext(resampled_file)[0]
        base = re.sub(r"_resampled$", "", base, flags=re.IGNORECASE)
        base = re.sub(r"_resampled_?[a-z0-9]*$", "", base, flags=re.IGNORECASE)

        resampled_path = os.path.join(RESAMPLED_MODELS_DIR, resampled_file)

        aligned_path = None
        for name in (f"{base}_align.obj", f"{base}_align.ply"):
            cand = os.path.join(ALIGNED_MODELS_DIR, name)
            if os.path.exists(cand):
                aligned_path = cand
                break

        if not aligned_path:
            print(f"SKIP: Missing aligned model for '{base}' (looked for {base}_align.obj/.ply).")
            continue

        tex_path = _guess_texture_path(base)
        if not tex_path:
            print(f"SKIP: Missing texture for '{base}' in {ORIGINAL_TEXTURES_DIR} "
                  f"(expected {base}.png/.jpg/.jpeg).")
            continue

        try:
            process_single(base, resampled_path, aligned_path, tex_path)
        except Exception as ex:
            print(f"!! ERROR while processing '{resampled_file}': {ex}")
            print("!! Skipping.")

    print("--- Batch re-texturing complete. ---")

if __name__ == "__main__":
    run_batch()