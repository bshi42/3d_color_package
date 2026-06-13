# Blender Batch Texture Padding/Baking Script (Version 2)
#
# This script is designed to handle OBJ models and textures stored in separate directories.
# It will manually link the texture to the model before baking with a margin.
#
# How to run:
# 1. Edit the THREE CONFIGURATION variables below.
# 2. Open a command prompt or terminal.
# 3. Run Blender in the background with this script:
#    blender --background --python /path/to/this/script/batch_pad_textures.py

import bpy
import os

# --- CONFIGURATION ---
# Edit these THREE paths to match your project structure.

# 1. Directory containing your original .obj models.
#    Use absolute paths. Example Windows: "C:\\Users\\YourUser\\Desktop\\Fish_Models"
#    Example Linux/Mac: "/home/user/projects/fish_models"
ORIGINAL_TEXTURED_MODELS_DIR = "/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/Models"

# 2. Directory containing your original .png texture files.
#    The script assumes texture names match model names (e.g., fish_a.obj -> fish_a.png).
RESAMPLED_MODELS_DIR = "/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/Texture"

# 3. Directory where the new models, materials (.mtl), and padded textures will be saved.
#    This script will create it if it doesn't exist.
FINAL_OUTPUT_DIR = "/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/BakedTexture"

# 4. The size of the color gutter/margin to add in pixels. 8 or 16 is usually good.
MARGIN_PIXELS = 16

# --- END CONFIGURATION ---

def process_models():
    """Main function to process all models."""
    print("--- Starting Batch Re-Texturing (v6.1) ---")
    
    # Validate paths
    for path in [ORIGINAL_TEXTURED_MODELS_DIR, RESAMPLED_MODELS_DIR]:
        if not os.path.isdir(path):
            print(f"Error: Required directory does not exist: {path}")
            return

    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    resampled_files = [f for f in os.listdir(RESAMPLED_MODELS_DIR) if f.lower().endswith(('.ply', '.obj'))]
    if not resampled_files:
        print(f"Error: No resampled models found in {RESAMPLED_MODELS_DIR}")
        return

    for resampled_file in resampled_files:
        try:
            process_single_model(resampled_file)
        except Exception as e:
            print(f"!! An error occurred while processing {resampled_file}: {e}")
            print("!! Skipping this file.")
    print("--- Batch re-texturing complete. ---")

def process_single_model(resampled_file):
    """Unwraps the resampled model and re-projects the original texture onto it."""
    print(f"\nProcessing: {resampled_file}")
    
    base_name = resampled_file.replace('_resampled.ply', '').replace('_resampled.obj', '')
    original_model_file = f"{base_name}.obj"
    original_model_path = os.path.join(ORIGINAL_TEXTURED_MODELS_DIR, original_model_file)
    
    if not os.path.exists(original_model_path):
        print(f"Warning: Corresponding original model not found at {original_model_path}. Skipping.")
        return

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Load resampled model (ACTIVE target)
    resampled_model_path = os.path.join(RESAMPLED_MODELS_DIR, resampled_file)
    if resampled_file.lower().endswith('.ply'):
        bpy.ops.wm.ply_import(filepath=resampled_model_path)
    else:
        bpy.ops.wm.obj_import(filepath=resampled_model_path)
    target_obj = bpy.context.selected_objects[0]
    target_obj.name = "Target_Resampled"

    # Load original textured model (SELECTED source)
    bpy.ops.wm.obj_import(filepath=original_model_path)
    source_obj = bpy.context.selected_objects[0]
    source_obj.name = "Source_Original"
    
    # --- BEGIN FIX: Manually find and apply the source texture ---
    
    # Construct the expected path to the padded texture from the previous script
    source_texture_filename = f"{base_name}_padded.png"
    source_texture_path = os.path.join(ORIGINAL_TEXTURED_MODELS_DIR, source_texture_filename)
    
    if not os.path.exists(source_texture_path):
        print(f"Warning: Source texture not found at {source_texture_path}. Skipping bake.")
        return
        
    # Load the image data into Blender
    source_image = bpy.data.images.load(source_texture_path)
    print(f"Successfully located source texture: {source_texture_path}")

    # Ensure the source object has a material and connect our image to it
    if not source_obj.data.materials:
        source_obj.data.materials.append(bpy.data.materials.new(name="SourceMaterial"))
    
    source_mat = source_obj.data.materials[0]
    source_mat.use_nodes = True
    source_nodes = source_mat.node_tree.nodes
    
    # Find or create an Image Texture node and assign our loaded image
    source_image_node = next((n for n in source_nodes if n.type == 'TEX_IMAGE'), None)
    if not source_image_node:
        source_image_node = source_nodes.new('ShaderNodeTexImage')
    source_image_node.image = source_image
    
    # Ensure it's connected to the shader
    principled_bsdf = next((n for n in source_nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if principled_bsdf:
        source_mat.node_tree.links.new(source_image_node.outputs['Color'], principled_bsdf.inputs['Base Color'])
        
    # --- END FIX ---
    
    print("Generating new UV map for resampled model...")
    bpy.context.view_layer.objects.active = target_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')

    width, height = source_image.size
    new_image_name = f"{base_name}_final_texture"
    bpy.ops.image.new(name=new_image_name, width=width, height=height)
    new_image = bpy.data.images[new_image_name]

    target_mat = bpy.data.materials.new(name=f"{base_name}_FinalMaterial")
    target_mat.use_nodes = True
    target_nodes = target_mat.node_tree.nodes
    target_nodes.clear()
    
    bake_target_node = target_nodes.new('ShaderNodeTexImage')
    bake_target_node.image = new_image
    target_nodes.active = bake_target_node
    
    bsdf = target_nodes.new('ShaderNodeBsdfPrincipled')
    out = target_nodes.new('ShaderNodeOutputMaterial')
    target_mat.node_tree.links.new(bake_target_node.outputs['Color'], bsdf.inputs['Base Color'])
    target_mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    
    target_obj.data.materials.clear()
    target_obj.data.materials.append(target_mat)
    
    print("Baking texture from original to resampled model...")
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.bake_type = 'DIFFUSE'
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.render.bake.use_pass_color = True
    bpy.context.scene.render.bake.margin = MARGIN_PIXELS
    bpy.context.scene.render.bake.use_selected_to_active = True
    bpy.context.scene.render.bake.extrusion = 0.1
    bpy.context.scene.render.bake.max_ray_distance = 0.1

    bpy.ops.object.select_all(action='DESELECT')
    source_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj

    bpy.ops.object.bake()

    final_texture_filename = f"{base_name}_final.png"
    final_texture_path = os.path.join(FINAL_OUTPUT_DIR, final_texture_filename)
    new_image.filepath_raw = final_texture_path
    new_image.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    new_image.save()
    print(f"Saved final re-projected texture to: {final_texture_path}")

    final_model_filename = f"{base_name}_final.obj"
    final_model_path = os.path.join(FINAL_OUTPUT_DIR, final_model_filename)
    bpy.ops.wm.obj_export(
        filepath=final_model_path,
        export_selected_objects=True
    )
    print(f"Saved final textured model to: {final_model_path}")


if __name__ == "__main__":
    process_models()