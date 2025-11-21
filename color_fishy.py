import bpy
import colorsys
import copy
import numpy as np
import sys
sys.path.append("/snap/blender/common/BlenderPython")
from sklearn.datasets import make_blobs
import pandas as pd
import os


"""
This script colors a fishy 3d model.
It generates multiple variations of the fish coloration.
The generated textures are saved as files and a CSV file is generated with the parameters of each texture.

The fish has 5 visual features:
- base color (yellow, with some hue variation, saturation, and value)
- black stripes (with some randomization in width, spacing, and number)
- belly color (blue, with some hue variation, strength, and translation)
- tail color (green, with some hue variation, strength, and translation)
- "rosy cheeks" (red, with some hue variation, strength, and translation)


The fish color palette consists of the following colors:
blue (for belly), yellow (for base), green (for tail), black (for stripes), red (for rosy cheeks)

The following parameters exist and are sampled from a distribution:
- base_color_hue: sampled from a normal distribution
- base_color_sat: sampled from a normal distribution
- base_color_val: sampled from a normal distribution
- stripe_count: 80% chance of 4, 20% chance of 5        <---- cluster feature
- stripe_spacing: sampled from a normal distribution
- stripe_width: sampled from a normal distribution
- stripe_longitudinal_offset: sampled from a normal distribution
- belly_hue: two hue clusters, 1d make_blobs            <---- cluster feature
- belly_strength: sampled from a normal distribution
- belly_translation: sampled from a normal distribution
- tail_hue: two hue clusters, 1d make_blobs             <---- cluster feature
- tail_strength: sampled from a normal distribution
- rosy_cheeks_present: 5% chance of being present       <---- cluster feature
- rosy_cheeks_hue: sampled from a normal distribution, 0 if not present
- rosy_cheeks_strength: sampled from a normal distribution, 0 if not present
- rosy_cheeks_translation_y: sampled from a normal distribution, 0 if not present
- rosy_cheeks_translation_z: sampled from a normal distribution, 0 if not present

"""

# OUTPUT_DIR = "/home/alek/projects/3d_color_package/data/fishy"
OUTPUT_DIR = "/tmp/ml_data/fishy"
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMAGE_SIZE = 2048  # 1024..8192
IMPORT_MESH_PATH = None 

SEED = 49
N_SAMPLES = 250

ADD_NOISE = True
NOISE_AMOUNT = 0.10    # blend weight
NOISE_SCALE = 8.0      # higher = finer noise

# Palette
BLUE = colorsys.rgb_to_hsv(39/255., 70/255., 144/255.)
YELLOW = colorsys.rgb_to_hsv(242/255., 255/255., 73/255.)
GREEN = colorsys.rgb_to_hsv(21/255., 127/255., 31/255.)
BLACK = colorsys.rgb_to_hsv(0., 0., 0.)
RED = colorsys.rgb_to_hsv(255/255., 102/255., 99/255.)

BASE_COLOR_HUE_STD = 0.015
BASE_COLOR_SAT_STD = 0.05
BASE_COLOR_VAL_STD = 0.05

NOMINAL_FIRST_STRIPE_START = (0.1, -0.25, 0.07)
NOMINAL_FIRST_STRIPE_END = (0.0, -0.24, 0.17)

STRIPE_COUNT_PROB_4 = 0.8

NOMINAL_STRIPE_SPACING = 0.08
STRIPE_SPACING_STD = 0.01

NOMINAL_STRIPE_WIDTH = 0.03
STRIPE_WIDTH_STD = 0.005

NOMINAL_STRIPE_LONGITUDINAL_OFFSET = 0.00
STRIPE_LONGITUDINAL_OFFSET_STD = 0.01

# for make_blobs
BELLY_HUE_SHIFT_RANGE = (-0.1, 0.1)
BELLY_HUE_SHIFT_STD = 0.015

BELLY_STRENGTH_STD = 0.05
BELLY_TRANSLATION_STD = 0.05

TAIL_HUE_SHIFT_RANGE = (-0.1, 0.1)
TAIL_HUE_SHIFT_STD = 0.015

TAIL_STRENGTH_STD = 0.05

ROSY_CHEEKS_PROB = 0.05
ROSY_CHEEKS_HUE_STD = 0.015
ROSY_CHEEKS_STRENGTH_STD = 0.05
ROSY_CHEEKS_TRANSLATION_Y_STD = 0.05
ROSY_CHEEKS_TRANSLATION_Z_STD = 0.05

def sample_space(n_samples, random_state=42):
    np.random.seed(random_state)
    base_color_hue = np.random.normal(YELLOW[0], BASE_COLOR_HUE_STD, n_samples) % 1.0
    base_color_sat = np.clip(np.random.normal(0, BASE_COLOR_SAT_STD, n_samples) + YELLOW[1], 0, 1)
    base_color_val = np.clip(np.random.normal(0, BASE_COLOR_VAL_STD, n_samples) + YELLOW[2], 0, 1)
    stripe_count = np.random.choice([4, 5], n_samples, p=[STRIPE_COUNT_PROB_4, 1 - STRIPE_COUNT_PROB_4])
    stripe_spacing = np.random.normal(NOMINAL_STRIPE_SPACING, STRIPE_SPACING_STD, n_samples)
    stripe_width = np.random.normal(NOMINAL_STRIPE_WIDTH, STRIPE_WIDTH_STD, n_samples)
    stripe_longitudinal_offset = np.random.normal(NOMINAL_STRIPE_LONGITUDINAL_OFFSET, STRIPE_LONGITUDINAL_OFFSET_STD, n_samples)
    belly_hue_shift = make_blobs(n_samples=n_samples, n_features=1, centers=2, cluster_std=BELLY_HUE_SHIFT_STD, center_box=BELLY_HUE_SHIFT_RANGE)[0][:, 0]
    belly_hue = (BLUE[0] + belly_hue_shift) % 1.0
    belly_strength = np.clip(np.random.normal(1.0 - BELLY_STRENGTH_STD, BELLY_STRENGTH_STD, n_samples), 0, 1)
    belly_translation = np.random.normal(0, BELLY_TRANSLATION_STD, n_samples)
    tail_hue_shift = make_blobs(n_samples=n_samples, n_features=1, centers=2, cluster_std=TAIL_HUE_SHIFT_STD, center_box=TAIL_HUE_SHIFT_RANGE)[0][:, 0]
    tail_hue = (GREEN[0] + tail_hue_shift) % 1.0
    tail_strength = np.clip(np.random.normal(1.0 - TAIL_STRENGTH_STD, TAIL_STRENGTH_STD, n_samples), 0, 1)
    rosy_cheeks_present = np.random.choice([0, 1], n_samples, p=[1 - ROSY_CHEEKS_PROB, ROSY_CHEEKS_PROB])
    rosy_cheeks_hue = np.random.normal(RED[0], ROSY_CHEEKS_HUE_STD, n_samples) % 1.0
    rosy_cheeks_strength = np.clip(np.random.normal(1.0 - ROSY_CHEEKS_STRENGTH_STD, ROSY_CHEEKS_STRENGTH_STD, n_samples), 0, 1)
    rosy_cheeks_translation_y = np.random.normal(0, ROSY_CHEEKS_TRANSLATION_Y_STD, n_samples)
    rosy_cheeks_translation_z = np.random.normal(0, ROSY_CHEEKS_TRANSLATION_Z_STD, n_samples)
    
    data = [{
        "base_color_hue": base_color_hue[i],
        "base_color_sat": base_color_sat[i],
        "base_color_val": base_color_val[i],
        "stripe_count": stripe_count[i],
        "stripe_spacing": stripe_spacing[i],
        "stripe_width": stripe_width[i],
        "stripe_longitudinal_offset": stripe_longitudinal_offset[i],
        "belly_hue": belly_hue[i],
        "belly_strength": belly_strength[i],
        "belly_translation": belly_translation[i],
        "tail_hue": tail_hue[i],
        "tail_strength": tail_strength[i],
        "rosy_cheeks_present": rosy_cheeks_present[i],
        "rosy_cheeks_hue": rosy_cheeks_hue[i],
        "rosy_cheeks_strength": rosy_cheeks_strength[i],
        "rosy_cheeks_translation_y": rosy_cheeks_translation_y[i],
        "rosy_cheeks_translation_z": rosy_cheeks_translation_z[i],
    } for i in range(n_samples)]
    return data

def generate_stripes(
        stripe_spacing,
        stripe_width,
        stripe_longitudinal_offset,
        stripe_count,
):
    stripes = []
    for i in range(stripe_count):
        stripe = {
            "type": "capsule", 
            "a": NOMINAL_FIRST_STRIPE_START, 
            "b": NOMINAL_FIRST_STRIPE_END,
            "sigma": stripe_width, 
            "color": BLACK, 
            "strength": 1.0
            }
        stripe["b"] = (stripe["b"][0], stripe["b"][1] + i*stripe_spacing + stripe_longitudinal_offset, stripe["b"][2])
        stripe["a"] = (stripe["a"][0], stripe["a"][1] + i*stripe_spacing + stripe_longitudinal_offset, stripe["a"][2])
        stripes.append(stripe)
        mirrored_stripe = copy.deepcopy(stripe)
        mirrored_stripe["a"] = (-stripe["a"][0], stripe["a"][1], stripe["a"][2])
        mirrored_stripe["b"] = (-stripe["b"][0], stripe["b"][1], stripe["b"][2])
        stripes.append(mirrored_stripe)
    return stripes

def generate_base_color(base_color_hue, base_color_sat, base_color_val):
    rgb_color = colorsys.hsv_to_rgb(base_color_hue, base_color_sat, base_color_val)
    return [
        {"type": "sphere", "center": (0.0, 0.00, 0.0), "solid_radius": 2, "sigma": 1, "color": rgb_color, "strength": 1.0},
    ]

def generate_belly(belly_hue, belly_strength, belly_translation):
    rgb_color = colorsys.hsv_to_rgb(belly_hue, BLUE[1], BLUE[2])
    return [
        {"type": "sphere", "center": (0.0, -0.1+belly_translation, -0.55), "solid_radius": 0.5, "sigma": 0.05, "color": rgb_color, "strength": belly_strength},
    ]

def generate_tail(tail_hue, tail_strength):
    rgb_color = colorsys.hsv_to_rgb(tail_hue, GREEN[1], GREEN[2])
    return [
        {"type": "sphere", "center": (0.0, 0.5, 0), "solid_radius": 0.05, "sigma": 0.2, "color": rgb_color, "strength": tail_strength},
    ]
    
def generate_rosy_cheeks( rosy_cheeks_present,rosy_cheeks_hue, rosy_cheeks_strength, rosy_cheeks_translation_y, rosy_cheeks_translation_z):
    rgb_color = colorsys.hsv_to_rgb(rosy_cheeks_hue, RED[1], RED[2])
    if rosy_cheeks_present == 0:
        return []
    return [
        {"type": "sphere", "center": (0.07, -0.37+rosy_cheeks_translation_y, -0.03+rosy_cheeks_translation_z), "solid_radius": 0.02, "sigma": 0.01, "color": rgb_color, "strength": rosy_cheeks_strength},
        {"type": "sphere", "center": (-0.07, -0.37+rosy_cheeks_translation_y, -0.03+rosy_cheeks_translation_z), "solid_radius": 0.02, "sigma": 0.01, "color": rgb_color, "strength": rosy_cheeks_strength},
    ]

def generate_fishy_coloration(
        base_color_hue,
        base_color_sat,
        base_color_val,
        stripe_count,
        stripe_spacing,
        stripe_width,
        stripe_longitudinal_offset,
        belly_hue,
        belly_strength,
        belly_translation,
        tail_hue,
        tail_strength,
        rosy_cheeks_present,
        rosy_cheeks_hue,
        rosy_cheeks_strength,
        rosy_cheeks_translation_y,
        rosy_cheeks_translation_z,
):
    fishy = []
    fishy += generate_base_color(base_color_hue, base_color_sat, base_color_val)
    fishy += generate_belly(belly_hue, belly_strength, belly_translation)
    fishy += generate_tail(tail_hue, tail_strength)
    fishy += generate_rosy_cheeks(rosy_cheeks_present, rosy_cheeks_hue, rosy_cheeks_strength, rosy_cheeks_translation_y, rosy_cheeks_translation_z)
    fishy += generate_stripes(stripe_spacing, stripe_width, stripe_longitudinal_offset, stripe_count)
    return fishy


def ensure_object_and_uv():
    # Load/obtain object
    if IMPORT_MESH_PATH:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.obj(filepath=IMPORT_MESH_PATH, axis_forward='-Z', axis_up='Y')
        obj = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = obj
    else:
        obj = bpy.context.view_layer.objects.active
        assert obj and obj.type == 'MESH', "Select a mesh object or set IMPORT_MESH_PATH."

    # Ensure it has at least one UV map
    me = obj.data
    if not me.uv_layers:
        # Simple unwrap; better to unwrap manually for production
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.uv.smart_project(island_margin=0.02)
        bpy.ops.object.mode_set(mode='OBJECT')
    return obj

def world_vertex_positions(obj):
    mw = obj.matrix_world
    verts = obj.data.vertices
    P = np.array([mw @ v.co for v in verts], dtype=np.float64)
    return P

def gaussian_falloff(d, sigma):
    # alpha in [0,1] with Gaussian profile
    return np.exp(-0.5 * (d / (sigma + 1e-12))**2)

def gaussian_tail(d, sigma):
    # Gaussian that equals 1 at d=0 and decays outward
    return np.exp(-0.5 * (d / (sigma + 1e-12))**2)

def dist_point_to_segment(p, a, b):
    # Returns distance and the closest point param t in [0,1]
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-20:
        return np.linalg.norm(p - a), 0.0
    t = np.dot(p - a, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(p - closest), t


def apply_brushes(
    P,
    brushes,
    base_rgb=None,          # e.g., (0.8, 0.85, 0.9) or None for black
    add_noise=False,
    noise_amount=0.10,
    noise_scale=8.0,
):
    """
    P: (N,3) world-space vertex positions (numpy)
    brushes: list[dict] with type in {"sphere","capsule"}; supports solid_radius + sigma
    base_rgb: optional uniform base (r,g,b) in [0,1]
    """

    N = P.shape[0]
    if base_rgb is None:
        col = np.zeros((N, 3), dtype=np.float64)
    else:
        col = np.tile(np.asarray(base_rgb, dtype=np.float64), (N, 1))

    if add_noise:
        g = (np.sin(P[:,0]*noise_scale*3.1) +
             np.sin(P[:,1]*noise_scale*2.3 + 1.7) +
             np.sin(P[:,2]*noise_scale*4.0 + 3.4)) / 3.0
        g = (g - g.min()) / (g.max() - g.min() + 1e-12)
        noise_rgb = np.stack([g, g, g], axis=1)
        col = (1 - noise_amount) * col + noise_amount * noise_rgb

    def dist_point_to_segment_batch(P_, a_, b_):
        ab = b_ - a_
        denom = np.dot(ab, ab)
        if denom < 1e-20:
            return np.linalg.norm(P_ - a_, axis=1)
        t = np.clip(((P_ - a_) @ ab) / denom, 0.0, 1.0)
        closest = a_[None, :] + t[:, None] * ab[None, :]
        return np.linalg.norm(P_ - closest, axis=1)

    for b in brushes:
        strength = float(b.get("strength", 1.0))
        if b["type"] == "sphere":
            c = np.array(b["center"], dtype=np.float64)
            d_center = np.linalg.norm(P - c, axis=1)

            solid_r = float(b.get("solid_radius", 0.0))
            sigma   = float(b["sigma"])

            alpha = np.zeros_like(d_center)
            inside = d_center <= solid_r
            alpha[inside] = 1.0
            if np.any(~inside):
                d_tail = np.maximum(d_center - solid_r, 0.0)
                alpha[~inside] = gaussian_tail(d_tail[~inside], sigma)

        elif b["type"] == "capsule":
            a_pt = np.array(b["a"], dtype=np.float64)
            b_pt = np.array(b["b"], dtype=np.float64)
            d_axis = dist_point_to_segment_batch(P, a_pt, b_pt)

            solid_r = float(b.get("solid_radius", 0.0))
            sigma   = float(b["sigma"])

            alpha = np.zeros_like(d_axis)
            inside = d_axis <= solid_r
            alpha[inside] = 1.0
            if np.any(~inside):
                d_tail = np.maximum(d_axis - solid_r, 0.0)
                alpha[~inside] = gaussian_tail(d_tail[~inside], sigma)

        else:
            continue

        a = np.clip(alpha * strength, 0.0, 1.0)
        brush_rgb = np.array(b["color"], dtype=np.float64)
        col = col * (1.0 - a[:, None]) + brush_rgb[None, :] * a[:, None]

    return np.clip(col, 0.0, 1.0)

def write_vertex_colors(obj, rgb):
    me = obj.data
    # Color attribute on CORNER domain works well for baking
    layer_name = "SynColor"
    if layer_name in me.color_attributes:
        ca = me.color_attributes[layer_name]
    else:
        ca = me.color_attributes.new(name=layer_name, domain='CORNER', type='BYTE_COLOR')
    data = ca.data

    # Assign per-loop color from per-vertex color
    loops = me.loops
    for li, loop in enumerate(loops):
        vi = loop.vertex_index
        r, g, b = rgb[vi]
        data[li].color = (float(r), float(g), float(b), 1.0)

    # Make sure attribute is active for rendering (not strictly required)
    me.attributes.active_color = ca

def ensure_bake_material(obj):
    # Create a material that exposes the vertex color via Emission for baking.
    mat = bpy.data.materials.new(name="Syn_Bake_Mat")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    for n in list(nodes): nodes.remove(n)

    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "SynColor"
    links.new(attr.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out.inputs["Surface"])

    # Create target image and an image texture node (must be active for bake)
    img = bpy.data.images.new("SynTex", width=IMAGE_SIZE, height=IMAGE_SIZE, alpha=False, float_buffer=False)
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    nodes.active = tex  # crucial: active image node is the bake target

    # Assign the material
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return img, tex

def bake_to_image(obj, img, output_image_path):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    # Optional: try GPU if available; otherwise Blender falls back to CPU
    scene.cycles.device = 'GPU'
    scene.render.bake.margin = 16
    scene.render.bake.target = 'IMAGE_TEXTURES'

    # Bake EMIT so colors are not affected by lighting
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type='EMIT')

    # Save
    img.filepath_raw = output_image_path
    img.file_format = 'PNG'
    img.save()

def main():
    samples = sample_space(N_SAMPLES, random_state=SEED)
    df = pd.DataFrame(samples)
    df.to_csv(os.path.join(OUTPUT_DIR, "samples.csv"), index=False)

    obj = ensure_object_and_uv()
    P = world_vertex_positions(obj)
    for i, sample in enumerate(samples):
        brushes = generate_fishy_coloration(**sample)
        rgb = apply_brushes(P, brushes, add_noise=ADD_NOISE, noise_amount=NOISE_AMOUNT, noise_scale=NOISE_SCALE)
        write_vertex_colors(obj, rgb)
        img, tex = ensure_bake_material(obj)
        output_image_path = os.path.join(OUTPUT_DIR, f"fishy_{i:06d}.png")
        bake_to_image(obj, img, output_image_path)
        print("Wrote:", output_image_path)

if __name__ == "__main__":
    main()
