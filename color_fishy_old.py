# ------------------------------------------------------------
# Synthetic texture generator with volumetric brushes (Blender)
# - Evaluates 3D primitives with Gaussian falloff to color vertices
# - Bakes vertex colors to an image using the model's UV map
# Tested on Blender 3.6/4.0
# ------------------------------------------------------------
import bpy
import numpy as np
from mathutils import Vector
import copy


# ----------------------- CONFIG -----------------------------
# If you already have the mesh selected in Blender, set IMPORT_MESH_PATH=None.
IMPORT_MESH_PATH = None  # e.g. r"/path/to/fish_like_mesh.obj"
OUTPUT_IMAGE_PATH = r"/tmp/synthetic_texture.png"  # change this
IMAGE_SIZE = 4096  # 1024..8192

# Base countershading (vertical gradient in world Z)
BASE_TOP = (0.85, 0.90, 0.95)     # dorsal
BASE_BOTTOM = (0.25, 0.28, 0.30)  # ventral
COUNTERSHADE_GAMMA = 1.6

# Random seed (for parameterized generation)
SEED = 42
np.random.seed(SEED)

# Palette
BLUE = (39/255., 70/255., 144/255.)
YELLOW = (242/255., 255/255., 73/255.)
GREEN = (21/255., 127/255., 31/255.)
BLACK = (0., 0., 0.)
RED = (255/255., 102/255., 99/255.)

# Brushes: sphere or capsule (line segment). Units: Blender world units.
# color is RGB in 0..1; strength is multiplier on alpha; sigma controls softness.
NOMINAL_FIRST_STRIPE_START = (0.1, -0.25, 0.07)
NOMINAL_FIRST_STRIPE_END = (0.0, -0.24, 0.17)

NOMINAL_STRIPE_SPACING = 0.08
STRIPE_SPACING_STD = 0.01

NOMINAL_STRIPE_WIDTH = 0.03
STRIPE_WIDTH_STD = 0.005

NOMINAL_STRIPE_LONGITUDINAL_OFFSET = 0.00
STRIPE_LONGITUDINAL_OFFSET_STD = 0.01

BELLY_HUE_SHIFT_RANGE = (-0.1, 0.1)
BELLY_HUE_SHIFT_STD = 0.025

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
     
STRIPES = generate_stripes(NOMINAL_STRIPE_SPACING, NOMINAL_STRIPE_WIDTH, NOMINAL_STRIPE_LONGITUDINAL_OFFSET, 5)

BRUSHES = [
    # base color
    {"type": "sphere", "center": (0.0, 0.00, 0.0), "solid_radius": 2, "sigma": 1, "color": YELLOW, "strength": 1.0},

    # lateral band as a capsule
    
]

# belly patch
BRUSHES += [
    {"type": "sphere", "center": (0.0, -0.1, -0.55), "solid_radius": 0.5, "sigma": 0.05, "color": BLUE, "strength": 1.0},
]
# tail patch
BRUSHES += [
    {"type": "sphere", "center": (0.0, 0.5, 0), "solid_radius": 0.05, "sigma": 0.2, "color": GREEN, "strength": 1.0},
]
# cheeks
BRUSHES += [
    {"type": "sphere", "center": (0.07, -0.37, -0.03), "solid_radius": 0.02, "sigma": 0.01, "color": RED, "strength": 1.0},
    {"type": "sphere", "center": (-0.07, -0.37, -0.03), "solid_radius": 0.02, "sigma": 0.01, "color": RED, "strength": 1.0},
]

BRUSHES += STRIPES



# Optionally add noise-based subtle mottling in 3D
ADD_NOISE = True
NOISE_AMOUNT = 0.10    # blend weight
NOISE_SCALE = 8.0      # higher = finer noise
# --------------------- END CONFIG ---------------------------

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

def apply_brushes(P):
    # Base countershading...
    z = P[:, 2]
    z0, z1 = z.min(), z.max()
    t = (z - z0) / max(1e-12, (z1 - z0))
    t = t ** COUNTERSHADE_GAMMA
    base = np.stack([
        BASE_BOTTOM[0] * (1 - t) + BASE_TOP[0] * t,
        BASE_BOTTOM[1] * (1 - t) + BASE_TOP[1] * t,
        BASE_BOTTOM[2] * (1 - t) + BASE_TOP[2] * t,
    ], axis=1)

    col = base.copy()

    if ADD_NOISE:
        g = (np.sin(P[:,0]*NOISE_SCALE*3.1) +
             np.sin(P[:,1]*NOISE_SCALE*2.3 + 1.7) +
             np.sin(P[:,2]*NOISE_SCALE*4.0 + 3.4)) / 3.0
        g = (g - g.min()) / (g.max() - g.min() + 1e-12)
        noise_rgb = np.stack([g, g, g], axis=1)
        col = (1 - NOISE_AMOUNT) * col + NOISE_AMOUNT * noise_rgb

    for b in BRUSHES:
        strength = b.get("strength", 1.0)
        if b["type"] == "sphere":
            c = np.array(b["center"], dtype=np.float64)
            d_center = np.linalg.norm(P - c, axis=1)

            solid_r = float(b.get("solid_radius", 0.0))
            sigma   = float(b["sigma"])

            # Hard core (alpha=1) inside solid_radius
            alpha = np.zeros_like(d_center)
            inside = d_center <= solid_r
            alpha[inside] = 1.0

            # Feather outside the core using Gaussian on (d - solid_r)
            if np.any(~inside):
                d_tail = np.maximum(d_center - solid_r, 0.0)
                alpha[~inside] = gaussian_tail(d_tail[~inside], sigma)

            a = np.clip(alpha * strength, 0.0, 1.0)

        elif b["type"] == "capsule":
            # unchanged from before; optional: support solid_radius similarly
            a_pt = np.array(b["a"], dtype=np.float64)
            b_pt = np.array(b["b"], dtype=np.float64)

            def dist_point_to_segment_batch(P_, a_, b_):
                ab = b_ - a_
                denom = np.dot(ab, ab)
                if denom < 1e-20:
                    return np.linalg.norm(P_ - a_, axis=1)
                t = np.clip(((P_ - a_) @ ab) / denom, 0.0, 1.0)
                closest = a_[None, :] + t[:, None] * ab[None, :]
                return np.linalg.norm(P_ - closest, axis=1)

            d_axis = dist_point_to_segment_batch(P, a_pt, b_pt)

            solid_r = float(b.get("solid_radius", 0.0))
            sigma   = float(b["sigma"])

            alpha = np.zeros_like(d_axis)
            inside = d_axis <= solid_r
            alpha[inside] = 1.0
            if np.any(~inside):
                d_tail = np.maximum(d_axis - solid_r, 0.0)
                alpha[~inside] = gaussian_tail(d_tail[~inside], sigma)

            a = np.clip(alpha * strength, 0.0, 1.0)

        else:
            continue

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

def bake_to_image(obj, img):
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
    img.filepath_raw = OUTPUT_IMAGE_PATH
    img.file_format = 'PNG'
    img.save()

def main():
    obj = ensure_object_and_uv()
    P = world_vertex_positions(obj)
    rgb = apply_brushes(P)
    write_vertex_colors(obj, rgb)
    img, _ = ensure_bake_material(obj)
    bake_to_image(obj, img)
    print("Wrote:", OUTPUT_IMAGE_PATH)

if __name__ == "__main__":
    main()
