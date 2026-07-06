"""Blender-side worker for downsample_meshes.py — do not run directly.

Invoked as:  blender -b --factory-startup --python blender_worker.py -- '<job json>'

The job JSON (built by downsample_meshes.py) selects one of two modes:

* decimate:     Decimate modifier (COLLAPSE) — quadric edge collapse that
                interpolates UVs, so the original texture keeps working.
* remesh_bake:  Uniform remesh (QuadriFlow when the mesh is manifold, voxel
                remesh otherwise), Smart UV Project, then a Cycles CPU
                DIFFUSE/COLOR bake of the original texture onto the new UVs.

Prints one machine-readable ``RESULT_JSON:{...}`` line for the caller.
Validated against Blender 5.1.2 headless.
"""

import json
import math
import sys
import time

import addon_utils
import bmesh
import bpy

# Symmetric OBJ axis convention: import->export round-trips coordinates
# unchanged, so downsampled meshes stay in the original pipeline's frame.
AXES = {"forward_axis": "NEGATIVE_Z", "up_axis": "Y"}


def log(*a):
    print("[worker]", *a, flush=True)


def import_obj(path):
    bpy.ops.wm.obj_import(filepath=path, validate_meshes=True, **AXES)
    mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError("no mesh found in OBJ")
    # Highest-poly mesh is the model; robust to stray empties/parts.
    return max(mesh_objs, key=lambda o: len(o.data.polygons))


def make_active(obj):
    for o in bpy.context.scene.objects:
        o.select_set(o is obj)
    bpy.context.view_layer.objects.active = obj


def export_obj(obj, path):
    make_active(obj)
    bpy.ops.wm.obj_export(
        filepath=path,
        export_selected_objects=True,
        export_uv=True,
        export_normals=True,
        export_materials=False,  # no mtllib: texture is paired by filename
        export_triangulated_mesh=True,
        path_mode="AUTO",
        **AXES,
    )


def tri_count(obj):
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def bbox_diagonal(obj):
    from mathutils import Vector
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = [min(c[i] for c in corners) for i in range(3)]
    hi = [max(c[i] for c in corners) for i in range(3)]
    return math.dist(lo, hi)


def uv_ok(me):
    if not me.uv_layers:
        return False
    return any(d.uv[0] != 0.0 or d.uv[1] != 0.0
               for d in me.uv_layers.active.data[:2000])


def clean_and_triangulate(obj, eps=1e-6):
    """Voxel remesh can emit zero-length edges / sliver faces; scrub them."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=eps)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bmesh.ops.dissolve_degenerate(bm, dist=eps, edges=bm.edges[:])
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


# ---------------------------------------------------------------- modes

def run_decimate(obj, target_faces):
    n_in = len(obj.data.polygons)
    make_active(obj)
    mod = obj.modifiers.new("decimate", type="DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = min(1.0, target_faces / n_in)
    mod.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=mod.name)
    if not uv_ok(obj.data):
        raise RuntimeError("UVs lost during decimation")
    return obj, {"method": "decimate_collapse"}


def uniform_remesh(obj, target_faces):
    """Remesh to ~target_faces triangles with near-uniform edge lengths."""
    make_active(obj)
    n_before = len(obj.data.polygons)

    # QuadriFlow gives the nicest uniform topology but silently CANCELLED-s
    # on non-manifold scans, leaving geometry untouched — hence both checks.
    try:
        res = bpy.ops.object.quadriflow_remesh(
            mode="FACES", target_faces=max(4, target_faces // 2),
            use_mesh_symmetry=False, seed=0)
        if "FINISHED" in res and len(obj.data.polygons) != n_before:
            return "quadriflow"
    except Exception as e:
        log("quadriflow raised:", e)

    # Voxel remesh fallback: surface tris ~ 2*area/voxel_size^2, then refine.
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    area = sum(f.calc_area() for f in bm.faces)
    bm.free()
    vs = math.sqrt(2.0 * area / max(1, target_faces))
    for _ in range(4):
        for m in list(obj.modifiers):
            obj.modifiers.remove(m)
        mod = obj.modifiers.new("vox", "REMESH")
        mod.mode = "VOXEL"
        mod.voxel_size = vs
        ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        ntris = sum(len(p.vertices) - 2 for p in ev.data.polygons)
        log(f"voxel_size={vs:.6g} -> {ntris} tris (target {target_faces})")
        if 0.93 * target_faces <= ntris <= 1.07 * target_faces:
            break
        vs *= math.sqrt(ntris / target_faces)
    make_active(obj)
    bpy.ops.object.modifier_apply(modifier="vox")
    return "voxel"


def run_remesh_bake(orig, job, target_faces):
    # Source material: original texture -> Base Color (the bake source).
    img_src = bpy.data.images.load(job["input_texture"])
    img_src.colorspace_settings.name = "sRGB"
    mat = bpy.data.materials.new("src_mat")
    mat.use_nodes = True
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = img_src
    mat.node_tree.links.new(
        tex.outputs["Color"],
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
    orig.data.materials.clear()
    orig.data.materials.append(mat)

    make_active(orig)
    bpy.ops.object.duplicate()
    remesh = bpy.context.view_layer.objects.active
    remesh.data.materials.clear()

    method = uniform_remesh(remesh, target_faces)
    clean_and_triangulate(remesh)
    log(f"remeshed via {method}: {tri_count(remesh)} tris")

    # New UVs for the new topology.
    make_active(remesh)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.003,
                             area_weight=0.0, correct_aspect=True,
                             scale_to_bounds=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Bake-target image node must be the material's active+selected node.
    size = job["bake_resolution"]
    bake_img = bpy.data.images.new("bake_tex", size, size,
                                   alpha=False, float_buffer=False)
    bake_img.colorspace_settings.name = "sRGB"
    tmat = bpy.data.materials.new("bake_mat")
    tmat.use_nodes = True
    bake_node = tmat.node_tree.nodes.new("ShaderNodeTexImage")
    bake_node.image = bake_img
    tmat.node_tree.nodes.active = bake_node
    bake_node.select = True
    remesh.data.materials.append(tmat)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    margin = max(16, size // 128)
    extrusion = bbox_diagonal(orig) * 0.01

    orig.select_set(True)  # source selected, target (remesh) active
    remesh.select_set(True)
    bpy.context.view_layer.objects.active = remesh
    t0 = time.time()
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"},
                        use_selected_to_active=True,
                        cage_extrusion=extrusion, margin=margin,
                        use_clear=True)
    log(f"bake: {time.time() - t0:.1f}s ({size}px, margin {margin})")

    # Sanity-gate the bake: use_clear leaves failed/empty bakes cleared to
    # black, so a near-zero non-black fraction means the bake wrote nothing.
    import numpy as np
    px = np.empty(len(bake_img.pixels), dtype=np.float32)
    bake_img.pixels.foreach_get(px)
    nonblack = float((px.reshape(-1, 4)[:, :3].max(axis=1) > 0.02).mean())
    log(f"bake non-black fraction: {nonblack:.4f}")
    if nonblack < 0.02:
        raise RuntimeError(f"bake produced a near-empty texture "
                           f"(non-black fraction {nonblack:.4f})")

    bake_img.filepath_raw = job["output_texture"]
    bake_img.file_format = "PNG"
    bake_img.save()
    return remesh, {"method": f"remesh_{method}_bake", "bake_px": size,
                    "bake_nonblack_frac": round(nonblack, 4)}


# ---------------------------------------------------------------- main

def main():
    job = json.loads(sys.argv[sys.argv.index("--") + 1])
    t0 = time.time()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    # read_factory_settings resets add-ons; the snap build needs Cycles
    # re-enabled explicitly before the engine can be selected for baking.
    if job["mode"] == "remesh_bake":
        addon_utils.enable("cycles", default_set=True, persistent=True)

    orig = import_obj(job["input_obj"])
    orig_faces, orig_verts = tri_count(orig), len(orig.data.vertices)
    # Closed triangulated surface: faces ~= 2 * vertices (Euler).
    target_faces = 2 * job["target_vertices"]

    if job["mode"] == "decimate":
        out_obj, extra = run_decimate(orig, target_faces)
    elif job["mode"] == "remesh_bake":
        out_obj, extra = run_remesh_bake(orig, job, target_faces)
    else:
        raise RuntimeError(f"unknown mode {job['mode']!r}")

    if not uv_ok(out_obj.data):
        raise RuntimeError("output mesh has no usable UVs")
    export_obj(out_obj, job["output_obj"])

    result = {
        "ok": True,
        "orig_vertices": orig_verts, "orig_faces": orig_faces,
        "vertices": len(out_obj.data.vertices), "faces": tri_count(out_obj),
        "seconds": round(time.time() - t0, 1),
        **extra,
    }
    print("RESULT_JSON:" + json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("RESULT_JSON:" + json.dumps({"ok": False, "error": str(e)}),
              flush=True)
        raise
