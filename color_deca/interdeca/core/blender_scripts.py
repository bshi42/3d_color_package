"""Deterministic Blender script generation for InterDeCA."""

import textwrap
from pathlib import Path

from color_deca.interdeca.types import BlenderBakeParameters, BlenderUvParameters


class BlenderScriptFactory:
    """Build Blender Python scripts without launching Blender."""

    def build_prepare_atlas_script(self) -> str:
        return textwrap.dedent(
            """
            import bpy, sys
            argv = sys.argv
            argv = argv[argv.index("--")+1:] if "--" in argv else []
            in_path  = argv[0]
            out_path = argv[1]
            merge_d  = float(argv[2])
            ang      = float(argv[3])
            island_m = float(argv[4])

            bpy.ops.wm.read_factory_settings(use_empty=True)

            try:
                bpy.ops.wm.obj_import(filepath=in_path, forward_axis='Y', up_axis='Z')
            except AttributeError:
                bpy.ops.import_scene.obj(filepath=in_path, use_split_objects=False, use_split_groups=False, axis_forward='Y', axis_up='Z')

            obj = [o for o in bpy.context.selected_objects if o.type=='MESH'][0]
            bpy.context.view_layer.objects.active = obj

            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            try:
                bpy.ops.mesh.remove_doubles(threshold=merge_d)
            except Exception:
                try:
                    bpy.ops.mesh.merge_by_distance(distance=merge_d)
                except Exception:
                    bpy.ops.mesh.merge(type='DISTANCE', distance=merge_d)

            bpy.ops.uv.smart_project(angle_limit=ang, island_margin=island_m, correct_aspect=True, scale_to_bounds=False)
            bpy.ops.object.mode_set(mode='OBJECT')

            bpy.ops.wm.obj_export(
                filepath=out_path,
                export_selected_objects=True,
                export_triangulated_mesh=True,
                forward_axis='Y', up_axis='Z',
                export_materials=False,
            )
            """
        )

    def build_bake_texture_script(self) -> str:
        return textwrap.dedent(
            """
            import bpy, sys, os
            argv = sys.argv
            argv = argv[argv.index("--")+1:] if "--" in argv else []
            src_path, tgt_path, png_in, png_out, sz, extru, margin, merge_d = argv
            sz = int(sz); extru = float(extru); margin = int(margin); merge_d = float(merge_d)

            bpy.ops.wm.read_factory_settings(use_empty=True)
            try: bpy.ops.wm.obj_import(filepath=tgt_path, forward_axis='Y', up_axis='Z')
            except AttributeError: bpy.ops.import_scene.obj(filepath=tgt_path, use_split_objects=False, use_split_groups=False, axis_forward='Y', axis_up='Z')
            tgt = [o for o in bpy.context.selected_objects if o.type=='MESH'][0]

            imp_ok = False
            try:
                bpy.ops.wm.obj_import(filepath=src_path, forward_axis='Y', up_axis='Z'); imp_ok=True
            except Exception: pass
            if not imp_ok:
                raise RuntimeError("Cannot import aligned mesh: " + src_path)
            src = [o for o in bpy.context.selected_objects if o.type=='MESH'][-1]

            bpy.context.view_layer.objects.active = tgt
            bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
            try: bpy.ops.mesh.remove_doubles(threshold=merge_d)
            except Exception:
                try: bpy.ops.mesh.merge_by_distance(distance=merge_d)
                except Exception: bpy.ops.mesh.merge(type='DISTANCE', distance=merge_d)
            bpy.ops.object.mode_set(mode='OBJECT')

            tgt.data.materials.clear(); src.data.materials.clear()
            m_src = bpy.data.materials.new("MatSrc"); m_src.use_nodes=True
            nt = m_src.node_tree; nodes = nt.nodes
            img_node = nodes.new('ShaderNodeTexImage'); img_node.image = bpy.data.images.load(png_in)
            bsdf = next(n for n in nodes if n.type=='BSDF_PRINCIPLED')
            nt.links.new(img_node.outputs['Color'], bsdf.inputs['Base Color'])
            src.data.materials.append(m_src)

            m_tgt = bpy.data.materials.new("MatTgt"); m_tgt.use_nodes=True
            nt2 = m_tgt.node_tree; nodes2 = nt2.nodes
            imgT = bpy.data.images.new("BakeTarget", width=sz, height=sz, alpha=False)
            img_node_t = nodes2.new('ShaderNodeTexImage'); img_node_t.image = imgT
            tgt.data.materials.append(m_tgt)

            bpy.ops.object.select_all(action='DESELECT')
            src.select_set(True); tgt.select_set(True)
            bpy.context.view_layer.objects.active = tgt

            for n in nodes2: n.select = False
            nodes2.active = img_node_t; img_node_t.select = True

            scn = bpy.context.scene
            scn.render.engine = 'CYCLES'
            scn.cycles.device = 'CPU'
            b = scn.render.bake
            b.use_selected_to_active = True
            b.cage_extrusion = extru
            b.margin = margin
            b.use_pass_direct = False
            b.use_pass_indirect = False
            b.use_pass_color = True

            bpy.ops.object.bake(type='DIFFUSE')

            imgT.filepath_raw = png_out
            imgT.file_format = 'PNG'
            imgT.save()
            """
        )

    def prepare_atlas_args(
        self,
        in_obj: Path,
        out_obj: Path,
        parameters: BlenderUvParameters,
    ) -> list[str]:
        return [
            str(in_obj),
            str(out_obj),
            str(parameters.merge_dist),
            str(parameters.smart_angle),
            str(parameters.island_margin),
        ]

    def bake_texture_args(
        self,
        source_mesh: Path,
        target_mesh: Path,
        input_texture: Path,
        output_texture: Path,
        parameters: BlenderBakeParameters,
    ) -> list[str]:
        return [
            str(source_mesh),
            str(target_mesh),
            str(input_texture),
            str(output_texture),
            str(parameters.bake_size),
            str(parameters.bake_extrusion),
            str(parameters.bake_margin_px),
            str(parameters.merge_dist),
        ]
