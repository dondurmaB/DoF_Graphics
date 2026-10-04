"""Measure a candidate BRDF, not a production shading change.

Blender --background --factory-startup --python-exit-code 1 --python this.py

Flat plane, orthographic camera, zero-angle unit sun, black world. Every pixel
has the same N,L,V, so averaging the central 8x8 removes edge/filter concerns.
Compare the common isotropic GGX + separable Smith + Schlick GLSL candidate
against Cycles Glossy with Color=F0, for both GGX and MULTI_GGX distributions.
This tests that particular mapping, not a claim that matching GGX is impossible.
Reference: Blender manual Glossy BSDF and Blender 4.0 Cycles release notes.
"""

import json
import math
from pathlib import Path
import tempfile
import bpy
import numpy as np
from mathutils import Vector


def candidate(nl, nv, roughness, f0, schlick=True):
    L = np.array((-math.sqrt(1 - nl * nl), 0, nl))
    V = np.array((math.sqrt(1 - nv * nv), 0, nv))
    H = (L + V) / np.linalg.norm(L + V)
    a = roughness * roughness
    D = a * a / (math.pi * (H[2] * H[2] * (a * a - 1) + 1) ** 2)

    def g(c):
        return 2 * c / (c + math.sqrt(a * a + (1 - a * a) * c * c))

    F = f0 + (1 - f0) * (1 - float(V @ H)) ** 5 if schlick else f0
    # BRDF * E * N.L with E=1; N.L cancels the BRDF denominator.
    return F * D * g(nl) * g(nv) / (4 * nv)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    s.cycles.samples = 64
    s.cycles.seed = 701
    s.cycles.use_denoising = False
    s.cycles.use_adaptive_sampling = False
    s.cycles.sample_clamp_direct = s.cycles.sample_clamp_indirect = 0
    s.cycles.max_bounces = 12
    s.cycles.glossy_bounces = 0
    s.render.resolution_x = s.render.resolution_y = 16
    s.render.resolution_percentage = 100
    s.render.image_settings.file_format = "OPEN_EXR"
    s.render.image_settings.color_depth = "32"
    s.render.image_settings.color_mode = "RGB"
    s.render.use_persistent_data = True
    s.world = bpy.data.worlds.new("black")
    s.world.use_nodes = True
    s.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0
    bpy.ops.mesh.primitive_plane_add(size=20)
    plane = bpy.context.object
    mat = bpy.data.materials.new("candidate_glossy")
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    n = mat.node_tree.nodes.new("ShaderNodeBsdfGlossy")
    out = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(n.outputs[0], out.inputs["Surface"])
    plane.data.materials.append(mat)
    camdata = bpy.data.cameras.new("camera")
    camdata.type = "ORTHO"
    camdata.ortho_scale = 0.1
    cam = bpy.data.objects.new("camera", camdata)
    s.collection.objects.link(cam)
    s.camera = cam
    lightdata = bpy.data.lights.new("unit_sun", "SUN")
    lightdata.energy = 1
    lightdata.angle = 0
    light = bpy.data.objects.new("sun", lightdata)
    s.collection.objects.link(light)
    rows = []
    with tempfile.TemporaryDirectory(prefix="dof-brdf-") as temporary:
        path = Path(temporary) / "patch.exr"
        for distribution in ("GGX", "MULTI_GGX"):
            n.distribution = distribution
            for nl, nv in ((1.0, 1.0), (0.5, 0.7), (0.2, 0.3)):
                L = Vector((-math.sqrt(1 - nl * nl), 0, nl))
                V = Vector((math.sqrt(1 - nv * nv), 0, nv))
                cam.location = V * 4
                cam.rotation_euler = (-V).to_track_quat("-Z", "Y").to_euler()
                light.rotation_euler = (-L).to_track_quat("-Z", "Y").to_euler()
                for rough in (0.15, 0.4, 0.8):
                    n.inputs["Roughness"].default_value = rough
                    for f0 in (0.04, 0.3, 0.8):
                        n.inputs["Color"].default_value = (f0, f0, f0, 1)
                        s.render.filepath = str(path)
                        bpy.ops.render.render(write_still=True)
                        img = bpy.data.images.load(str(path), check_existing=False)
                        pixels = np.array(img.pixels[:], dtype=np.float32).reshape(16, 16, 4)
                        measured = float(pixels[4:12, 4:12, :3].mean())
                        bpy.data.images.remove(img)
                        expected = candidate(nl, nv, rough, f0)
                        rows.append(
                            dict(
                                distribution=distribution,
                                n_dot_l=nl,
                                n_dot_v=nv,
                                roughness=rough,
                                F0=f0,
                                cycles=measured,
                                schlick_ggx=expected,
                                constant_f0_ggx=candidate(nl, nv, rough, f0, False),
                                relative_error=abs(measured - expected) / max(abs(measured), 1e-9),
                            )
                        )
    result = {
        "blender": bpy.app.version_string,
        "samples": 64,
        "seed": 701,
        "denoising": False,
        "candidate": "isotropic GGX, alpha=roughness^2, separable Smith G1 product, Schlick Fresnel",
        "mapping": "Cycles Glossy Color=F0 (not assumed to supply Schlick Fresnel)",
        "rows": rows,
        "max_relative_error": max(r["relative_error"] for r in rows),
    }
    path = Path(__file__).resolve().parents[2] / "output/verification/specular.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("SPECULAR_MEASUREMENT", path, result["max_relative_error"])


if __name__ == "__main__":
    main()
