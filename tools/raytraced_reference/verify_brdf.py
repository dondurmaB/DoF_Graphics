"""Blender --background --python this_file: measure the real GLSL and Cycles graph.

Zero-angular-size sun and a planar patch isolate the BRDF from shadow filters,
pixel coverage, interpolation and finite sun-disc integration. Linear EXR only.
"""
from pathlib import Path
import json
import math
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from brdf import cases, specular
from materials import surface_material


def main():
    import bpy
    from mathutils import Vector
    root = Path(__file__).resolve().parents[2]
    output = root/'output/brdf'
    output.mkdir(parents=True, exist_ok=True)
    samples = cases()
    gpu_input = ''.join(' '.join(map(str, (*c['l'], *c['v'], c['rough'], c['f0'])))+'\n' for c in samples)
    result = subprocess.run([str(root/'build/brdf_gpu_probe')], input=gpu_input, text=True, capture_output=True, check=True)
    gpu = list(map(float, result.stdout.split()))
    assert len(gpu)==len(samples)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    scene.render.engine='CYCLES'
    scene.cycles.device='CPU'
    scene.cycles.samples=64
    scene.cycles.use_denoising=False
    scene.cycles.use_adaptive_sampling=False
    scene.cycles.diffuse_bounces=scene.cycles.glossy_bounces=0
    scene.cycles.transmission_bounces=scene.cycles.volume_bounces=0
    scene.cycles.max_bounces=1
    scene.cycles.sample_clamp_direct=scene.cycles.sample_clamp_indirect=0
    scene.render.resolution_x=scene.render.resolution_y=16
    scene.render.resolution_percentage=100
    scene.render.use_compositing=False
    scene.render.image_settings.file_format='OPEN_EXR'
    scene.render.image_settings.color_depth='32'
    scene.render.filepath=str(output/'probe.exr')
    scene.world=bpy.data.worlds.new('black')
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=0
    bpy.ops.mesh.primitive_plane_add(size=200)
    plane=bpy.context.object
    camera_data=bpy.data.cameras.new('orthographic_probe')
    camera=bpy.data.objects.new('orthographic_probe',camera_data)
    scene.collection.objects.link(camera)
    camera_data.type='ORTHO';camera_data.ortho_scale=0.001
    scene.camera=camera
    sun_data=bpy.data.lights.new('unit_delta_sun','SUN')
    sun_data.energy=1;sun_data.angle=0
    sun=bpy.data.objects.new('unit_delta_sun',sun_data);scene.collection.objects.link(sun)
    records=[]
    for i,c in enumerate(samples):
        camera.location=Vector(c['v'])*3
        camera.rotation_euler=(-Vector(c['v'])).to_track_quat('-Z','Y').to_euler()
        sun.rotation_euler=(-Vector(c['l'])).to_track_quat('-Z','Y').to_euler()
        albedo=(0.2,0.4,0.7);sky=(0.1,0.2,0.3)
        mat=surface_material(bpy,'probe',sky,albedo=albedo,rough=c['rough'],spec=c['f0'])
        plane.data.materials.clear();plane.data.materials.append(mat)
        bpy.ops.render.render(write_still=True)
        image=bpy.data.images.load(scene.render.filepath,check_existing=False)
        pixels=list(image.pixels)
        measured=[sum(pixels[channel::4])/256 for channel in range(3)]
        bpy.data.images.remove(image)
        oracle=specular((0,0,1),c['l'],c['v'],c['rough'],c['f0'])
        expected=[albedo[k]*(1-c['f0'])*(sky[k]+c['l'][2]/math.pi)+gpu[i]*c['l'][2] for k in range(3)]
        error=max(abs(a-b) for a,b in zip(measured,expected))
        relative=error/max(max(expected),1e-6)
        gpu_rel=abs(gpu[i]-oracle)/max(abs(oracle),1e-6)
        records.append(dict(c,glsl_brdf=gpu[i],python_brdf=oracle,cycles_rgb=measured,glsl_rgb=expected,
                            absolute_error=error,relative_error=relative,gpu_oracle_relative_error=gpu_rel))
        print(f'BRDF {i+1}/{len(samples)}: Cycles/GLSL relative={relative:.6g}, GLSL/oracle={gpu_rel:.6g}',flush=True)
    report=dict(blender=bpy.app.version_string,device='CPU',cycles_samples=64,cases=records,
                max_absolute_error=max(r['absolute_error'] for r in records),
                max_relative_error=max(r['relative_error'] for r in records),
                max_gpu_oracle_relative_error=max(r['gpu_oracle_relative_error'] for r in records),
                tolerance='Cycles <= 0.005 relative + 0.00002 absolute; GLSL/oracle <= 0.003 relative + 1e-6 absolute')
    (output/'agreement.json').write_text(json.dumps(report,indent=2)+'\n')
    for r in records:
        assert r['absolute_error'] <= 0.00002+0.005*max(r['glsl_rgb']),r
        assert abs(r['glsl_brdf']-r['python_brdf']) <= 1e-6+0.003*abs(r['python_brdf']),r
    print('PASS: actual production GLSL and Cycles material agree.',flush=True)

if __name__=='__main__':main()
