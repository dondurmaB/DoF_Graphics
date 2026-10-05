"""Stage-2 contract regressions; GPU group executes the production GLSL.

The stale/missing/dimension cases port the guarantees in cafe-rebuild's
test_comparison.py to linear EXRs and per-artifact provenance.
"""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'renderer/mitsuba'))
import render as R
import experiment_contract as E
import stage2_experiment as X
from traditional_dof import signed_coc_radius_px


class SensorFootprint(unittest.TestCase):
    def test_actual_sample_ray_footprint(self):
        import mitsuba as mi
        mi.set_variant('scalar_rgb')
        for f, s, stop in ((50.,1.8873722024030972,1.8),(85.,2.,1.2),(28.,.5,16.)):
            lens=R.Lens([0,0,0],[0,0,1],f,24.,stop,s)
            sensor=mi.load_dict(lens.sensor(960,540,1,True,'box'))
            for z in (.3*s,s,10*s):
                def hit(v):
                    ray,_=sensor.sample_ray(0.,.5,mi.Point2f(.5,v),mi.Point2f(.5,1.))
                    return np.array(ray.o)+np.array(ray.d)*((z-ray.o.z)/ray.d.z)
                p,q=hit(.5),hit(.51)
                measured=abs(float(p[1]/((q[1]-p[1])/.01)))*540
                expected=abs(signed_coc_radius_px(z,f/1000,.024,stop,s,540))
                self.assertAlmostEqual(measured,expected,delta=max(2e-4,expected*2e-5))
                self.assertAlmostEqual(lens.coc_pixels(z,540)/2,expected,delta=1e-9)


class Contracts(unittest.TestCase):
    def setUp(self):
        import mitsuba as mi
        mi.set_variant('scalar_rgb')
        self.mi=mi
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.d=Path(self.temp.name)
        self.context=dict(scene_hash='scene-1', camera={'focus_distance_m':2.},resolution=[20,20],
                          integrator={'type':'path','max_depth':12,'rr_depth':6,'hide_emitters':False},
                          gather={'shader_sha256':'actual-source-hash'},noise_fraction=.1)
        self.depth=np.ones((20,20),np.float32);self.depth[:,10:]=5
        np.save(self.d/'depth.npy',self.depth);E.seal(self.d/'depth.npy',self.context,'depth')
        for name in ('sharp-a','sharp-b','reference-a','reference-b','gather-naive-a','gather-naive-b',
                     'gather-weighted-a','gather-weighted-b'):
            a=np.full((20,20,3),.5 if name.startswith('reference') else .7,np.float32)
            p=self.d/(name+'.exr');mi.Bitmap(a).write(str(p))
            extra=dict(seed=17 if name.endswith('a') else 997,spp=1024,denoising=False,
                       integrator=self.context['integrator'])
            if name.startswith('gather'):
                extra.update(variant=name.split('-')[1],shader_sha256='actual-source-hash',
                             depth_sha256=E.file_hash(self.d/'depth.npy'),
                             input_sha256=E.file_hash(self.d/f'sharp-{name[-1]}.exr'))
            E.seal(p,self.context,name,**extra)

    def test_valid_pair_and_linear_metrics(self):
        r=E.compare(self.d,self.context)
        self.assertTrue(r['qualified'])
        self.assertAlmostEqual(r['rows'][0]['mae'],.2,places=6)
        self.assertAlmostEqual(r['rows'][0]['psnr_db'],13.9794,places=3)

    def test_no_pairs_fails(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(ValueError):X.main(['--out',empty,'--compare-only'])

    def test_stale_scene_or_camera_rejected(self):
        for field,value in (('scene_hash','changed'),('camera',{'focus_distance_m':3.})):
            c=copy.deepcopy(self.context);c[field]=value
            with self.assertRaises(ValueError):E.compare(self.d,c)

    def test_modified_file_rejected_even_with_same_name(self):
        p=self.d/'reference-a.exr';p.write_bytes(p.read_bytes()+b'changed')
        with self.assertRaises(ValueError):E.compare(self.d,self.context)

    def test_dimensions_rejected_even_with_valid_digest(self):
        p=self.d/'reference-a.exr';self.mi.Bitmap(np.ones((19,20,3),np.float32)).write(str(p))
        meta=json.loads(Path(str(p)+'.json').read_text());meta['sha256']=E.file_hash(p)
        E.write_json(str(p)+'.json',meta)
        with self.assertRaises(ValueError):E.compare(self.d,self.context)

    def test_missing_qualification_or_reused_seed_rejected(self):
        p=Path(str(self.d/'reference-b.exr')+'.json');original=json.loads(p.read_text())
        for field,value in (('seed',17),('spp',None),('spp',512),('denoising',True),('integrator',{})):
            record=copy.deepcopy(original);record[field]=value;E.write_json(p,record)
            with self.assertRaises(ValueError):E.compare(self.d,self.context)

    def test_pair_noise_formula_and_failed_gate(self):
        a=np.zeros((20,20,3));b=np.ones_like(a);b[:10]*=-1
        n=E.noise(a,b)
        self.assertAlmostEqual(n['std_estimate'],1/math.sqrt(2))
        self.assertAlmostEqual(n['rmse_estimate'],1/math.sqrt(2))
        p=self.d/'reference-b.exr';self.mi.Bitmap(np.ones_like(b,dtype=np.float32)).write(str(p))
        E.seal(p,self.context,'reference-b',seed=997,spp=1024,denoising=False,integrator=self.context['integrator'])
        with self.assertRaises(ValueError):E.compare(self.d,self.context)
        self.assertFalse((self.d/'results.json').exists())

    def test_gather_variant_or_input_mismatch_rejected(self):
        p=Path(str(self.d/'gather-naive-a.exr')+'.json');r=json.loads(p.read_text());r['variant']='weighted'
        E.write_json(p,r)
        with self.assertRaises(ValueError):E.compare(self.d,self.context)


class PortEquivalence(unittest.TestCase):
    def test_native_depth_encoding_matches_linear_adapter(self):
        from gl_gather import Gather
        rng=np.random.default_rng(12)
        rgb=rng.random((48,64,3),dtype=np.float32)*3
        depth=np.full((48,64),1.5,np.float32);depth[:,32:]=8.;depth[20:25]=2.
        camera=dict(focal_length_mm=85.,sensor_height_mm=24.,focus_distance_m=2.,f_number=1.2)
        n,f=.01,1000.
        raw=((f+n-2*n*f/depth)/(f-n)+1)*.5
        with Gather() as g:
            np.testing.assert_array_equal(g.run(rgb,depth,camera,mode=1),rgb)
            for variant in ('naive','weighted'):
                linear=g.run(rgb,depth,camera,variant)
                native=g.run(rgb,raw,camera,variant,raw_depth=True)
                np.testing.assert_allclose(linear,native,atol=4e-4,rtol=1e-4)
            # At focus, BOTH variants must preserve HDR values and orientation.
            focused=np.full_like(depth,2.)
            for variant in ('naive','weighted'):
                np.testing.assert_array_equal(g.run(rgb,focused,camera,variant),rgb)

    def test_sensor_formula_executed_in_production_shader(self):
        from gl_gather import Gather,SHADER
        source=SHADER.read_text().split('void main()')[0]
        source+='void main(){fragColor=vec4(vec3(signedCoCRadiusPixels(texture(uDepth,texCoord).r)),1);}'
        z=np.array([[.4,1.,1.8873722,10.]],np.float32)
        camera=dict(focal_length_mm=50.,sensor_height_mm=24.,focus_distance_m=1.8873722,f_number=1.8)
        with Gather(source) as g:
            actual=g.run(np.zeros((1,4,3),np.float32),z,camera)[...,0]
        expected=signed_coc_radius_px(z,.05,.024,1.8,1.8873722,1)
        np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-7)

    def test_large_radius_not_capped_at_32(self):
        from gl_gather import Gather
        rgb=np.zeros((256,256,3),np.float32);rgb[:,70:]=1
        depth=np.full((256,256),100.,np.float32)
        c=dict(focal_length_mm=150.,sensor_height_mm=24.,focus_distance_m=.9,f_number=1.)
        with Gather() as g:
            out=g.run(rgb,depth,c,'naive',120.)
        self.assertGreater(float(out[128,10,0]),.02) # >60px reach, impossible with old cap.

    def test_ggx_normal_peak_on_gpu(self):
        from gl_gather import Gather
        s=(ROOT/'shaders/dof_scene/scene.frag').read_text()
        a=s.index('float distributionGGX');b=s.index('float geometrySchlickGGX')
        frag='#version 330 core\nout vec4 fragColor;\nconst float kPi=3.14159265359;\n'+s[a:b]
        frag+='void main(){fragColor=vec4(vec3(distributionGGX(1.,.15)),1);}'
        c=dict(focal_length_mm=50.,sensor_height_mm=24.,focus_distance_m=2.,f_number=1.2)
        with Gather(frag) as g:actual=g.run(np.zeros((2,2,3),np.float32),np.ones((2,2),np.float32),c)
        np.testing.assert_allclose(actual,1/(math.pi*.15**4),rtol=2e-6)


class ViewerSnapshot(unittest.TestCase):
    def test_real_png_and_gather_optics(self):
        import mitsuba as mi
        mi.set_variant('scalar_rgb')
        from viewer import ViewerState,Renderer
        from PIL import Image
        state=ViewerState('home',{'teapot':[0,1,-1]});state.mode='trad';state.f_number=1.4
        with tempfile.TemporaryDirectory() as temp:
            r=object.__new__(Renderer);r.mi=mi;r.state=state;r.out_dir=Path(temp)
            r.latest_rgb=np.full((4,5,3),.4,np.float32);r.stats={'spp':4};r.frame_jpeg=b'not a PNG'
            result=r._req_snapshot({},None);directory=Path(result['saved'])
            self.assertEqual(Image.open(directory/'trad.png').format,'PNG')
            meta=json.loads((directory/'metadata.json').read_text())
            self.assertEqual(meta['camera']['f_number'],1.4)
            self.assertEqual(meta['gather_optics']['f_number'],1.4)
            self.assertEqual(meta['rendering_camera']['f_number'],1e5)


if __name__=='__main__':
    group=sys.argv.pop(1)
    selected={'sensor':SensorFootprint,'contracts':Contracts,'gpu':PortEquivalence,'viewer':ViewerSnapshot}[group]
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(selected)
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
