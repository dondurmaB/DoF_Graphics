#!/usr/bin/env python3
"""One scene -> Mitsuba sharp/depth -> actual GL gather -> qualified thin lens.

Run from the project root:
  .venv/bin/python renderer/mitsuba/stage2_experiment.py --out output/stage2/run-001
An existing output directory is refused. Any stage/qualification failure exits
nonzero; its expensive renders remain intact for inspection, never deleted.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import render as R
from experiment_contract import (digest, file_hash, seal, write_json, compare, edge_band)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def source_manifest():
    paths = [HERE / n for n in ('cafe_scene.py', 'procedural.py', 'props.py', 'render.py',
                               'stage2_experiment.py', 'gl_gather.py', 'experiment_contract.py')]
    paths += [ROOT / 'shaders/dof_scene/screen.frag']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def depth_at_centres(scene, lens, width, height):
    import mitsuba as mi
    import drjit as dr
    sensor = mi.load_dict(lens.sensor(width, height, 1, False, 'box'))
    i = dr.arange(mi.UInt32, width * height)
    uv = mi.Point2f((mi.Float(i % width) + .5) / width, (mi.Float(i // width) + .5) / height)
    ray, _ = sensor.sample_ray(0., .5, uv, mi.Point2f(.5))
    si = scene.ray_intersect(ray)
    z = dr.dot(si.p - mi.Point3f(lens.origin.tolist()), mi.Vector3f(lens.forward.tolist()))
    return np.array(dr.select(si.is_valid(), z, 1e4)).reshape(height, width).astype(np.float32)


def save_image(path, rgb, context, role, **extra):
    import mitsuba as mi
    mi.Bitmap(np.ascontiguousarray(rgb, np.float32)).write(str(path))
    record = seal(path, context, role, **extra)
    R.save_png(path.with_suffix('.png'), R.tonemap(rgb, 0.))
    seal(path.with_suffix('.png'), context, role + '-display', linear_source_sha256=record['sha256'],
         display_transform='ACES fit after exposure multiplier .8, then sRGB; 8 bit')
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=HERE / 'stage2.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--variant', default='auto')
    ap.add_argument('--compare-only', action='store_true', help='Reauthenticate a completed run; render nothing')
    args = ap.parse_args(argv)
    if args.compare_only:
        import mitsuba as mi
        mi.set_variant('scalar_rgb')
        manifests = sorted(args.out.glob('f*/context.json'))
        if not manifests:
            raise ValueError('No usable pairs: no authenticated experiment contexts')
        for p in manifests:
            context = json.loads(p.read_text())
            if context['sources'] != source_manifest():
                raise ValueError('Experiment source changed: stored results are historical, not current validation')
            compare(p.parent, context)
        return 0
    config = json.loads(args.config.read_text())
    if (config['spp'] <= 0 or config['chunk_spp'] <= 0 or len(config['stops']) != 2 or
            min(config['stops']) <= 0 or config['gather'] != {'max_radius_px':120., 'samples':100} or
            config['sampler'] != 'independent' or config['filter'] != 'box' or
            not 0 < config['noise_fraction'] <= .1):
        raise ValueError('Invalid stage-2 configuration')
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'settings.json', config)
    variant = R.pick_variant(args.variant)
    if variant == 'scalar_rgb':
        raise RuntimeError('Full experiment requires a vectorized backend; auto found no GPU/LLVM')
    import mitsuba as mi
    from cafe_scene import build_scene
    from gl_gather import Gather, SHADER
    start = time.time()
    assets = HERE / 'generated'
    scene_dict, _ = build_scene(assets)
    # The existing asset cache is explicitly hashed: stale assets cannot be
    # silently equated with a newly generated scene on another machine.
    sources = source_manifest()
    assets_hashes = {str(p.relative_to(assets)): file_hash(p) for p in sorted(assets.rglob('*')) if p.is_file()}
    scene_hash = digest(dict(recipe={k:v for k,v in sources.items() if Path(k).name in
                                    ('cafe_scene.py','procedural.py','props.py')}, assets=assets_hashes, seed=7))
    write_json(args.out / 'scene-assets.json', dict(scene_hash=scene_hash, files=assets_hashes))
    scene_dict['integrator'] = config['integrator']
    scene = mi.load_dict(scene_dict)
    view = R.VIEWS[config['view']]
    w, h = config['resolution']
    def lens(stop):
        return R.Lens(view['origin'], view['target'], config['focal_length_mm'],
                      config['sensor_height_mm'], stop, config['focus_distance_m'])
    base_lens = lens(config['stops'][0])
    depth = depth_at_centres(scene, base_lens, w, h)
    sharp = []
    for replica, seed in zip(('a','b'), config['seeds']['sharp']):
        rgb, _ = R.render_beauty(scene, base_lens.sensor(w,h,1,False,config['filter']),
                                config['spp'], config['chunk_spp'], seed, 'sharp-' + replica)
        sharp.append(rgb)
        # Save immediately: a later failure must not lose expensive renders.
        save_image(args.out / f'sharp-source-{replica}.exr', rgb,
                   dict(scene_hash=scene_hash, camera=base_lens.metadata(w,h)), 'sharp-source',
                   seed=seed, spp=config['spp'], integrator=config['integrator'], denoising=False)
    results = []
    for stop in config['stops']:
        camera = lens(stop)
        directory = args.out / f'f{stop:g}'
        directory.mkdir()
        context = dict(scene_hash=scene_hash, sources=sources, camera=camera.metadata(w,h),
                       resolution=[w,h], integrator=config['integrator'], sampler=config['sampler'],
                       filter=config['filter'], variant=variant, mitsuba=mi.__version__,
                       seed_schedule='render_beauty: seed*100003 + chunk_index',
                       chunk_spp=config['chunk_spp'], noise_fraction=config['noise_fraction'],
                       depth='first intersection at pixel centre; optical-axis metres; misses=10000m',
                       gather=dict(**config['gather'], shader_sha256=file_hash(SHADER),
                                   colour='RGB32F linear, bilinear', depth='R32F metres, nearest',
                                   target='RGBA32F, linear readback, no display transform'))
        write_json(directory / 'context.json', context)
        np.save(directory / 'depth.npy', depth)
        seal(directory / 'depth.npy', context, 'depth')
        radius = .5 * camera.lens_m**2 / (stop * camera.sensor_m) * (depth-camera.focus_m) / (depth*camera.focus_m) * h
        footprint = dict(max_visible_radius_px=float(np.max(abs(radius))),
                         infinity_radius_px=.5*context['camera']['coc_diameter_px_at_infinity'],
                         clipped_pixels=int(np.count_nonzero(abs(radius)>120)), cap_px=120, samples=100)
        write_json(directory / 'footprint.json', footprint)
        if footprint['clipped_pixels']:
            raise ValueError('Chosen experiment clips the physical footprint at radius 120')
        for replica, seed, rgb in zip(('a','b'), config['seeds']['sharp'], sharp):
            save_image(directory / f'sharp-{replica}.exr', rgb, context, f'sharp-{replica}',
                       seed=seed, spp=config['spp'], integrator=config['integrator'], denoising=False,
                       sensor='perspective')
        for replica, seed in zip(('a','b'), config['seeds']['reference']):
            rgb, _ = R.render_beauty(scene, camera.sensor(w,h,1,True,config['filter']),
                                    config['spp'], config['chunk_spp'], seed, f'f{stop:g}-reference-{replica}')
            save_image(directory / f'reference-{replica}.exr', rgb, context, f'reference-{replica}',
                       seed=seed, spp=config['spp'], integrator=config['integrator'], denoising=False,
                       sensor='thinlens')
        with Gather() as gl:
            for kind in ('naive','weighted'):
                for replica, rgb in zip(('a','b'), sharp):
                    image = gl.run(rgb, depth, context['camera'], kind, 120.)
                    save_image(directory / f'gather-{kind}-{replica}.exr', image, context,
                               f'gather-{kind}-{replica}', variant=kind, renderer=gl.renderer,
                               shader_sha256=file_hash(SHADER),
                               depth_sha256=file_hash(directory/'depth.npy'),
                               input_sha256=file_hash(directory/f'sharp-{replica}.exr'))
        result = compare(directory, context)  # raises on any unqualified row
        result['stop'] = stop
        results.append(result)
        truth = np.asarray(mi.Bitmap(str(directory/'reference-a.exr')))[...,:3]
        mask = edge_band(depth)
        R.save_png(directory/'edge-mask.png', np.repeat(mask[...,None],3,axis=2).astype(float))
        seal(directory/'edge-mask.png', context, 'edge-mask')
        for kind in ('naive','weighted'):
            candidate = np.asarray(mi.Bitmap(str(directory/f'gather-{kind}-a.exr')))[...,:3]
            error = abs(candidate-truth).astype(np.float32)
            save_image(directory/f'difference-{kind}.exr',error,context,f'difference-{kind}')
            # Explicit common linear scale for qualitative heat maps, no auto-normalization.
            R.save_png(directory/f'difference-{kind}-linear-x4.png',np.clip(error*4,0,1))
            seal(directory/f'difference-{kind}-linear-x4.png',context,f'difference-{kind}-linear-x4')
    # Independent wide-angle inspection view; explicitly not a qualified reference.
    overview = config['overview']; v = R.VIEWS[overview['view']]
    camera = R.Lens(v['origin'],v['target'],overview['focal_length_mm'],24.,16.,config['focus_distance_m'])
    ow,oh = overview['resolution']
    rgb,_ = R.render_beauty(scene,camera.sensor(ow,oh,1,False,'box'),overview['spp'],
                            config['chunk_spp'],config['seeds']['overview'],'wide-angle inspection')
    save_image(args.out/'wide-angle-inspection.exr',rgb,
               dict(scene_hash=scene_hash,camera=camera.metadata(ow,oh)), 'inspection-only',
               qualified_reference=False,seed=config['seeds']['overview'],spp=overview['spp'],
               integrator=config['integrator'],denoising=False)
    write_json(args.out/'results.json',dict(settings=config, results=results,seconds=time.time()-start))
    print(f'QUALIFIED experiment complete: {args.out}',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
