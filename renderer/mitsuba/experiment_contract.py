"""Authenticated artifacts and linear-domain comparison. No filename-only pairing."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def seal(path, context, role, **extra):
    record = dict(schema=1, file=Path(path).name, sha256=file_hash(path),
                  scene_hash=context['scene_hash'], camera_hash=digest(context['camera']),
                  context_hash=digest(context), context=context, role=role, **extra)
    write_json(str(path) + '.json', record)
    return record


def authenticate(path, context, role=None):
    path = Path(path)
    meta = json.loads(Path(str(path) + '.json').read_text())
    if (meta.get('schema') != 1 or meta.get('file') != path.name or
            meta.get('sha256') != file_hash(path) or
            meta.get('context_hash') != digest(context) or meta.get('context') != context or
            meta.get('scene_hash') != context['scene_hash'] or
            meta.get('camera_hash') != digest(context['camera'])):
        raise ValueError(f'Stale or mismatched artifact: {path}')
    if role is not None and meta.get('role') != role:
        raise ValueError(f'Wrong artifact role: {path}')
    return meta


def edge_band(depth):
    d = np.asarray(depth, np.float64)
    if not np.isfinite(d).all() or (d <= 0).any():
        raise ValueError('Depth must be finite positive planar metres')
    log = np.log(d)
    mask = np.zeros(d.shape, bool)
    dx, dy = abs(np.diff(log, axis=1)) > math.log(1.3), abs(np.diff(log, axis=0)) > math.log(1.3)
    mask[:, 1:] |= dx
    mask[:, :-1] |= dx
    mask[1:, :] |= dy
    mask[:-1, :] |= dy
    mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(15))) > 0
    if not mask.any():
        raise ValueError('No depth discontinuities: edge benchmark would be empty')
    return mask


def metrics(a, b, mask=None):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError('RGB dimensions do not match')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite radiance')
    delta = a - b
    if mask is not None:
        if mask.shape != a.shape[:2] or not mask.any():
            raise ValueError('Invalid or empty metric mask')
        delta = delta[mask]
    rmse = float(np.sqrt(np.mean(delta * delta)))
    return dict(mae=float(np.abs(delta).mean()), rmse=rmse,
                psnr_db=None if rmse == 0 else float(-20 * math.log10(rmse)),
                psnr_peak_linear=1.0)


def noise(a, b, mask=None):
    """Equal-N independent estimates: difference RMS / sqrt(2) for estimate A.

    Also report centred difference standard deviation / sqrt(2). The RMS
    includes any observed mean offset and is the more conservative gate.
    This is an empirical spatial/channel aggregate, not a per-pixel bound or
    a 95% confidence interval. Reference A is used, NOT the average of A/B.
    """
    delta = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    if mask is not None:
        delta = delta[mask]
    if not np.isfinite(delta).all() or delta.size == 0:
        raise ValueError('Invalid noise pair')
    return dict(rmse_estimate=float(np.sqrt(np.mean(delta**2) / 2)),
                std_estimate=float(np.std(delta) / math.sqrt(2)),
                difference_mean=float(delta.mean()),
                difference_mae=float(np.abs(delta).mean()),
                method='equal-N independent A,B; reference=A; difference/sqrt(2)')


def compare(directory, context):
    """Refuse missing/stale/unqualified inputs. Recompute all qualification numbers."""
    import mitsuba as mi
    d = Path(directory)
    arrays, records = {}, {}
    for name in ('sharp-a', 'sharp-b', 'reference-a', 'reference-b',
                 'gather-naive-a', 'gather-naive-b', 'gather-weighted-a', 'gather-weighted-b'):
        p = d / (name + '.exr')
        records[name] = authenticate(p, context, name)
        bitmap = mi.Bitmap(str(p))
        if str(bitmap.component_format()) != 'Type.Float32':
            raise ValueError(f'Expected float32 EXR: {p}')
        arrays[name] = np.asarray(bitmap, np.float32)[..., :3]
    authenticate(d / 'depth.npy', context, 'depth')
    depth = np.load(d / 'depth.npy', allow_pickle=False)
    expected_shape = tuple(reversed(context['resolution'])) + (3,)
    if depth.shape != expected_shape[:2] or any(a.shape != expected_shape for a in arrays.values()):
        raise ValueError('Artifact dimensions disagree with camera resolution')
    for prefix in ('sharp', 'reference'):
        a, b = records[prefix + '-a'], records[prefix + '-b']
        if (a.get('seed') == b.get('seed') or not isinstance(a.get('seed'), int) or
                not isinstance(b.get('seed'), int) or a.get('spp', 0) <= 0 or a.get('spp') != b.get('spp') or
                a.get('denoising') is not False or b.get('denoising') is not False or
                a.get('integrator') != context['integrator'] or b.get('integrator') != context['integrator']):
            raise ValueError('Missing equal-spp independent-seed undenoised reference qualification')
    for variant in ('naive', 'weighted'):
        for replica in ('a', 'b'):
            record = records[f'gather-{variant}-{replica}']
            if (record.get('input_sha256') != records[f'sharp-{replica}']['sha256'] or
                    record.get('depth_sha256') != file_hash(d / 'depth.npy') or
                    record.get('shader_sha256') != context['gather']['shader_sha256'] or
                    record.get('variant') != variant):
                raise ValueError('Gather provenance mismatch')
    mask = edge_band(depth)
    ref_a, ref_b = arrays['reference-a'], arrays['reference-b']
    rows, qualified = [], True
    for region, selection in (('whole', None), ('depth_edges', mask)):
        rn = noise(ref_a, ref_b, selection)
        for variant in ('naive', 'weighted'):
            a, b = arrays[f'gather-{variant}-a'], arrays[f'gather-{variant}-b']
            result = metrics(a, ref_a, selection)
            gn = noise(a, b, selection)
            combined = math.hypot(rn['rmse_estimate'], gn['rmse_estimate'])
            passes = combined <= context['noise_fraction'] * result['mae']
            qualified &= passes
            rows.append(dict(variant=variant, region=region, **result,
                             reference_noise=rn, propagated_input_noise=gn,
                             combined_noise_rmse=combined,
                             noise_to_mae=combined / max(result['mae'], 1e-30), qualified=passes))
    result = dict(schema=1, context_hash=digest(context), qualified=bool(qualified),
                  colour_space='scene-linear RGB radiance; no tone mapping or clipping',
                  reference='reference-a.exr; independent equal-N reference-b.exr estimates noise',
                  criterion='hypot(reference noise, propagated sharp noise) <= noise_fraction * MAE in every row',
                  edge_definition='adjacent planar depth ratio >1.3, marked on both sides, dilated by 7 pixels',
                  edge_fraction=float(mask.mean()), rows=rows,
                  input_hashes={k: v['sha256'] for k, v in records.items()})
    # Failed qualification remains evidence, but never masquerades as results.
    write_json(d / 'qualification.json', result)
    if not qualified:
        raise ValueError('Reference/input noise is not sufficiently below measured gather error')
    # Attach the measured qualification to each reference's own sidecar too.
    # It is recomputed, never trusted as a free-standing boolean on recompare.
    for replica in ('a', 'b'):
        p = d / f'reference-{replica}.exr'
        record = records[f'reference-{replica}']
        record['qualification'] = dict(
            report_sha256=file_hash(d / 'qualification.json'),
            independent_pair_sha256=records['reference-b' if replica=='a' else 'reference-a']['sha256'],
            whole=noise(ref_a,ref_b), depth_edges=noise(ref_a,ref_b,mask),
            scope='empirical equal-N noise; not a systematic-bias or worst-pixel bound')
        write_json(str(p) + '.json', record)
    write_json(d / 'results.json', result)
    return result
