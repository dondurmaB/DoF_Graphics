"""Controlled opaque/direct-light material variant of the existing cafe.

Geometry and transforms are retained exactly. This is not the full-GI/glass
cafe: every surface is diffuse and opaque, local emitters/environment are
disabled, one fixed point light at the benchmark camera remains, and the path depth is explicitly 2.
The purpose is aperture visibility, not validating light transport or materials.
"""
from cafe_scene import build_scene


def build_controlled_scene(assets):
    scene, targets = build_scene(assets)
    bsdf_types = {'principled','diffuse','roughconductor','conductor','dielectric',
                  'roughdielectric','twosided'}
    for name, value in list(scene.items()):
        if not isinstance(value, dict):
            continue
        if value.get('type') in bsdf_types:
            original = value.get('bsdf', value) if value['type'] == 'twosided' else value
            albedo = original.get('base_color', original.get('reflectance',
                                 {'type':'rgb','value':[.45,.45,.45]}))
            diffuse = {'type':'diffuse','reflectance':albedo}
            scene[name] = {'type':'twosided','bsdf':diffuse} if value['type']=='twosided' else diffuse
        elif 'emitter' in value:
            # Keep the lamp geometry; it no longer emits in this experiment.
            value.pop('emitter')
            value.setdefault('bsdf', {'type':'ref','id':'emitter_backing'})
    scene.pop('sky')
    # Fixed in world space, including during the wide-angle inspection view.
    scene['controlled_flash'] = {'type':'point','position':[.34,1.22,1.85],
                                'intensity':{'type':'rgb','value':[15.,15.,15.]}}
    return scene, targets
