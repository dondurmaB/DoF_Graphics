"""Controlled Lambert + single-scatter GGX material shared by render and numerical probe.

rough is perceptual roughness (alpha=rough^2), spec is scalar F0.
White Edge Tint on Cycles F82 makes its correction B exactly zero: Schlick.
Diffuse and ambient use albedo*(1-F0). F0=0 disables the entire specular lobe.
"""


def surface_material(bpy, name, sky=(0, 0, 0), *, albedo=None, rough=0.5, spec=0.0,
                     full_gi=False):
    if not hasattr(bpy.types, "ShaderNodeBsdfMetallic"):
        raise RuntimeError("Matched materials require the Metallic BSDF node; verified with Blender 5.2.2.")
    material = bpy.data.materials.new(name)
    if bpy.app.version < (5, 0, 0):
        material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    if albedo is None:
        color = nodes.new('ShaderNodeVertexColor')
        color.layer_name = 'Col'
        color_out = color.outputs['Color']
        inputs = []
        for name in ('Roughness', 'Specular'):
            attribute = nodes.new('ShaderNodeAttribute')
            attribute.attribute_name = name
            inputs.append(attribute.outputs['Fac'])
        rough_out, spec_out = inputs
    else:
        color = nodes.new('ShaderNodeRGB')
        color.outputs[0].default_value = (*albedo, 1.0)
        color_out = color.outputs[0]
        rough_node, spec_node = nodes.new('ShaderNodeValue'), nodes.new('ShaderNodeValue')
        rough_node.outputs[0].default_value, spec_node.outputs[0].default_value = rough, spec
        rough_out, spec_out = rough_node.outputs[0], spec_node.outputs[0]
    remaining = nodes.new('ShaderNodeMath')
    remaining.operation = 'SUBTRACT'
    remaining.inputs[0].default_value = 1.0
    links.new(spec_out, remaining.inputs[1])
    diffuse_color = nodes.new('ShaderNodeVectorMath')
    diffuse_color.operation = 'SCALE'
    links.new(color_out, diffuse_color.inputs[0])
    links.new(remaining.outputs[0], diffuse_color.inputs['Scale'])
    diffuse = nodes.new('ShaderNodeBsdfDiffuse')
    diffuse.inputs['Roughness'].default_value = 0.0
    links.new(diffuse_color.outputs['Vector'], diffuse.inputs['Color'])
    # A Fresnel/Layer Weight Mix Shader uses N.V, not V.H, so is NOT equivalent.
    metal = nodes.new('ShaderNodeBsdfMetallic')
    metal.distribution, metal.fresnel_type = 'GGX', 'F82'
    metal.inputs['Edge Tint'].default_value = (1, 1, 1, 1)
    metal.inputs['Anisotropy'].default_value = 0.0
    metal.inputs['Thin Film Thickness'].default_value = 0.0
    links.new(spec_out, metal.inputs['Base Color'])
    links.new(rough_out, metal.inputs['Roughness'])
    enabled = nodes.new('ShaderNodeMath')
    enabled.operation = 'GREATER_THAN'
    links.new(spec_out, enabled.inputs[0])
    enabled.inputs[1].default_value = 0.0
    gate = nodes.new('ShaderNodeMixShader')  # Unconnected first shader is zero, not transparency.
    links.new(enabled.outputs[0], gate.inputs[0])
    links.new(metal.outputs[0], gate.inputs[2])
    combine = nodes.new('ShaderNodeAddShader')
    links.new(diffuse.outputs[0], combine.inputs[0])
    links.new(gate.outputs[0], combine.inputs[1])
    surface = combine.outputs[0]
    if not full_gi:
        ambient = nodes.new('ShaderNodeVectorMath')
        ambient.operation = 'MULTIPLY'
        links.new(diffuse_color.outputs['Vector'], ambient.inputs[0])
        ambient.inputs[1].default_value = sky
        fill = nodes.new('ShaderNodeEmission')
        links.new(ambient.outputs['Vector'], fill.inputs['Color'])
        fill.inputs['Strength'].default_value = 1.0
        plus_fill = nodes.new('ShaderNodeAddShader')
        links.new(surface, plus_fill.inputs[0])
        links.new(fill.outputs[0], plus_fill.inputs[1])
        surface = plus_fill.outputs[0]
    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(surface, output.inputs['Surface'])
    return material
