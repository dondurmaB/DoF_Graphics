#version 330 core

in vec3 interpolatedColor;

out vec4 fragColor;

uniform float uIntensity;

void main()
{
    fragColor = vec4(interpolatedColor * uIntensity, 1.0);
}
