#version 330 core

in vec3 interpolatedColor;

out vec4 fragColor;

void main()
{
    fragColor = vec4(interpolatedColor, 1.0);
}
