#version 330 core

in vec3 position;
in vec2 uv;
in vec3 normal;

uniform sampler2D tex;

out vec4 color;

void main()
{
    vec3 dir = normalize(-position);
    float ambiant = 0.3;
    vec4 texture_color = texture(tex, uv).rgbb;
    float light = clamp(max(0, dot(normal, dir)/(1.+ambiant))+ambiant, 0, 1);
    color = light*texture_color;
}