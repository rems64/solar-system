#version 330 core

in VS_OUT {
   vec3 position;
   vec3 normal;
   vec2 uv;
} vs_in;

layout (location = 0) out vec3 gPosition;
layout (location = 1) out vec3 gNormal;
layout (location = 2) out vec4 gAlbedo;

uniform sampler2D tex;

void main()
{
    vec3 dir = normalize(-vs_in.position);
    float ambiant = 0.3;
    vec4 texture_color = texture(tex, vs_in.uv).rgbb;
    float light = clamp(max(0, dot(vs_in.normal, dir)/(1.+ambiant))+ambiant, 0, 1);
    vec4 color = light*texture_color;

    gPosition = vs_in.position;
    gNormal = vs_in.normal;
    gAlbedo = color;
}