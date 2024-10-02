#version 330 core

in VS_OUT {
   vec3 position;
   vec3 normal;
   vec2 uv;
} vs_in;

layout (location = 0) out vec4 gPosition;
layout (location = 1) out vec4 gNormal;
layout (location = 2) out vec4 gAlbedo;
layout (location = 3) out vec4 gEmissive;

uniform sampler2D tex;

void main()
{
   vec4 texture_color = texture(tex, vs_in.uv).rgbr;
   vec4 color = texture_color;
   gPosition = vec4(vs_in.position, 1);
   gNormal = vec4(vs_in.normal, 1);
   gAlbedo = color;
   gEmissive = vec4(0);
}