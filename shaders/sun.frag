#version 330 core

in VS_OUT {
   vec3 position;
   vec3 normal;
   vec2 uv;
   mat3 TBN;
} vs_in;

layout(location = 0) out vec4 gAlbedo;
layout(location = 1) out vec4 gPosition;
layout(location = 2) out vec4 gNormal;
layout(location = 3) out vec4 gPbr;

uniform sampler2D tex;

void main()
{
   vec3 texture_color = texture(tex, vs_in.uv).rgb;
   vec4 color = vec4(texture_color, 1);
   gPosition = vec4(vs_in.position, 1);
   gNormal = vec4(vs_in.normal, 1);
   gAlbedo = 1.3*color;
   gPbr = vec4(0, 0, 1, 1);
   // gEmissive = vec4(texture_color, 1);
}