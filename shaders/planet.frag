#version 330 core

in VS_OUT {
   vec3 position;
   vec3 normal;
   vec2 uv;
} vs_in;

layout (location = 0) out vec4 gPosition;
layout (location = 1) out vec4 gNormal;
layout (location = 2) out vec4 gAlbedo;

uniform sampler2D tex;

void main()
{   
   gPosition = vec4(vs_in.position, 0);
   gNormal = vec4(vs_in.normal, 0);
   gAlbedo = color;
}