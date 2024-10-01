#version 330 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec2 in_uv;
layout (location = 2) in vec3 in_normal;

out vec3 position;
out vec2 uv;
out vec3 normal;

uniform mat4 model;
uniform mat4 vp;

void main()
{
   vec4 pos = model * vec4(in_position, 1);
   position = pos.xyz/pos.w;
   gl_Position = vp*pos;
   uv = in_uv;
   normal = in_normal;
}