#version 330 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;

out VS_OUT {
   vec3 position;
   vec3 normal;
   vec2 uv;
} vs_out;

uniform mat4 local_model;
uniform mat4 model;
uniform mat4 vp;

void main()
{
   vec4 position = model * vec4(in_position, 1);
   gl_Position = vp * position;

   vs_out.position = position.xyz/position.w;
   mat3 normal_matrix = transpose(inverse(mat3(model)));
   vs_out.normal = normal_matrix * in_normal;
   vs_out.uv = in_uv;
}