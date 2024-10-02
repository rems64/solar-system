#version 460 core

layout (location = 0) in vec2 in_position;

out vec2 uv;

void main()
{
   gl_Position = vec4(in_position, 1.0, 1.0);
   uv = (in_position+1.0)/2.0;
}