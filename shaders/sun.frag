#version 330 core

in vec3 position;
in vec2 uv;
in vec3 normal;

uniform sampler2D tex;

out vec4 color;

void main()
{
   vec3 texCol = texture(tex, uv).rgb;
   color = vec4(texCol, 1);
}