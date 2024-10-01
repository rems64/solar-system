#version 330 core

in vec3 position;
in vec2 uv;
in vec3 normal;

uniform sampler2D tex;

out vec4 color;

const vec3 light_position = vec3(0, 0, 1.5);

void main()
{
   vec3 dir = normalize(light_position - position);
   float ambiant = 0.1;
   vec3 texCol = texture(tex, uv).rgb;
   color = vec4(texCol, 1);
   // color = vec4(vec3(clamp(max(0, dot(normal, dir)/(1.+ambiant))+ambiant, 0, 1)), 1);
}