#version 460 core

out vec4 color;

layout(binding = 0) uniform sampler2D s_galbedo;
layout(binding = 1) uniform sampler2D s_gposition;
layout(binding = 2) uniform sampler2D s_gnormal;
layout(binding = 3) uniform sampler2D s_gpbr;

uniform mat4 view_projection;
uniform vec2 viewport_size;
const float PI = 3.1415;

void main() {
    // retrieve data from G-buffer
    vec2 uv = vec2(gl_FragCoord.x,  gl_FragCoord.y) - 0.5;
    uv /= viewport_size;
    vec3 albedo = texture(s_galbedo, uv).rgb;
    vec3 position = texture(s_gposition, uv).rgb;
    vec3 normal = texture(s_gnormal, uv).rgb;
    vec4 pbr = texture(s_gpbr, uv).rgba;

    color = vec4(albedo, 1);
}