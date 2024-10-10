#version 460 core

in vec2 uv;

layout(binding = 0) uniform sampler2D s_albedo;
layout(binding = 1) uniform sampler2D s_position;
layout(binding = 2) uniform sampler2D s_normal;
layout(binding = 3) uniform sampler2D s_pbr;
layout(binding = 4) uniform sampler2D s_stars;

layout(location = 0) out vec4 out_albedo;
layout(location = 1) out vec4 out_position;
layout(location = 2) out vec4 out_normal;
layout(location = 3) out vec4 out_pbr;

uniform mat4 view_projection;

const float PI = 3.1415;

vec3 starness(float theta, float phi) {
    vec2 coord = vec2((phi + PI) / (2 * PI), theta / PI);
    // color = color / (color + vec3(1.0));
    // color = pow(color, vec3(1.0 / 2.2));
    vec3 color = 3 * texture(s_stars, coord).rgb;
    // return pow(color, vec3(2.2)) * (color + vec3(1.0));
    vec3 c = pow(color, vec3(2.2));
    return c / (vec3(1) - c);
}

void main() {
    // retrieve data from G-buffer
    vec4 albedo = texture(s_albedo, uv);
    vec4 position = texture(s_position, uv);
    vec4 normal = texture(s_normal, uv);
    vec4 pbr = texture(s_pbr, uv);

    vec4 _ray = inverse(view_projection) * vec4(1 * (2 * uv - 1), 1, 1);
    vec3 ray = normalize(_ray.xyz / _ray.w);
    float phi = atan(ray.y, ray.x);
    float theta = acos(ray.z);

    vec3 background = starness(theta, phi);

    vec3 color = background * (1. - albedo.a) + albedo.rgb * (albedo.a);
    out_albedo = vec4(color, 1.);
    out_position = position;
    out_normal = normal;
    out_pbr = pbr;
}