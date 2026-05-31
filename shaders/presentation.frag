#version 460 core

in vec2 uv;

layout(binding = 0) uniform sampler2D s_albedo;
layout(binding = 1) uniform sampler2D s_bloom;

layout(location = 0) out vec4 out_color;

vec3 lerp(vec3 a, vec3 b, float t) {
    return t * b + (1 - t) * a;
}

uniform float bloom_blend_factor;
uniform float white_point_value;

vec3 linear_to_gamma(vec3 c, float clamp_point) {
    return (c * (vec3(1.) + c / pow(clamp_point, 2))) / (c + vec3(1.0));
}

void main() {
    vec4 albedo = texture(s_albedo, uv);
    vec4 bloom = texture(s_bloom, uv);

    // float gamma = 2.2;
    // float gamma = 1.;
    // float clamp_point = 4.;
    float clamp_point = white_point_value;
    vec3 gs_color = linear_to_gamma(albedo.rgb, clamp_point);
    vec3 gs_bloom = linear_to_gamma(bloom.rgb, clamp_point);
    // color = pow(color, vec3(1.0 / gamma));

    // float blend_factor = 0.04;
    // float blend_factor = 0.06;

    out_color = vec4(lerp(gs_color, bloom.rgb, bloom_blend_factor) / (1 + bloom_blend_factor * white_point_value), 1.);
    // out_color = vec4(gs_color, 1);
}
