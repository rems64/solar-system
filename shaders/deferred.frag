#version 460 core

in vec2 uv;

out vec4 color;

layout(binding = 0) uniform sampler2D s_gposition;
layout(binding = 1) uniform sampler2D s_gnormal;
layout(binding = 2) uniform sampler2D s_galbedo;
layout(binding = 3) uniform sampler2D s_gemissive;
layout(binding = 4) uniform sampler2D s_texture;

uniform mat4 view_projection;
const float PI = 3.1415;

float random(vec2 ab) {
    float f = (cos(dot(ab, vec2(21.9898, 78.233))) * 43758.5453);
    return fract(f);
}

float noise(vec2 xy) {
    vec2 ij = floor(xy);
    vec2 uv = xy - ij;
    uv = uv * uv * (3.0 - 2.0 * uv);

    float a = random(vec2(ij.x, ij.y));
    float b = random(vec2(ij.x + 1., ij.y));
    float c = random(vec2(ij.x, ij.y + 1.));
    float d = random(vec2(ij.x + 1., ij.y + 1.));
    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k3 = a - b - c + d;
    return (k0 + k1 * uv.x + k2 * uv.y + k3 * uv.x * uv.y);
}

vec3 starness(float theta, float phi) {
    vec2 coord = vec2((phi+PI)/(2*PI), theta/PI);
    vec2 coord_scaled = vec2(coord.x*cos((coord.y-0.5)*PI), coord.y);
    // return vec3(clamp(pow(noise(1000. * coord_scaled), 200.0) * 100.0, 0, 1));
    return 3 * texture(s_texture, coord).rgb;
}

void main() {
    // retrieve data from G-buffer
    vec3 position = texture(s_gposition, uv).rgb;
    vec3 normal = texture(s_gnormal, uv).rgb;
    vec3 albedo = texture(s_galbedo, uv).rgb;
    vec3 emissive = texture(s_gemissive, uv).rgb;
    float v = clamp(texture(s_galbedo, uv).a, 0, 1);

    vec3 noise = normalize(vec3(0.123, 0.534, 0.344));

    vec3 tangent = normalize(noise - normal * dot(noise, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);

    vec3 dir = normalize(-position);
    float ambiant = 0.1;
    float light = clamp(max(0, dot(normal, dir) / (1. + ambiant)) + ambiant, 0, 1);
    vec3 foreground = light * albedo;
    foreground += emissive;

    vec4 _ray = inverse(view_projection) * vec4(1*(2*uv-1), 1, 1);
    vec3 ray = normalize(_ray.xyz / _ray.w);
    float phi = atan(ray.y, ray.x);
    float theta = acos(ray.z);

    vec4 background = vec4(starness(theta, phi), 1);

    v = clamp(v, 0, 1);
    color = background * (1 - v) + vec4(foreground, 1) * v;
}