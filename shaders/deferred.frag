#version 460 core

in vec2 uv;

out vec4 out_color;

layout(binding = 0) uniform sampler2D s_galbedo;
layout(binding = 1) uniform sampler2D s_gposition;
layout(binding = 2) uniform sampler2D s_gnormal;
layout(binding = 3) uniform sampler2D s_gpbr;

layout(binding = 4) uniform sampler2D s_texture;
layout(binding = 5) uniform sampler2D s_atmosphere;

uniform mat4 view_projection;

uniform vec3 camera_position;

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
    vec2 coord = vec2((phi + PI) / (2 * PI), theta / PI);
    vec2 coord_scaled = vec2(coord.x * cos((coord.y - 0.5) * PI), coord.y);
    // return vec3(clamp(pow(noise(1000. * coord_scaled), 200.0) * 100.0, 0, 1));
    return 3 * texture(s_texture, coord).rgb;
}

float luminosity(vec3 color) {
    return 0.21 * color.r + 0.71 * color.g + 0.07 * color.b;
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

const vec3 sun_position = vec3(0.);
const vec3 sun_color = 10. * vec3(1.);

void main() {
    // retrieve data from G-buffer
    vec3 albedo = texture(s_galbedo, uv).rgb;
    float foreground_mask = texture(s_galbedo, uv).a;
    vec3 position = texture(s_gposition, uv).rgb;
    vec3 normal = texture(s_gnormal, uv).rgb;
    vec4 pbr = texture(s_gpbr, uv).rgba;
    float metallic = pbr.r;
    float roughness = pbr.g;
    float emissiveness = pbr.b;
    float ao = pbr.a;

    vec3 atmosphere = texture(s_atmosphere, uv).rgb;

    vec4 _ray = inverse(view_projection) * vec4(1 * (2 * uv - 1), 1, 1);
    vec3 ray = normalize(_ray.xyz / _ray.w);
    float phi = atan(ray.y, ray.x);
    float theta = acos(ray.z);

    vec3 background = starness(theta, phi);

    vec3 V = normalize(camera_position - position);

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    // reflectance equation
    vec3 Lo = vec3(0.0);
    vec3 L = normalize(sun_position - position);
    vec3 H = normalize(V + L);
    float distance = length(sun_position - position);
    float attenuation = 1.0 / (distance * distance);
    vec3 radiance = sun_color * attenuation;        

    // cook-torrance brdf
    float NDF = DistributionGGX(normal, H, roughness);
    float G = GeometrySmith(normal, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(normal, V), 0.0) * max(dot(normal, L), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;  

    // add to outgoing radiance Lo
    float NdotL = max(dot(normal, L), 0.0);
    Lo += (kD * albedo / PI + specular) * radiance * NdotL;

    // vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 ambient = vec3(0.0);
    vec3 color = emissiveness > 0.5 ? albedo : (ambient + Lo);

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
    color = foreground_mask > 0.5 ? color : background;

    out_color = vec4(color, 1.0);
}