#version 460 core

in vec2 uv;

out vec4 color;

layout(binding=0) uniform sampler2D s_gposition;
layout(binding=1) uniform sampler2D s_gnormal;
layout(binding=2) uniform sampler2D s_galbedo;
layout(binding=3) uniform sampler2D s_gemissive;

uniform mat4 projection;

void main()
{
    // retrieve data from G-buffer
    vec3 position = texture(s_gposition, uv).rgb;
    vec3 normal = texture(s_gnormal, uv).rgb;
    vec3 albedo = texture(s_galbedo, uv).rgb;
    vec3 emissive = texture(s_gemissive, uv).rgb;

    vec3 noise = normalize(vec3(0.123, 0.534, 0.344));

    vec3 tangent   = normalize(noise - normal * dot(noise, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN       = mat3(tangent, bitangent, normal);

    vec3 dir = normalize(-position);
    float ambiant = 0.1;
    float light = clamp(max(0, dot(normal, dir)/(1.+ambiant))+ambiant, 0, 1);
    color = vec4(light*albedo, 1);
    color += vec4(emissive, 0);
}