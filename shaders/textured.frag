#version 330 core
#extension GL_ARB_explicit_uniform_location : require

in VS_OUT {
   vec3 position;
   vec3 normal;
   vec2 uv;
   mat3 TBN;
} vs_in;

layout(location = 0) out vec4 gAlbedo;
layout(location = 1) out vec4 gPosition;
layout(location = 2) out vec4 gNormal;
layout(location = 3) out vec4 gPbr;

uniform sampler2D s_albedo;
uniform sampler2D s_normal;
uniform sampler2D s_specular;
uniform sampler2D s_emissive;

uniform vec3 tint;

vec4 lerp(vec4 a, vec4 b, float t) {
   return b * t + a * (1. - t);
}

void main() {
   vec4 albedo = texture(s_albedo, vs_in.uv);
   vec3 normal = texture(s_normal, vs_in.uv).rgb;
   float specular = texture(s_specular, vs_in.uv).r;
   vec4 emissive = texture(s_emissive, vs_in.uv);
   gPosition = vec4(vs_in.position, 1);
   vec3 combined_normal = normal * 2.0 - 1.0;
   combined_normal = normalize(vs_in.TBN * combined_normal);

   vec3 sun_direction = normalize(-vs_in.position);

   float emissivness = max(0., dot(combined_normal, -sun_direction)) * emissive.r;
   gNormal = vec4(combined_normal, 1);
   // gAlbedo = vec4(1, 0, 0, 1);
   // gAlbedo = albedo;
   gAlbedo = lerp(albedo, emissive, emissivness);
   // metallic + roughness + emissiveness
   gPbr = vec4(0, 0.3 + (1 - specular) * (1 - 0.3), emissivness, 1);
   // gEmissive = vec4(vec3(0), 1);
}