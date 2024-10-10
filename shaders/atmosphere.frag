#version 460 core

#define MAX_FLOAT 99999999
#define PI 3.141592
#define num_inscatter_points 10
#define num_optical_depth_points 10
#define density_falloff 4.

out vec4 out_color;

layout(binding = 0) uniform sampler2D s_galbedo;
layout(binding = 1) uniform sampler2D s_gposition;
layout(binding = 2) uniform sampler2D s_gnormal;
layout(binding = 3) uniform sampler2D s_gpbr;

uniform mat4 inv_vp;
uniform vec2 viewport_size;

uniform vec3 camera_position;
uniform vec3 atmosphere_center;
uniform float atmosphere_radius;

uniform float planet_radius;

uniform vec3 scattering_coefficients;

vec2 ray_sphere(vec3 center, float radius, vec3 ray_origin, vec3 ray_direction) {
    // ray_direction has to be normalized
    vec3 offset = ray_origin - center;
    float a = 1;
    float b = 2 * dot(offset, ray_direction);
    float c = dot(offset, offset) - radius * radius;
    float d = b * b - 4 * a * c;

    if(d > 0) {
        float s = sqrt(d);
        float dist_to_sphere_near = max(0, (-b - s) / (2 * a));
        float dist_to_sphere_far = (-b + s) / (2 * a);

        if(dist_to_sphere_far >= 0) {
            return vec2(dist_to_sphere_near, dist_to_sphere_far - dist_to_sphere_near);
        }
    }
    return vec2(MAX_FLOAT, 0);
}

float density_at_point(vec3 point) {
    float height_above_surface = length(point - atmosphere_center) - planet_radius;
    float height01 = height_above_surface / (atmosphere_radius - planet_radius);
    float local_density = exp(-height01 * density_falloff) * (1 - height01);
    return local_density;
}

float optical_depth(vec3 ray_origin, vec3 ray_direction, float ray_length) {
    vec3 density_sample_point = ray_origin;
    float step_size = ray_length / (num_optical_depth_points - 1);
    float optical_depth = 0;

    for(int i = 0; i < num_optical_depth_points; i++) {
        float local_density = density_at_point(density_sample_point);
        optical_depth += local_density * step_size;
        density_sample_point += ray_direction * step_size;
    }

    return optical_depth;
}

vec3 calculate_light(vec3 ray_origin, vec3 ray_direction, float ray_length, vec3 original_color) {
    vec3 inscatter_point = ray_origin;
    float step_size = ray_length / (num_inscatter_points - 1);
    vec3 inscatter_light = vec3(0);
    float view_ray_optical_depth = 0.;

    for(int i = 0; i < num_inscatter_points; i++) {
        vec3 dir_to_sun = normalize(-inscatter_point);
        float sun_ray_length = ray_sphere(atmosphere_center, atmosphere_radius, inscatter_point, dir_to_sun).y;
        float sun_ray_optical_depth = optical_depth(inscatter_point, dir_to_sun, sun_ray_length);
        view_ray_optical_depth = optical_depth(inscatter_point, -ray_direction, step_size * i);
        vec3 transmittance = exp(-(sun_ray_optical_depth + view_ray_optical_depth) * scattering_coefficients);
        float local_density = density_at_point(inscatter_point);

        inscatter_light += local_density * transmittance * scattering_coefficients * step_size;
        inscatter_point += ray_direction * step_size;
    }
    float original_color_transmittance = exp(-view_ray_optical_depth);
    return original_color * original_color_transmittance + inscatter_light;
}

void main() {
    // retrieve data from G-buffer
    // warning: no -0.5 here, despite what's written on the internet :sad_face:
    vec2 uv = vec2(gl_FragCoord.x, gl_FragCoord.y);
    uv /= viewport_size;
    vec4 albedo = texture(s_galbedo, uv);
    vec3 position = texture(s_gposition, uv).rgb;
    vec3 normal = texture(s_gnormal, uv).rgb;
    vec4 pbr = texture(s_gpbr, uv).rgba;

    vec4 _ray = inv_vp * vec4((2 * uv - 1), 1, 1);
    vec3 ray_direction = normalize(_ray.xyz / _ray.w - camera_position);

    float distance_to_sphere = length(position - camera_position);

    vec2 hit_info = ray_sphere(atmosphere_center, atmosphere_radius, camera_position, ray_direction);
    float dist_to_atmosphere = hit_info.x;

    // Prevents (0, 0, 0) of the sky to be used as position
    float distance_through_atmosphere_to_sphere = length(position) > 0 ? distance_to_sphere - dist_to_atmosphere : MAX_FLOAT;
    float dist_through_atmosphere = min(hit_info.y, distance_through_atmosphere_to_sphere);

    vec3 color;

    if(dist_through_atmosphere > 0) {
        const float epsilon = 0.001;
        vec3 point_in_atmosphere = camera_position + ray_direction * (dist_to_atmosphere + epsilon);
        vec3 light = calculate_light(point_in_atmosphere, ray_direction, dist_through_atmosphere - 2 * epsilon, albedo.rgb);
        color = light;
    } else {
        color = albedo.rgb;
        // color = vec4(1, 0, 1, 1);
    }

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
    out_color = vec4(color, 1);
}