#version 450

layout(location = 0) in vec3 frag_world_position;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec2 frag_uv;

layout(location = 0) out vec4 out_color;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    vec3 directional_light_direction;
    vec3 directional_light_color;
    vec3 ambient_light;
    uint point_light_count;
    uint spot_light_count;
} scene_uniforms;

struct PointLight {
    vec3 position;
    vec3 color;
};

struct SpotLight {
    vec3 position;
    vec3 color;
    vec3 direction;
    float inner_cos;
    float outer_cos;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

layout(std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    mat4 normal_matrix;
    vec3 albedo;
    vec3 specular;
    float shininess;
    uint unlit;
} model_uniforms;

layout(set = 1, binding = 1) uniform sampler2D albedo_texture;
layout(set = 1, binding = 2) uniform sampler2D specular_texture;
layout(set = 1, binding = 3) uniform sampler2D emissive_texture;

layout(push_constant) uniform PushConstants {
    uint effect;
    float time;
    float strength;
    float padding;
} push;

const uint EFFECT_NONE = 0u;
const uint EFFECT_WOBBLE = 1u;
const uint EFFECT_SWIRL = 2u;
const uint EFFECT_WIDE_ANGLE = 3u;
const uint EFFECT_KALEIDOSCOPE = 4u;
const uint EFFECT_FROSTED_GLASS = 5u;
const uint EFFECT_ANAGLYPH = 6u;

const float PI = 3.14159265359;
const int FROSTED_GLASS_SAMPLES = 12;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec4 sampleTexture(sampler2D tex, vec2 uv) {
    float t = push.time;
    float s = push.strength;

    switch (push.effect) {
    case EFFECT_WOBBLE: {
        vec2 offset = 0.02 * s * vec2(sin(uv.y * 25.0 + t * 3.0), cos(uv.x * 25.0 + t * 2.5));
        vec4 texel = texture(tex, uv + offset);
        texel.rgb *= 0.75 + 0.25 * sin(t * 2.0 + (uv.x + uv.y) * 30.0);
        return texel;
    }
    case EFFECT_SWIRL: {
        vec2 d = uv - 0.5;
        float angle = 4.0 * s * sin(t * 0.7) * smoothstep(0.7, 0.0, length(d));
        float c = cos(angle);
        float sn = sin(angle);
        return texture(tex, 0.5 + vec2(c * d.x - sn * d.y, sn * d.x + c * d.y));
    }
    case EFFECT_WIDE_ANGLE: {
        vec2 d = (uv - 0.5) * 2.0;
        float k = s * (0.6 + 0.4 * sin(t * 0.8));
        return texture(tex, 0.5 + 0.5 * d * (1.0 + 1.5 * k * dot(d, d)));
    }
    case EFFECT_KALEIDOSCOPE: {
        vec2 d = uv - 0.5;
        float segment = 2.0 * PI / 6.0;
        float angle = mod(atan(d.y, d.x + 1e-6) + t * 0.25, segment);
        angle = abs(angle - 0.5 * segment);
        return texture(tex, 0.5 + length(d) * s * vec2(cos(angle), sin(angle)));
    }
    case EFFECT_FROSTED_GLASS: {
        float radius = 0.02 * s * (1.0 + 0.5 * sin(t));
        float frame = floor(t * 10.0);
        vec4 sum = vec4(0.0);
        for (int i = 0; i < FROSTED_GLASS_SAMPLES; ++i) {
            vec2 seed = gl_FragCoord.xy + float(i) * 17.0 + frame * 5.0;
            vec2 jitter = vec2(hash12(seed), hash12(seed.yx + 31.0)) - 0.5;
            sum += texture(tex, uv + 2.0 * radius * jitter);
        }
        return sum / float(FROSTED_GLASS_SAMPLES);
    }
    case EFFECT_ANAGLYPH: {
        float shift = 0.012 * s * (1.0 + sin(t * 1.5));
        vec4 left = texture(tex, uv - vec2(shift, 0.0));
        vec4 right = texture(tex, uv + vec2(shift, 0.0));
        return left * vec4(1.0, 0.0, 0.0, 1.0) + right * vec4(0.0, 1.0, 1.0, 0.0);
    }
    }

    return texture(tex, uv);
}

vec3 blinnPhong(vec3 n, vec3 v, vec3 l, vec3 radiance, vec3 albedo, vec3 specular) {
    float n_dot_l = max(dot(n, l), 0.0);
    vec3 h = normalize(l + v);
    float n_dot_h = max(dot(n, h), 0.0);
    float specular_factor = n_dot_l > 0.0 ? pow(n_dot_h, model_uniforms.shininess) : 0.0;

    return radiance * (albedo * n_dot_l + specular * specular_factor);
}

void main() {
    if (model_uniforms.unlit != 0u) {
        out_color = vec4(model_uniforms.albedo, 1.0);
        return;
    }

    vec3 albedo = model_uniforms.albedo * sampleTexture(albedo_texture, frag_uv).rgb;
    vec3 specular = model_uniforms.specular * sampleTexture(specular_texture, frag_uv).rgb;
    vec3 emissive = sampleTexture(emissive_texture, frag_uv).rgb;

    vec3 n = normalize(frag_normal);
    vec3 v = normalize(scene_uniforms.camera_position - frag_world_position);

    vec3 color = scene_uniforms.ambient_light * albedo;

    color += blinnPhong(n, v, -scene_uniforms.directional_light_direction,
                        scene_uniforms.directional_light_color, albedo, specular);

    for (uint i = 0u; i < scene_uniforms.point_light_count; ++i) {
        vec3 to_light = point_lights[i].position - frag_world_position;
        float dist2 = max(dot(to_light, to_light), 1e-4);
        vec3 l = to_light * inversesqrt(dist2);

        color += blinnPhong(n, v, l, point_lights[i].color / dist2, albedo, specular);
    }

    for (uint i = 0u; i < scene_uniforms.spot_light_count; ++i) {
        vec3 to_light = spot_lights[i].position - frag_world_position;
        float dist2 = max(dot(to_light, to_light), 1e-4);
        vec3 l = to_light * inversesqrt(dist2);

        float cos_angle = dot(-l, spot_lights[i].direction);
        float edge = clamp((cos_angle - spot_lights[i].outer_cos) /
                           max(spot_lights[i].inner_cos - spot_lights[i].outer_cos, 1e-4), 0.0, 1.0);

        color += blinnPhong(n, v, l, spot_lights[i].color * (edge / dist2), albedo, specular);
    }

    out_color = vec4(color + emissive, 1.0);
}
