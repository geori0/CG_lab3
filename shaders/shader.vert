#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec3 frag_world_position;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec2 frag_uv;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    vec3 directional_light_direction;
    vec3 directional_light_color;
    vec3 ambient_light;
    uint point_light_count;
    uint spot_light_count;
} scene_uniforms;

layout(std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    mat4 normal_matrix;
    vec3 albedo;
    vec3 specular;
    float shininess;
    uint unlit;
} model_uniforms;

void main() {
    vec4 world_position = model_uniforms.model * vec4(in_position, 1.0);
    gl_Position = scene_uniforms.view_projection * world_position;

    frag_world_position = world_position.xyz;
    frag_normal = mat3(model_uniforms.normal_matrix) * in_normal;
    frag_uv = in_uv;
}
