#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (set = 0, binding = 0) uniform UniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

layout (location = 0) in vec2 position;
layout (location = 1) in vec2 uv;

layout (location = 0) out vec2 uvPos;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 0.0, 1.0);
	uvPos = uv;
}
