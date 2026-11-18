shader_type spatial;
render_mode skip_vertex_transform, cull_back, diffuse_burley, specular_schlick_ggx;

uniform sampler2D grain_state : filter_nearest, repeat_disable;
uniform sampler2D grain_palette : filter_nearest, repeat_disable;
uniform ivec3 grid_size;
uniform float cell_size;
uniform float interpolation;

varying vec3 albedo;

void vertex() {
	int width = textureSize(grain_state, 0).x;
	vec4 grain = texelFetch(grain_state, ivec2(INSTANCE_ID % width, INSTANCE_ID / width), 0);
	ivec3 encoded = ivec3(grain.xyz);
	int color_index = int(grain.w);
	int style = color_index >> 8;
	int extent = 1 << (style & 3);
	vec3 previous = vec3(encoded / grid_size);
	vec3 current = vec3(encoded % grid_size);
	vec3 center = (mix(previous, current, interpolation) + vec3(float(extent) * 0.5)) * cell_size;
	float size = float(extent) * cell_size * (extent == 1 ? 0.974 : 0.98);
	// The batch is anchored at the voxel world's origin. Its instance slots
	// identify texels, while the texture holds all positions and sizes.
	VERTEX = (VIEW_MATRIX * vec4(center + VERTEX * size, 1.0)).xyz;
	NORMAL = mat3(VIEW_MATRIX) * NORMAL;
	albedo = texelFetch(grain_palette, ivec2(color_index & 255, style), 0).rgb;
}

void fragment() {
	ALBEDO = albedo;
	ROUGHNESS = 0.9;
}
