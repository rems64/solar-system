#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <memory>
#include <set>

#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// clang-format off
std::vector<float> fullscreen_rect_vertices = {
    -1.0f, -1.0f,
     1.0f, -1.0f,
     1.0f,  1.0f,
    -1.0f,  1.0f,
};

std::vector<unsigned int> fullscreen_rect_indices = {
    0, 1, 3,
    1, 2, 3
};
// clang-format on

void error_callback(int error,
					const char *description)
{
	std::cout << "Error: " << description << std::endl;
}

void add_float3_radius_angles(std::vector<float> &vector,
							  float radius,
							  float theta,
							  float phi)
{
	vector.push_back(radius * sin(theta) * cos(phi));
	vector.push_back(radius * sin(theta) * sin(phi));
	vector.push_back(radius * cos(theta));
}

char *readfile(const char *filepath)
{
	FILE *fp;
	fp = fopen(filepath, "r");
	if (!fp)
	{
		printf("[ERROR] Failed to open %s", filepath);
		return NULL;
	}
	fseek(fp, 0L, SEEK_END);
	long lSize = ftell(fp);
	rewind(fp);
	char *buffer = (char *)calloc(1, lSize + 1);
	if (!buffer)
	{
		printf("[ERROR] Failed to allocate memory for file %s", filepath);
		return NULL;
	}
	fread(buffer, lSize, 1, fp);
	return buffer;
}

void create_shader(const char *vertex_path,
				   const char *fragment_path,
				   unsigned int *program)
{
	int success;
	char info_log[512];

	unsigned int vertex_shader, fragment_shader;
	*program = glCreateProgram();
	char *vertex_shader_source = readfile(vertex_path);
	vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
	glCompileShader(vertex_shader);
	// Check compilation
	glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertex_shader, 512, NULL, info_log);
		std::cout << "Vertex shader compilation failed\n"
				  << info_log << std::endl;
	}
	free(vertex_shader_source);

	char *fragment_shader_source = readfile(fragment_path);
	fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);
	glCompileShader(fragment_shader);
	// Check compilation
	glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragment_shader, 512, NULL, info_log);
		std::cout << "Fragment shader compilation failed\n"
				  << info_log << std::endl;
	}
	free(fragment_shader_source);

	glAttachShader(*program, vertex_shader);
	glAttachShader(*program, fragment_shader);

	glLinkProgram(*program);

	glGetProgramiv(*program, GL_LINK_STATUS, &success);
	if (!success)
	{
		glGetProgramInfoLog(*program, 512, NULL, info_log);
		std::cout << "Shader linking failed\n"
				  << info_log << std::endl;
	}

	glDeleteShader(vertex_shader);
	glDeleteShader(fragment_shader);
}

void create_compute_shader(const char *compute_path,
						   unsigned int *program)
{
	int success;
	char info_log[512];

	unsigned int compute_shader;
	*program = glCreateProgram();
	char *compute_shader_source = readfile(compute_path);
	compute_shader = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(compute_shader, 1, &compute_shader_source, NULL);
	glCompileShader(compute_shader);

	// Check compilation
	glGetShaderiv(compute_shader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(compute_shader, 512, NULL, info_log);
		std::cout << "Compute shader compilation failed\n"
				  << info_log << std::endl;
	}
	free(compute_shader_source);

	glAttachShader(*program, compute_shader);
	glLinkProgram(*program);

	glGetProgramiv(*program, GL_LINK_STATUS, &success);
	if (!success)
	{
		glGetProgramInfoLog(*program, 512, NULL, info_log);
		std::cout << "Shader linking failed\n"
				  << info_log << std::endl;
	}

	glDeleteShader(compute_shader);
}

float rfloat(float min, float max)
{
	return ((float)rand() / RAND_MAX) * (max - min) + min;
}

std::pair<std::vector<float>, std::vector<unsigned int>>
generate_sphere(size_t rings,
				size_t segments,
				float radius,
				bool inverse_normals = false)
{
	const size_t vertices_count = (rings - 1) * segments + 2;
	const size_t triangles_count = (rings - 2) * segments * 2 + 2 * segments;

	const size_t vertex_components_count = 6; // position(3), normal(3)

	std::vector<float> vertices;
	std::vector<unsigned int> indices;

	size_t index_index = 0;
	const size_t vertex_stride = 3;

	for (size_t ring = 0; ring <= rings; ring++)
	{
		// We have to have two overlapping vertices on one segment
		for (size_t segment = 0; segment <= segments; segment++)
		{
			// position
			add_float3_radius_angles(
				vertices,
				// radius * rfloat(0.98f, 1.02f),
				radius,
				M_PI * ring / rings,
				2 * M_PI * segment / segments);
			// if (segment == segments)
			// {
			// 	vertices[vertices.size() - 0] = vertices[vertices.size() - 8 * (segments - 1) - 0];
			// 	vertices[vertices.size() - 1] = vertices[vertices.size() - 8 * (segments - 1) - 1];
			// 	vertices[vertices.size() - 2] = vertices[vertices.size() - 8 * (segments - 1) - 2];
			// }
			// normal
			const size_t index = vertices.size();
			add_float3_radius_angles(
				vertices,
				radius,
				M_PI * ring / rings,
				2 * M_PI * segment / segments);
			float length = sqrt(vertices[index] * vertices[index] + vertices[index + 1] * vertices[index + 1] + vertices[index + 2] * vertices[index + 2]);
			if (inverse_normals)
				length *= -1.f;
			vertices[index + 0] /= length;
			vertices[index + 1] /= length;
			vertices[index + 2] /= length;

			// uv
			vertices.push_back((float)segment / segments);
			vertices.push_back((float)ring / rings);
		}
	}

	for (size_t ring = 0; ring <= (rings - 1); ring++)
	{
		for (size_t segment = 0; segment < segments; segment++)
		{
			size_t v1 = ring * (segments + 1) + segment;
			size_t v2 = v1 + 1;
			size_t v3 = (ring + 1) * (segments + 1) + segment;
			size_t v4 = v3 + 1;
			// Triangle 1
			indices.push_back(v2);
			indices.push_back(v1);
			indices.push_back(v3);

			// Triangle 2
			indices.push_back(v4);
			indices.push_back(v2);
			indices.push_back(v3);
		}
	}

	return std::pair<std::vector<float>, std::vector<unsigned int>>(vertices, indices);
}

struct Transform
{
	// Local space information
	glm::vec3 position = {0.0f, 0.0f, 0.0f};
	glm::vec3 rotation = {0.0f, 0.0f, 0.0f};
	glm::vec3 scale = {1.0f, 1.0f, 1.0f};

	// Global space matrix
	glm::mat4 model_matrix = glm::mat4(1.0f);

	glm::mat4 get_local_model_matrix() const
	{
		glm::mat4 rotation_matrix = glm::mat4(1.f);
		rotation_matrix = glm::rotate(rotation_matrix, rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
		rotation_matrix = glm::rotate(rotation_matrix, rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
		rotation_matrix = glm::rotate(rotation_matrix, rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));

		// Order matters too
		return glm::translate(glm::mat4(1.0f), position) * rotation_matrix * glm::scale(glm::mat4(1.0f), scale);
	}
};

constexpr GLenum get_format(const uint32_t depth)
{
	switch (depth)
	{
	case 3:
		return GL_RGB32F;
	case 4:
		return GL_RGBA32F;
	default:
		break;
	}
	std::cout << "[WARNING] unsupported format (" << depth << ")" << std::endl;
	return GL_RGBA32F;
}

constexpr GLenum get_components(const uint32_t depth)
{
	switch (depth)
	{
	case 3:
		return GL_RGB;
	case 4:
		return GL_RGBA;
	default:
		break;
	}
	std::cout << "[WARNING] unsupported format (" << depth << ")" << std::endl;
	return GL_RGBA;
}

class Shader
{
public:
	Shader(const char *vertex_path, const char *fragment_path)
	{
		create_shader(vertex_path, fragment_path, &m_program);
	};

	void bind()
	{
		glUseProgram(m_program);
	}

	void set_uniform_mat4fv(const char *name, const glm::mat4 &matrix)
	{
		glUniformMatrix4fv(
			glGetUniformLocation(m_program, name),
			1,
			GL_FALSE,
			glm::value_ptr(matrix));
	}

	void set_uniform_vec2fv(const char *name, const glm::vec2 &vec)
	{
		glUniform2fv(
			glGetUniformLocation(m_program, name),
			1,
			glm::value_ptr(vec));
	}

	void set_uniform_vec3fv(const char *name, const glm::vec3 &vec)
	{
		glUniform3fv(
			glGetUniformLocation(m_program, name),
			1,
			glm::value_ptr(vec));
	}

	void set_uniform_float(const char *name, const float &value)
	{
		glUniform1f(
			glGetUniformLocation(m_program, name),
			value);
	}

	void set_uniform_int(const char *name, const unsigned int &value)
	{
		glUniform1i(
			glGetUniformLocation(m_program, name),
			value);
	}

private:
	unsigned int m_program;
};

class Texture
{
public:
	Texture(const char *path) : m_resource{0}, m_data(nullptr)
	{
		m_data = stbi_load(path, &m_width, &m_height, &m_depth, 0);

		generate_texture(GL_UNSIGNED_BYTE, get_format(m_depth));

		stbi_image_free(m_data);
	}

	Texture(uint32_t width, uint32_t height, uint32_t depth, GLenum type, GLenum internal_format, bool mipmaps = true) : m_resource(0), m_data(nullptr), m_width(width), m_height(height), m_depth(depth)
	{
		generate_texture(type, internal_format, mipmaps);
	}

	const unsigned int get_resource() { return m_resource; };

	const void bind() { glBindTexture(GL_TEXTURE_2D, m_resource); }

	void delete_texture()
	{
		if (!m_resource)
		{
			glDeleteTextures(1, &m_resource);
			m_resource = 0;
		}
	}

public:
	void generate_texture(GLenum type, GLenum internal_format, bool mipmaps = true)
	{
		delete_texture();

		glGenTextures(1, &m_resource);

		glBindTexture(GL_TEXTURE_2D, m_resource);

		glTexImage2D(GL_TEXTURE_2D, 0, internal_format, m_width, m_height,
					 0, get_components(m_depth), type, m_data);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

		glBindImageTexture(0, m_resource, 0, GL_FALSE, 0, GL_READ_WRITE, internal_format);

		glGenerateMipmap(GL_TEXTURE_2D);
	}

private:
	int32_t m_width;
	int32_t m_height;
	int32_t m_depth;
	unsigned char *m_data;
	unsigned int m_resource;
};

class Material
{
public:
	Material(std::shared_ptr<Shader> shader, std::vector<std::shared_ptr<Texture>> textures) : m_shader(shader), m_textures(textures) {};

	const void bind()
	{
		m_shader->bind();
		size_t i = 0;
		for (auto it = m_textures.begin(); it != m_textures.end(); it++, i++)
		{
			glActiveTexture(GL_TEXTURE0 + i);
			// std::cout << "active texture " << i << " (" << it->get()->get_resource() << ")" << std::endl;
			it->get()->bind();
		}
	}

	const std::shared_ptr<Shader> get_shader() { return m_shader; };

private:
	std::shared_ptr<Shader> m_shader;
	std::vector<std::shared_ptr<Texture>> m_textures;
};

class Drawable
{
public:
	Drawable(std::shared_ptr<Material> material) : m_material(material) {};
	~Drawable() = default;

	virtual void draw() const
	{
		m_material->bind();
	};

	const std::shared_ptr<Material> get_material() { return m_material; };

private:
	std::shared_ptr<Material> m_material;
};

class SceneElement
{
public:
	SceneElement() : m_transform{}, m_parent(nullptr), m_children{} {};
	~SceneElement() = default;

	void set_position(glm::vec3 position) { m_transform.position = position; };
	void add_position(glm::vec3 delta_position) { m_transform.position += delta_position; };
	glm::vec3 get_position() { return m_transform.position; };
	glm::vec3 get_global_position() { return glm::vec3(get_model_matrix() * glm::vec4(0, 0, 0, 1)); };

	void set_rotation(glm::vec3 rotation) { m_transform.rotation = rotation; };
	void add_rotation(glm::vec3 delta_rotation) { m_transform.rotation += delta_rotation; };
	glm::vec3 get_rotation() { return m_transform.rotation; };

	void set_scale(glm::vec3 scale) { m_transform.scale = scale; };
	void add_scale(glm::vec3 delta_scale) { m_transform.scale += delta_scale; };
	glm::vec3 get_scale() { return m_transform.scale; };

	const glm::mat4 &get_model_matrix() const { return m_transform.model_matrix; };
	const Transform &get_transform() const { return m_transform; };

	virtual void update()
	{
		if (m_parent)
		{
			m_transform.model_matrix = m_parent->m_transform.model_matrix * m_transform.get_local_model_matrix();
		}
		else
		{
			m_transform.model_matrix = m_transform.get_local_model_matrix();
		}

		for (auto it = m_children.begin(); it != m_children.end(); it++)
		{
			if (*it)
				(*it)->update();
		}
	}

	void add_child(SceneElement *child)
	{
		m_children.insert(child);
	}

	void set_parent(SceneElement *parent)
	{
		m_parent = parent;
		parent->add_child(this);
	}

protected:
	Transform m_transform;

private:
	SceneElement *m_parent;
	std::set<SceneElement *> m_children;
};

class Mesh : public Drawable, public SceneElement
{
public:
	Mesh(std::shared_ptr<Material> material) : Drawable(material), m_vao(0), m_vertices{}, m_indices{} {};

	void build_vao(std::vector<float> &vertices, std::vector<unsigned int> &indices)
	{
		unsigned int vbo;
		unsigned int ebo;

		// VAO
		glGenVertexArrays(1, &m_vao);
		glGenBuffers(1, &vbo);
		glGenBuffers(1, &ebo);

		glBindVertexArray(m_vao);

		// VBO
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER,
					 vertices.size() * sizeof(float),
					 vertices.data(), GL_STATIC_DRAW);
		// EBO
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
					 ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER,
					 indices.size() * sizeof(unsigned int),
					 indices.data(),
					 GL_STATIC_DRAW);

		glVertexAttribPointer(0,
							  3,
							  GL_FLOAT,
							  GL_FALSE,
							  8 * sizeof(float),
							  (void *)0);
		glEnableVertexAttribArray(0);

		glVertexAttribPointer(1,
							  3,
							  GL_FLOAT,
							  GL_FALSE,
							  8 * sizeof(float),
							  (void *)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);

		glVertexAttribPointer(2,
							  2,
							  GL_FLOAT,
							  GL_FALSE,
							  8 * sizeof(float),
							  (void *)(6 * sizeof(float)));
		glEnableVertexAttribArray(2);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	void draw() const
	{
		Drawable::draw();
		glBindVertexArray(m_vao);
		glDrawElements(GL_TRIANGLES, m_indices.size(), GL_UNSIGNED_INT, 0);
	}

protected:
	std::vector<float> m_vertices;
	std::vector<unsigned int> m_indices;

	unsigned int m_vao;
};

class Sphere : public Mesh
{
public:
	Sphere(size_t rings, size_t segments, float radius, std::shared_ptr<Material> material, bool inverse_normals = false) : Mesh(material), m_radius(radius)
	{
		auto [vertices, indices] = generate_sphere(rings, segments, radius, inverse_normals);
		m_vertices = std::move(vertices);
		m_indices = std::move(indices);

		build_vao(m_vertices, m_indices);
	}

	constexpr float radius() const { return m_radius; }

private:
	float m_radius;
};

class ComputeShader
{
public:
	ComputeShader(const char *shader_path)
	{
		create_compute_shader(shader_path, &m_program);
	};

	void bind()
	{
		glUseProgram(m_program);
	}

	void set_uniform_mat4fv(const char *name, const glm::mat4 &matrix)
	{
		glUniformMatrix4fv(
			glGetUniformLocation(m_program, name),
			1,
			GL_FALSE,
			glm::value_ptr(matrix));
	}

private:
	unsigned int m_program;
};

class Camera : public SceneElement
{
public:
	Camera(float fovy, float aspect, float near, float far) : m_target(glm::vec3(0)),
															  m_projection(glm::mat4(1.)),
															  m_vp(glm::mat4(1.)),
															  m_fovy(fovy),
															  m_aspect(aspect),
															  m_near(near),
															  m_far(far)
	{
	}

	void set_aspect(float aspect) { m_aspect = aspect; }

	void update() override
	{
		SceneElement::update();

		m_projection = glm::perspective(
			m_fovy / 180.f * glm::pi<float>(),
			m_aspect,
			m_near,
			m_far);
		m_vp = m_projection * glm::inverse(get_model_matrix());
	};

	void look_at(glm::vec3 target)
	{
		m_target = target;
	}

	glm::mat4 &get_vp()
	{
		return m_vp;
	}

	virtual void move(glm::vec2 delta) {};

private:
	float m_fovy;
	float m_aspect;
	float m_near;
	float m_far;

	glm::vec3 m_target;
	glm::mat4 m_projection;
	glm::mat4 m_vp;
};

class CameraController : public SceneElement
{
public:
	CameraController(float fovy, float aspect, float near, float far)
	{
		m_camera = std::make_shared<Camera>(fovy, aspect, near, far);
		m_camera->set_parent(this);
		m_camera->add_rotation(glm::vec3(glm::pi<float>() / 2.f, 0.f, 0.f));
	}

	std::shared_ptr<Camera> camera() const { return m_camera; };

	virtual void move(glm::vec2 delta) {};
	virtual void zoom(float delta) {};

protected:
	std::shared_ptr<Camera> m_camera;
};

class OrbitCameraController : public CameraController
{
public:
	OrbitCameraController(float fovy, float aspect, float near, float far, float distance) : CameraController(fovy, aspect, near, far), m_distance(distance)
	{
		set_position(glm::vec3(0.f));
		update_camera_distance();
	}

	void update_camera_distance()
	{
		m_camera->set_position(glm::vec3(0.f, -m_distance, 0.f));
		update();
	}

	void move(glm::vec2 delta) override
	{
		add_rotation(glm::vec3(-delta.y, 0.f, -delta.x));
		update();
	}

	void zoom(float delta) override
	{
		m_distance += -delta;
		update_camera_distance();
	}

private:
	float m_distance;
};

struct G
{
	struct Input
	{
		glm::vec2 cursor_position;
		bool cursor_left_down;
	} input;

	struct Screen
	{
		int width = 1024;
		int height = 1024;
		bool dirty = true;
	} screen;

	std::shared_ptr<CameraController> camera;
} g;

void resize_callback(GLFWwindow *window, int width, int height)
{
	g.screen.width = width;
	g.screen.height = height;
	g.screen.dirty = true;
	glViewport(0, 0, width, height);
}

void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos)
{
	glm::vec2 new_pos = glm::vec2(xpos, ypos);
	glm::vec2 delta = new_pos - g.input.cursor_position;
	if (g.input.cursor_left_down)
	{
		glm::vec3 camera_position = g.camera->get_position();
		float dphi = 0.01f * delta.x;
		float dtheta = 0.01f * delta.y;
		g.camera->move(glm::vec2(dphi, dtheta));
		// g.camera->set_position(glm::vec3(camera_position.x * cos(dphi) - camera_position.y * sin(dphi), camera_position.x * sin(dphi) + camera_position.y * cos(dphi), camera_position.z + dtheta));
	}
	g.input.cursor_position = new_pos;
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
	g.camera->zoom(yoffset);
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
	switch (button)
	{
	case GLFW_MOUSE_BUTTON_LEFT:
		g.input.cursor_left_down = action == GLFW_PRESS;
		break;

	default:
		break;
	}
}

void setup(GLFWwindow *&window)
{
	glfwSetErrorCallback(error_callback);

	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(g.screen.width, g.screen.height, "Window", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
	}

	glfwMakeContextCurrent(window);
	gladLoadGL();

	glfwSetWindowSizeCallback(window, resize_callback);
	glfwSetCursorPosCallback(window, cursor_pos_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO &io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;	  // IF using Docking Branch

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(window, true); // Second param install_callback=true will install GLFW callbacks and chain to existing ones.
	ImGui_ImplOpenGL3_Init();
}

struct FramebufferAttachment
{
	GLenum type = GL_FLOAT;
	uint32_t components_count = 4;
	GLenum internal_format = GL_RGBA;
};

class Framebuffer
{
public:
	Framebuffer(uint32_t width, uint32_t height, std::vector<FramebufferAttachment> attachments) : m_width(width), m_height(height), m_attachments(attachments)
	{
		record(width, height);
	};

	void bind()
	{
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_framebuffer);
	}

	void record(uint32_t width, uint32_t height)
	{
		if (!m_framebuffer)
			glDeleteFramebuffers(1, &m_framebuffer);

		m_buffers.clear();

		glGenFramebuffers(1, &m_framebuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer);

		// TODO: a "default" texture is definitely not the way to do it...
		size_t i = 0;
		for (auto it = m_attachments.begin(); it != m_attachments.end(); it++, i++)
		{
			auto buffer = std::make_shared<Texture>(width, height, it->components_count, it->type, it->internal_format, false);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, buffer->get_resource(), 0);

			m_buffers.push_back(buffer);
		}

		std::vector<unsigned int> attachments{};
		for (size_t i = 0; i < m_attachments.size(); i++)
		{
			attachments.push_back(GL_COLOR_ATTACHMENT0 + i);
		}
		glDrawBuffers(attachments.size(), attachments.data());

		glGenRenderbuffers(1, &m_depth);
		glBindRenderbuffer(GL_RENDERBUFFER, m_depth);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depth);

		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			std::cout << "[WARNING] Framebuffer not complete!" << std::endl;
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	};

	std::vector<std::shared_ptr<Texture>> &get_buffers() { return m_buffers; };

private:
	uint32_t m_width;
	uint32_t m_height;
	uint32_t m_depth;
	std::vector<FramebufferAttachment> m_attachments;
	std::vector<std::shared_ptr<Texture>> m_buffers;

	unsigned int m_framebuffer;
};

#include "capture.hpp"

int main()
{
	// video_encode_example("test.mp4", AV_CODEC_ID_H264);
	GLFWwindow *window;

	setup(window);

	// Nice distribution
	srand(42);

	std::vector<std::shared_ptr<Drawable>> drawables;

	std::vector<FramebufferAttachment> gbuffer_attachments = {
		FramebufferAttachment{.type = GL_FLOAT, .components_count = 4, .internal_format = GL_RGBA16F}, // color (HDR)
		FramebufferAttachment{.type = GL_FLOAT, .components_count = 4, .internal_format = GL_RGBA16F}, // position (4 for alignment)
		FramebufferAttachment{.type = GL_FLOAT, .components_count = 4, .internal_format = GL_RGBA16F}, // normal (4 for alignment)
		FramebufferAttachment{.type = GL_FLOAT, .components_count = 4, .internal_format = GL_RGBA16F}, // metallic + roughness + emissiveness
	};
	auto g_buffer = std::make_shared<Framebuffer>(g.screen.width, g.screen.height, gbuffer_attachments);

	auto shading_framebuffer = std::make_shared<Framebuffer>(g.screen.width, g.screen.height, gbuffer_attachments);

	auto composite_framebuffer = std::make_shared<Framebuffer>(g.screen.width, g.screen.height, gbuffer_attachments);

	auto solar_root = std::make_shared<SceneElement>();

	auto sun_texture = std::make_shared<Texture>("textures/2k_sun.jpg");
	auto earth_texture = std::make_shared<Texture>("textures/2k_earth_daymap.jpg");
	auto earth_normal_texture = std::make_shared<Texture>("textures/2k_earth_normal_map.png");
	auto earth_specular_texture = std::make_shared<Texture>("textures/2k_earth_specular_map.png");
	auto moon_texture = std::make_shared<Texture>("textures/2k_moon.jpg");

	// auto stars_texture = std::make_shared<Texture>("textures/2k_sun.jpg");
	auto stars_texture = std::make_shared<Texture>("textures/2k_stars.jpg");

	auto simple_texture_shader = std::make_shared<Shader>("shaders/base.vert", "shaders/textured.frag");
	auto sun_shader = std::make_shared<Shader>("shaders/base.vert", "shaders/sun.frag");
	auto atmosphere_shader = std::make_shared<Shader>("shaders/compositing.vert", "shaders/atmosphere.frag");

	auto sun_material = std::make_shared<Material>(sun_shader, std::vector<std::shared_ptr<Texture>>{sun_texture});
	auto earth_material = std::make_shared<Material>(simple_texture_shader, std::vector<std::shared_ptr<Texture>>{earth_texture, earth_normal_texture, earth_specular_texture});
	auto moon_material = std::make_shared<Material>(simple_texture_shader, std::vector<std::shared_ptr<Texture>>{moon_texture});
	auto atmosphere_material = std::make_shared<Material>(atmosphere_shader, std::vector<std::shared_ptr<Texture>>{});

	auto sun = std::make_shared<Sphere>(20, 40, 1.0f, sun_material);
	auto earth = std::make_shared<Sphere>(20, 40, 0.5f, earth_material);
	auto moon = std::make_shared<Sphere>(20, 40, 0.1f, moon_material);

	// auto atmosphere = std::make_shared<Sphere>(20, 40, earth->radius() * 1.5f, atmosphere_material, true);
	float atmosphere_multiplier = 1.15f;
	bool move_planets = false;

	drawables.push_back(sun);
	drawables.push_back(earth);
	drawables.push_back(moon);

	moon->set_parent(earth.get());
	earth->set_parent(solar_root.get());
	sun->set_parent(solar_root.get());
	// atmosphere->set_parent(earth.get());

	earth->set_position(glm::vec3(4., 0., 0.));
	moon->set_position(glm::vec3(2., 0., 0.));

	solar_root->update();

	std::shared_ptr<OrbitCameraController> camera = std::make_shared<OrbitCameraController>(60.f, (float)g.screen.width / (float)g.screen.height, 0.1f, 100.f, 10.f);
	g.camera = camera;

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	// glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE);

	// glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	// glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	// glBlendFunc(GL_DST_ALPHA, GL_MIN);
	// glBlendFunc(GL_SRC_ALPHA, GL_MIN);
	// glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	// glBlendFunc(GL_SRC_ALPHA, GL_SRC_ALPHA_SATURATE);

	simple_texture_shader->bind();
	simple_texture_shader->set_uniform_int("s_albedo", 0);
	simple_texture_shader->set_uniform_int("s_normal", 1);
	simple_texture_shader->set_uniform_int("s_specular", 2);
	simple_texture_shader->set_uniform_vec3fv("tint", glm::vec3(1, 0.5, 0.5));

	// Deferred shading
	auto shading_shader = std::make_shared<Shader>("shaders/shading.vert", "shaders/shading.frag");
	auto compositing_shader = std::make_shared<Shader>("shaders/compositing.vert", "shaders/compositing.frag");
	unsigned int fullscreen_vao, fullscreen_vbo, fullscreen_ebo;
	glGenVertexArrays(1, &fullscreen_vao);
	glGenBuffers(1, &fullscreen_vbo);
	glGenBuffers(1, &fullscreen_ebo);
	glBindVertexArray(fullscreen_vao);
	// VBO
	glBindBuffer(GL_ARRAY_BUFFER, fullscreen_vbo);
	glBufferData(GL_ARRAY_BUFFER, fullscreen_rect_vertices.size() * sizeof(float), fullscreen_rect_vertices.data(), GL_STATIC_DRAW);
	// EBO
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fullscreen_ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, fullscreen_rect_indices.size() * sizeof(unsigned int), fullscreen_rect_indices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
	glEnableVertexAttribArray(0);

	double start_time = glfwGetTime();

	const char *focus_bodies[] = {"sun", "earth", "moon"};
	static const char *current_focus_body = focus_bodies[0];

	while (!glfwWindowShouldClose(window))
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		double time = glfwGetTime() - start_time;

		g_buffer->bind();

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, g.screen.width, g.screen.height);

		if (g.screen.dirty)
		{
			g_buffer->record(g.screen.width, g.screen.height);
			composite_framebuffer->record(g.screen.width, g.screen.height);
			shading_framebuffer->record(g.screen.width, g.screen.height);
			g.screen.dirty = false;
		}

		camera->camera()->set_aspect((float)g.screen.width / g.screen.height);
		glm::vec3 camera_position;
		if (strcmp(current_focus_body, "sun") == 0)
			camera_position = sun->get_global_position();
		else if (strcmp(current_focus_body, "earth") == 0)
			camera_position = earth->get_global_position();
		else if (strcmp(current_focus_body, "moon") == 0)
			camera_position = moon->get_global_position();
		else
			camera_position = glm::vec3(0.f);

		camera->set_position(camera_position);
		camera->update();
		// camera->set_rotation(cam_rot);
		// camera->update();

		if (move_planets)
		{
			earth->set_position(4.f * glm::vec3(glm::cos(time), glm::sin(time), 0.));
			moon->set_position(1.f * glm::vec3(glm::cos(3 * time), glm::sin(3 * time), 0.));
			sun->set_rotation(glm::vec3(0, 0, -0.4f * time));
		}
		solar_root->update();

		for (auto it = drawables.begin(); it != drawables.end(); it++)
		{
			SceneElement *scene_element = dynamic_cast<SceneElement *>(it->get());
			it->get()->get_material()->bind();
			if (scene_element)
			{
				// TODO: group by material (material instances...)
				it->get()->get_material()->get_shader()->set_uniform_mat4fv("local_model", scene_element->get_transform().get_local_model_matrix());
				it->get()->get_material()->get_shader()->set_uniform_mat4fv("vp", camera->camera()->get_vp());
				it->get()->get_material()->get_shader()->set_uniform_mat4fv("model", (scene_element->get_model_matrix()));
			}

			it->get()->draw();
		}

		glDisable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);

		// Shading
		shading_framebuffer->bind();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, g.screen.width, g.screen.height);
		size_t i = 0;
		auto buffers = g_buffer->get_buffers();
		for (auto it = buffers.begin(); it != buffers.end(); it++, i++)
		{
			glActiveTexture(GL_TEXTURE0 + i);
			it->get()->bind();
		}
		shading_shader->bind();
		glBindVertexArray(fullscreen_vao);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		// Compositing
		composite_framebuffer->bind();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, g.screen.width, g.screen.height);

		i = 0;
		buffers = shading_framebuffer->get_buffers();
		for (auto it = buffers.begin(); it != buffers.end(); it++, i++)
		{
			glActiveTexture(GL_TEXTURE0 + i);
			it->get()->bind();
		}

		glActiveTexture(GL_TEXTURE4);
		stars_texture->bind();

		compositing_shader->bind();
		compositing_shader->set_uniform_mat4fv("view_projection", camera->camera()->get_vp());

		glBindVertexArray(fullscreen_vao);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		// Atmosphere
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, g.screen.width, g.screen.height);
		i = 0;
		buffers = composite_framebuffer->get_buffers();
		for (auto it = buffers.begin(); it != buffers.end(); it++, i++)
		{
			glActiveTexture(GL_TEXTURE0 + i);
			it->get()->bind();
		}

		atmosphere_material->bind();
		// atmosphere_material->get_shader()->set_uniform_mat4fv("vp", camera->get_vp());
		// atmosphere_material->get_shader()->set_uniform_mat4fv("model", (atmosphere->get_model_matrix()));

		atmosphere_material->get_shader()->set_uniform_mat4fv("inv_vp", glm::inverse(camera->camera()->get_vp()));
		atmosphere_material->get_shader()->set_uniform_vec2fv("viewport_size", glm::vec2(g.screen.width, g.screen.height));
		atmosphere_material->get_shader()->set_uniform_vec3fv("camera_position", glm::vec3(camera->camera()->get_model_matrix() * glm::vec4(0, 0, 0, 1)));
		atmosphere_material->get_shader()->set_uniform_vec3fv("atmosphere_center", glm::vec3(earth->get_model_matrix() * glm::vec4(0, 0, 0, 1)));
		// atmosphere_material->get_shader()->set_uniform_float("atmosphere_radius", atmosphere->radius());
		atmosphere_material->get_shader()->set_uniform_float("atmosphere_radius", earth->radius() * atmosphere_multiplier);
		atmosphere_material->get_shader()->set_uniform_float("planet_radius", earth->radius());

		const glm::vec3 wavelengths = glm::vec3(700, 530, 440);
		float scattering_strength = 20.f;
		float scatter_r = pow(400 / wavelengths.x, 4) * scattering_strength;
		float scatter_g = pow(400 / wavelengths.y, 4) * scattering_strength;
		float scatter_b = pow(400 / wavelengths.z, 4) * scattering_strength;
		atmosphere_material->get_shader()->set_uniform_vec3fv("scattering_coefficients", glm::vec3(scatter_r, scatter_g, scatter_b));

		// atmosphere->draw();
		glBindVertexArray(fullscreen_vao);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		ImGui::Begin("Atmosphere");
		ImGui::DragFloat("Atmosphere size multiplier", &atmosphere_multiplier, 0.1f, 1.f, 2.f);
		ImGui::End();

		ImGui::Begin("Solar system");
		ImGui::Checkbox("Move planets", &move_planets);
		ImGui::End();

		ImGui::Begin("Camera");

		if (ImGui::BeginCombo("Focus body", current_focus_body))
		{
			for (int n = 0; n < IM_ARRAYSIZE(focus_bodies); n++)
			{
				bool is_selected = (current_focus_body == focus_bodies[n]);
				if (ImGui::Selectable(focus_bodies[n], is_selected))
					current_focus_body = focus_bodies[n];
				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}
		ImGui::End();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();

	return EXIT_SUCCESS;
}
