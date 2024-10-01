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

std::pair<std::vector<float>, std::vector<unsigned int>>
generate_sphere(size_t rings,
				size_t segments,
				float radius)
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
		size_t segments_count = (ring == 0 || ring == rings) ? 0 : segments;
		// We have to have two overlapping vertices on one segment
		for (size_t segment = 0; segment <= segments_count; segment++)
		{
			// position
			add_float3_radius_angles(
				vertices,
				radius,
				M_PI * ring / rings,
				2 * M_PI * segment / segments);
			// uv
			vertices.push_back((float)segment / segments);
			vertices.push_back((float)ring / rings);

			// normal
			const size_t index = vertices.size();
			add_float3_radius_angles(
				vertices,
				radius,
				M_PI * ring / rings,
				2 * M_PI * segment / segments);
			const float length = sqrt(vertices[index] * vertices[index] + vertices[index + 1] * vertices[index + 1] + vertices[index + 2] * vertices[index + 2]);
			vertices[index + 0] /= length;
			vertices[index + 1] /= length;
			vertices[index + 2] /= length;
		}
	}

	// First ring
	for (size_t segment = 0; segment < segments; segment++)
	{
		indices.push_back(0);
		indices.push_back(1 + segment);
		indices.push_back(1 + segment + 1);
	}
	for (size_t ring = 0; ring < (rings - 2); ring++)
	{
		for (size_t segment = 0; segment < segments; segment++)
		{
			size_t v1 = 1 + ring * (segments + 1) + segment;
			size_t v2 = v1 + 1;
			size_t v3 = 1 + (ring + 1) * (segments + 1) + segment;
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

	// Last ring
	const size_t last_ring_index = 1 + (segments + 1) * (rings - 2);
	const size_t last_index = 1 + (segments + 1) * (rings - 1);
	for (size_t segment = 0; segment < segments; segment++)
	{
		indices.push_back(last_ring_index + segment + 1);
		indices.push_back(last_ring_index + segment);
		indices.push_back(last_index);
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

	glm::mat4 get_local_model_matrix()
	{
		const glm::mat4 transform_x = glm::rotate(glm::mat4(1.0f),
												  rotation.x,
												  glm::vec3(1.0f, 0.0f, 0.0f));
		const glm::mat4 transform_y = glm::rotate(glm::mat4(1.0f),
												  rotation.y,
												  glm::vec3(0.0f, 1.0f, 0.0f));
		const glm::mat4 transform_z = glm::rotate(glm::mat4(1.0f),
												  rotation.z,
												  glm::vec3(0.0f, 0.0f, 1.0f));

		// Order matters :)
		const glm::mat4 roation_matrix = transform_y * transform_x * transform_z;

		// Order matters too
		return glm::translate(glm::mat4(1.0f), position) *
			   roation_matrix *
			   glm::scale(glm::mat4(1.0f), scale);
	}
};

class Drawable
{
public:
	Drawable() = default;
	virtual ~Drawable() = default;

	virtual void draw() const = 0;
};

class SceneElement
{
public:
	SceneElement() : m_transform{}, m_parent(nullptr), m_children{} {};
	~SceneElement() = default;

	void set_position(glm::vec3 position) { m_transform.position = position; };
	glm::vec3 get_position() { return m_transform.position; };

	void set_rotation(glm::vec3 rotation) { m_transform.rotation = rotation; };
	glm::vec3 get_rotation() { return m_transform.rotation; };

	void set_scale(glm::vec3 scale) { m_transform.scale = scale; };
	glm::vec3 get_scale() { return m_transform.scale; };

	const glm::mat4 &get_model_matrix() const { return m_transform.model_matrix; };

	void update()
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
	Mesh() : m_vao(0), m_vertices{}, m_indices{} {};

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
							  2,
							  GL_FLOAT,
							  GL_FALSE,
							  8 * sizeof(float),
							  (void *)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);

		glVertexAttribPointer(2,
							  3,
							  GL_FLOAT,
							  GL_FALSE,
							  8 * sizeof(float),
							  (void *)(5 * sizeof(float)));
		glEnableVertexAttribArray(2);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	void draw() const
	{
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
	Sphere(size_t rings, size_t segments, float radius) : Mesh()
	{
		auto [vertices, indices] = generate_sphere(rings, segments, radius);
		m_vertices = std::move(vertices);
		m_indices = std::move(indices);

		build_vao(m_vertices, m_indices);
	}
};

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

	void set_uniform_int(const char *name, const float &value)
	{
		glUniform1f(
			glGetUniformLocation(m_program, name),
			value);
	}

private:
	unsigned int m_program;
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
	Camera(float fovy, float aspect, float near, float far) : m_view(glm::mat4(1.)),
															  m_projection(glm::mat4(1.)),
															  m_vp(glm::mat4(1.)),
															  m_fovy(fovy),
															  m_aspect(aspect),
															  m_near(near),
															  m_far(far)
	{
	}
	void set_position(glm::vec3 position)
	{
		SceneElement::set_position(position);
		update_vp();
	}

	void set_aspect(float aspect)
	{
		m_aspect = aspect;
		update_vp();
	}

	void update_vp()
	{
		m_projection = glm::perspective(
			m_fovy / 180.f * glm::pi<float>(),
			m_aspect,
			m_near,
			m_far);
		m_vp = m_projection * m_view;
	}

	void look_at(glm::vec3 target)
	{
		m_view = glm::lookAt(
			m_transform.position,
			target,
			glm::vec3(0, 0, 1));
		update_vp();
	}

	glm::mat4 &get_vp()
	{
		return m_vp;
	}

private:
	float m_fovy;
	float m_aspect;
	float m_near;
	float m_far;

	glm::mat4 m_view;
	glm::mat4 m_projection;
	glm::mat4 m_vp;
};

int viewport_width = 512;
int viewport_height = 512;

void resize_callback(GLFWwindow *window, int width, int height)
{
	viewport_width = width;
	viewport_height = height;
	glViewport(0, 0, width, height);
}

void setup(GLFWwindow *&window)
{
	glfwSetErrorCallback(error_callback);

	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(viewport_width, viewport_height, "Window", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
	}

	glfwMakeContextCurrent(window);
	gladLoadGL();

	glfwSetWindowSizeCallback(window, resize_callback);
}

float rfloat(float min, float max)
{
	return ((float)rand() / RAND_MAX) * (max - min) + min;
}

int main()
{
	GLFWwindow *window;

	setup(window);

	// Nice distribution
	srand(42);

	std::vector<std::shared_ptr<Drawable>> drawables;

	auto solar_root = std::make_shared<SceneElement>();

	auto moon = std::make_shared<Sphere>(20, 40, 0.1f);
	auto earth = std::make_shared<Sphere>(20, 40, 0.5f);
	auto sun = std::make_shared<Sphere>(20, 40, 1.0f);

	drawables.push_back(sun);
	drawables.push_back(earth);
	drawables.push_back(moon);

	moon->set_parent(earth.get());
	earth->set_parent(solar_root.get());
	sun->set_parent(solar_root.get());

	earth->set_position(glm::vec3(4., 0., 0.));
	moon->set_position(glm::vec3(2., 0., 0.));

	solar_root->update();

	Shader shader("shaders/sphere.vert", "shaders/sphere.frag");
	Camera camera(60.f, (float)viewport_width / (float)viewport_height, 0.1f, 100.f);

	camera.set_position(glm::vec3(0., -7., 5.));
	camera.look_at(glm::vec3(0., 0., 0.));
	camera.update_vp();

	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);

	// Compute
	ComputeShader compute("shaders/test.compute");
	// texture size
	const unsigned int TEXTURE_WIDTH = 512, TEXTURE_HEIGHT = 512;
	unsigned int texture;

	glGenTextures(1, &texture);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, TEXTURE_WIDTH, TEXTURE_HEIGHT, 0, GL_RGBA,
				 GL_FLOAT, NULL);

	glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

	compute.bind();
	glDispatchCompute(TEXTURE_WIDTH, TEXTURE_HEIGHT, 1);

	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	glActiveTexture(GL_TEXTURE0);
	// glBindTexture(GL_TEXTURE_2D, texture);

	int x, y, n;
	unsigned char *data = stbi_load("textures/2k_earth_daymap.jpg", &x, &y, &n, 0);
	if (!data)
	{
		printf("Failed to load earth texture\n");
		return EXIT_FAILURE;
	}
	unsigned int earth_texture;
	glGenTextures(1, &earth_texture);
	glBindTexture(GL_TEXTURE_2D, earth_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, x, y, 0, n == 3 ? GL_RGB : GL_RGBA, GL_UNSIGNED_BYTE, data);
	glGenerateMipmap(GL_TEXTURE_2D);
	stbi_image_free(data);

	glBindTexture(GL_TEXTURE_2D, earth_texture);

	shader.set_uniform_int("tex", 0);

	double start_time = glfwGetTime();
	while (!glfwWindowShouldClose(window))
	{
		double time = glfwGetTime() - start_time;
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		camera.set_aspect((float)viewport_width / viewport_height);
		// camera.set_position(5.f * glm::vec3(glm::cos(time), glm::sin(time), 0.3));
		// camera.look_at(glm::vec3(0, 0, 0));
		camera.update_vp();

		earth->set_position(4.f * glm::vec3(glm::cos(time), glm::sin(time), 0.));
		moon->set_position(1.f * glm::vec3(glm::cos(3 * time), glm::sin(3 * time), 0.));
		sun->set_rotation(glm::vec3(0, 0, 4 * time));
		solar_root->update();

		shader.bind();
		shader.set_uniform_mat4fv("vp", camera.get_vp());
		for (auto it = drawables.begin(); it != drawables.end(); it++)
		{
			SceneElement *scene_element = dynamic_cast<SceneElement *>(it->get());
			if (scene_element)
			{
				shader.set_uniform_mat4fv("model", (scene_element->get_model_matrix()));
			}

			it->get()->draw();
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();

	return EXIT_SUCCESS;
}
