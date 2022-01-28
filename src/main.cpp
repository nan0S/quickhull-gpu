#include <string>
#include <vector>
#include <cstring>

#define GLFW_INCLUDE_NONE
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "GPU/GPU.cuh"
#include "CPU/CPU.h"
#include "Graphics/GLError.h"
#include "Graphics/Shader.h"
#include "Debug/Logging.h"
#include "Config.h"

class comma_numpunct : public std::numpunct<char>
{
protected:
	char do_thousands_sep() const { return ','; }
	std::string do_grouping() const { return "\03"; }
};

enum class ComputeMode
{
	NONE, GPU, CPU
};

const char* USAGE_STR =
"Usage: ./ConvexHull [OPTION]... --gpu|--cpu NUMBER_OF_POINTS...\n\n"
"Compute and display convex hull of a set of points on GPU on CPU, e.g. "
"./ConvexHull --gpu --desc 1000 100000 1000000\n\n"
"List of possible options:\n"
"   --gpu         compute convex hull on GPU\n"
"   --cpu         compute convex hull on CPU\n"
"   --disc        DISC dataset of points\n"
"   --ring        RING dataset of points\n"
"   --circle      CIRCLE dataset of points\n"
"   --seed VALUE  set random seed to VALUE\n"
"   --hostmem     use only host memory, don't map OpenGL to CUDA\n"
"   --help        print this help";

const char* INSTRUCTIONS_STR =
"Press SPACE/ENTER to go to the next test case.\n"
"Press ESCAPE to exit.\n";

const char* vertex_shader_source =
R"(
#version 330 core

layout(location = 0) in float v_PosX;
layout(location = 1) in float v_PosY;

out vec4 v_Color;

uniform float pointSize;
uniform vec4 color;

void main()
{
	v_Color = color;
	gl_PointSize = pointSize;
	gl_Position = vec4(v_PosX, v_PosY, 0, 1);
})";

const char* fragment_shader_source =
R"(
#version 330 core

in vec4 v_Color;
out vec4 f_Color;

void main()
{
	f_Color = v_Color;
})";

static constexpr float HULL_POINT_SIZE = 7.f;
static constexpr float HULL_POINT_COLOR[4] = { 0.f, 1.f, 0.f, 1.f };
static constexpr float NORMAL_POINT_SIZE = 1.f;
static constexpr float NORMAL_POINT_COLOR[4] = { 1.f, 1.f, 1.f, 1.f };
static constexpr float LINE_WIDTH = 0.7f;
static constexpr float LINE_COLOR[4] = { 0.f, 1.f, 0.f, 0.5f };

static constexpr int WIDTH = 800;
static constexpr int HEIGHT = 800;

static constexpr Config DEFAULT_CONFIG{
	DatasetType::DISC, 1234, false };

GLFWwindow* window;
GLint point_size_loc, color_loc;
int N, hull_count;
bool is_displaying, is_next_case_request;

void windowErrorHandler(int errCode, const char* desc);
void windowResizeHandler(GLFWwindow*, int width, int height);
void windowKeyInputHandler(GLFWwindow* window, int key, int scancode,
	int action, int mods);
void setWindowViewport(int width, int height);
void drawHull();
bool displayHull(); // returns false when immediate exit was requested by user

int main(int argc, const char* argv[])
{
	{
		std::locale comma_locale(std::locale(), new comma_numpunct);
		std::cout.imbue(comma_locale);
		std::cerr.imbue(comma_locale);
	}

	ASSERT(argc > 0);

	// Parse and partition flags (-arg and --arg).
	ComputeMode compute_mode = ComputeMode::NONE;
	Config config = DEFAULT_CONFIG;
	int pivot_idx = 1;
	for (int i = 1; i < argc; ++i)
	{
		const char* arg = argv[i];
		if (arg[0] != '-')
		{
			std::swap(argv[i], argv[pivot_idx++]);
			continue;
		}
		const char* flag = (arg[1] == '-' ? arg + 2 : arg + 1);
		if (strcmp(flag, "gpu") == 0)
		{
			if (compute_mode != ComputeMode::NONE)
				ERROR("Specified more than one compute mode.");
			compute_mode = ComputeMode::GPU;
		}
		else if (strcmp(flag, "cpu") == 0)
		{
			if (compute_mode != ComputeMode::NONE)
				ERROR("Specified more than one compute mode.");
			compute_mode = ComputeMode::CPU;
		}
		else if (strcmp(flag, "disc") == 0)
			config.dataset_type = DatasetType::DISC;
		else if (strcmp(flag, "ring") == 0)
			config.dataset_type = DatasetType::RING;
		else if (strcmp(flag, "circle") == 0)
			config.dataset_type = DatasetType::CIRCLE;
		else if (strcmp(flag, "seed") == 0)
		{
			if (i == argc - 1)
				ERROR("Argument for flag '", arg, "' is required.");
			config.seed = std::stoi(argv[++i]);
		}
		else if (strcmp(flag, "hostmem") == 0)
			config.is_host_mem = true;
		else if (strcmp(flag, "help") == 0)
		{
			print(USAGE_STR);
			exit(EXIT_SUCCESS);
		}
		else
			ERROR("Flag '", arg, "' is no recognized.");
	}
	argc = pivot_idx;

	if (compute_mode == ComputeMode::NONE)
		ERROR("Please specify compute mode with --gpu or --cpu.");
	if (argc < 2)
		ERROR("Invalid number of arguments.");

	// Parse positional arguments (number of points).
	std::vector<int> ns(argc - 1);
	for (int i = 1; i < argc; ++i)
	{
		int n = std::stoi(argv[i]);
		if (n <= 0)
			ERROR("Number of points has to be positive.");
		if (n < 3)
			ERROR("Number of points has to be at least 3.");
		ns[i - 1] = n;
	}
	
	glfwSetErrorCallback(windowErrorHandler);
	if (!glfwInit())
		ERROR("Failed to initialize GLFW.");

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(WIDTH, HEIGHT, "Convex Hull", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		ERROR("Failed to create window.");
	}

	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, windowResizeHandler);
	glfwSetKeyCallback(window, windowKeyInputHandler);
	setWindowViewport(WIDTH, HEIGHT);

	if (glewInit() != GLEW_OK)
	{
		glfwDestroyWindow(window);
		glfwTerminate();
		ERROR("Failed to initialize GLEW.");
	}

	glCall(glClearColor(0, 0, 0, 1));
	glCall(glEnable(GL_VERTEX_PROGRAM_POINT_SIZE));
	glCall(glEnable(GL_LINE_SMOOTH));

	GLuint vao;
	glCall(glGenVertexArrays(1, &vao));
	glCall(glBindVertexArray(vao));

	GLuint vbo;
	glCall(glGenBuffers(1, &vbo));
	glCall(glBindBuffer(GL_ARRAY_BUFFER, vbo));
	glCall(glEnableVertexAttribArray(0));
	glCall(glEnableVertexAttribArray(1));

	GLuint shader = Graphics::compileShader(
		vertex_shader_source, fragment_shader_source);
	glCall(glUseProgram(shader));
	glCall(point_size_loc = glGetUniformLocation(shader, "pointSize"));
	glCall(color_loc = glGetUniformLocation(shader, "color"));

	glCall(glLineWidth(LINE_WIDTH));
	glCall(glEnable(GL_BLEND));
	glCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

	glCall(glClear(GL_COLOR_BUFFER_BIT));
	glfwSwapBuffers(window);

	print(INSTRUCTIONS_STR);
	switch (compute_mode)
	{
		case ComputeMode::GPU:
			print("GPU mode selected.");
			break;
		case ComputeMode::CPU:
			print("CPU mode selected.");
			break;
	}
	const char* dataset_name;
	switch (config.dataset_type)
	{
		case DatasetType::DISC:
			dataset_name = "DISC";
			break;
		case DatasetType::RING:
			dataset_name = "RING";
			break;
		case DatasetType::CIRCLE:
			dataset_name = "CIRCLE";
			break;
	}
	print("Running tests on ", dataset_name, " dataset.");

	switch (compute_mode)
	{
		case ComputeMode::GPU:
		{
			GPU::init(config, ns, vbo);
			for (int n : ns)
			{
				hull_count = GPU::calculate(N = n);
				if (!displayHull())
					break;
			}
			GPU::terminate();
			break;
		}
		case ComputeMode::CPU:
		{
			CPU::init(config, ns);
			for (int n : ns)
			{
				hull_count = CPU::calculate(N = n);
				if (!displayHull())
					break;
			}
			CPU::terminate();
			break;
		}
	}

	glCall(glDeleteProgram(shader));
	glCall(glDeleteBuffers(1, &vbo));
	glCall(glDeleteVertexArrays(1, &vao));
	glfwTerminate();

	return 0;
}

void windowErrorHandler(int errCode, const char* desc)
{
	WARNING("[GLFW Error] ", desc, " ", errCode);
}

void windowResizeHandler(GLFWwindow*, int width, int height)
{
	setWindowViewport(width, height);
	if (is_displaying)
		drawHull();
}

void windowKeyInputHandler(GLFWwindow* window, int key, int,
	int action, int)
{
	switch (key)
	{
		case GLFW_KEY_ESCAPE:
			glfwSetWindowShouldClose(window, GLFW_TRUE);
			break;
		case GLFW_KEY_ENTER:
		case GLFW_KEY_SPACE:
			if (action == GLFW_PRESS)
				is_next_case_request = true;
			break;
	}
}

void setWindowViewport(int width, int height)
{
	int minSize = std::min(width, height);
	static constexpr float borderFactor = 0.05f;
	int border = static_cast<int>(borderFactor * minSize);
	int corner = minSize - 2 * border;
	glCall(glViewport(border, border, corner, corner));
}

void drawHull()
{
	glCall(glClear(GL_COLOR_BUFFER_BIT));

	glCall(glUniform1f(point_size_loc, NORMAL_POINT_SIZE));
	glCall(glUniform4fv(color_loc, 1, NORMAL_POINT_COLOR));
	glCall(glDrawArrays(GL_POINTS, hull_count, N - hull_count));
	glCall(glUniform4fv(color_loc, 1, LINE_COLOR));
	glCall(glDrawArrays(GL_LINE_LOOP, 0, hull_count));
	glCall(glUniform1f(point_size_loc, HULL_POINT_SIZE));
	glCall(glUniform4fv(color_loc, 1, HULL_POINT_COLOR));
	glCall(glDrawArrays(GL_POINTS, 0, hull_count));

	glfwSwapBuffers(window);
}

bool displayHull()
{
	drawHull();
	is_next_case_request = false;
	is_displaying = true;
	while (!glfwWindowShouldClose(window))
	{
		if (is_next_case_request)
		{
			is_displaying = false;
			glCall(glClear(GL_COLOR_BUFFER_BIT));
			glfwSwapBuffers(window);
			return true;
		}
		glfwWaitEvents();
	}
	is_displaying = false;
	return false;
}
