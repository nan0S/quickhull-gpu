#include <string>
#include <vector>
#include <cstring>
#include <cassert>

#define GLFW_INCLUDE_NONE
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "GPU/GPU.cuh"
#include "CPU/CPU.h"
#include "Graphics/Error.h"
#include "Graphics/Shader.h"
#include "Utils/Log.h"
#include "Config.h"

/* enums and structs */
enum class ComputeMode
{
   NONE, GPU, CPU
};

class comma_numpunct : public std::numpunct<char>
{
protected:
   char do_thousands_sep() const { return ','; }
   std::string do_grouping() const { return "\03"; }
};

struct AppState
{
    GLFWwindow* window;
    GLint point_size_loc;
    GLint color_loc;
    int n_points;
    int hull_count;
};

/* forward declarations */
void glfwErrorHandler(int errCode, const char* desc);
void windowResizeHandler(GLFWwindow*, int width, int height);
void windowKeyInputHandler(GLFWwindow*, int key, int, int action, int);
bool displayHull(AppState* state); // returns whether we should continue with the next test case
void drawHull(AppState* state);

/* constants */
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
"Press ESCAPE/Q to exit.\n";

const char* VERTEX_SHADER_SOURCE =
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

const char* FRAGMENT_SHADER_SOURCE =
R"(
#version 330 core

in vec4 v_Color;
out vec4 f_Color;

void main()
{
   f_Color = v_Color;
})";

constexpr float HULL_POINT_SIZE = 7.f;
constexpr float HULL_POINT_COLOR[4] = { 0.f, 1.f, 0.f, 1.f };
constexpr float NORMAL_POINT_SIZE = 1.f;
constexpr float NORMAL_POINT_COLOR[4] = { 1.f, 1.f, 1.f, 1.f };
constexpr float LINE_WIDTH = 0.7f;
constexpr float LINE_COLOR[4] = { 0.f, 1.f, 0.f, 0.5f };

constexpr int WIDTH = 800;
constexpr int HEIGHT = 800;

constexpr Config DEFAULT_CONFIG{
   DatasetType::DISC, 1234, false };

/* variables */
bool is_next_case_request;
bool is_redraw_request;

int main(int argc, const char* argv[])
{
   {
      std::locale comma_locale(std::locale(), new comma_numpunct);
      std::cout.imbue(comma_locale);
      std::cerr.imbue(comma_locale);
   }

   assert(argc > 0);
   ComputeMode compute_mode = ComputeMode::NONE;
   Config config = DEFAULT_CONFIG;

   /* Parse program arguments */
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

   if (compute_mode == ComputeMode::NONE)
      ERROR("Please specify compute mode with --gpu or --cpu.");
   if (argc < 2)
      ERROR("Invalid number of arguments.");

   argc = pivot_idx;
   std::vector<int> num_points(argc - 1);
   for (int i = 1; i < argc; ++i)
   {
      int n = std::stoi(argv[i]);
      if (n <= 0)
         ERROR("Number of points has to be positive.");
      if (n < 3)
         ERROR("Number of points has to be at least 3.");
      num_points[i - 1] = n;
   }

   AppState state;

   /* Initialize OpenGL (GLFW and glew). */
   glfwSetErrorCallback(glfwErrorHandler);
   if (!glfwInit())
      ERROR("Failed to initialize GLFW.");

   glfwWindowHint(GLFW_SAMPLES, 4);
   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
   glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

   state.window = glfwCreateWindow(WIDTH, HEIGHT, "Convex Hull", NULL, NULL);
   if (!state.window)
   {
      glfwTerminate();
      ERROR("Failed to create window.");
   }

   glfwMakeContextCurrent(state.window);
   windowResizeHandler(state.window, WIDTH, HEIGHT);
   glfwSetFramebufferSizeCallback(state.window, windowResizeHandler);
   glfwSetKeyCallback(state.window, windowKeyInputHandler);

   if (glewInit() != GLEW_OK)
   {
      glfwTerminate();
      ERROR("Failed to initialize GLEW.");
   }

   GL_CALL(glClearColor(0, 0, 0, 1));
   GL_CALL(glEnable(GL_VERTEX_PROGRAM_POINT_SIZE));
   GL_CALL(glEnable(GL_LINE_SMOOTH));

   GLuint vao;
   GL_CALL(glGenVertexArrays(1, &vao));
   GL_CALL(glBindVertexArray(vao));

   GLuint vbo;
   GL_CALL(glGenBuffers(1, &vbo));
   GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, vbo));
   GL_CALL(glEnableVertexAttribArray(0));
   GL_CALL(glEnableVertexAttribArray(1));

   GLuint shader = Graphics::compileShader(VERTEX_SHADER_SOURCE,
                                           FRAGMENT_SHADER_SOURCE);
   GL_CALL(glUseProgram(shader));
   GL_CALL(state.point_size_loc = glGetUniformLocation(shader, "pointSize"));
   GL_CALL(state.color_loc = glGetUniformLocation(shader, "color"));

   GL_CALL(glLineWidth(LINE_WIDTH));
   GL_CALL(glEnable(GL_BLEND));
   GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

   GL_CALL(glClear(GL_COLOR_BUFFER_BIT));
   glfwSwapBuffers(state.window);

   /* Print configuration information. */
   print(INSTRUCTIONS_STR);
   switch (compute_mode)
   {
      case ComputeMode::GPU:
         print("GPU mode selected.");
         break;
      case ComputeMode::CPU:
         print("CPU mode selected.");
         break;
        default:
            assert(false);
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

   /* Perform computations. */
   switch (compute_mode)
   {
      case ComputeMode::GPU:
      {
         GPU::init(&config, num_points, vbo);
         for (int n : num_points)
         {
            state.n_points = n;
            state.hull_count = GPU::calculate(n);
            if (!displayHull(&state))
               break;
         }
         GPU::cleanup();
         break;
      }
      case ComputeMode::CPU:
      {
         CPU::init(&config, num_points);
         for (int n : num_points)
         {
            state.n_points = n;
            state.hull_count = CPU::calculate(n);
            if (!displayHull(&state))
               break;
         }
         CPU::cleanup();
         break;
      }
      default:
         assert(false);
   }

   /* Cleanup. */
   GL_CALL(glDeleteProgram(shader));
   GL_CALL(glDeleteBuffers(1, &vbo));
   GL_CALL(glDeleteVertexArrays(1, &vao));
   glfwTerminate();

   return 0;
}

void glfwErrorHandler(int errCode, const char* desc)
{
   WARNING("[GLFW Error] ", desc, " ", errCode);
}

void windowResizeHandler(GLFWwindow*, int width, int height)
{
   int minSize = std::min(width, height);
   constexpr float borderFactor = 0.05f;
   int border = static_cast<int>(borderFactor * minSize);
   int corner = minSize - 2 * border;
   GL_CALL(glViewport(border, border, corner, corner));
   is_redraw_request = true;
}

void windowKeyInputHandler(GLFWwindow* window, int key, int,
   int action, int)
{
   switch (key)
   {
      case GLFW_KEY_ESCAPE:
      case GLFW_KEY_Q:
         glfwSetWindowShouldClose(window, GLFW_TRUE);
         break;
      case GLFW_KEY_ENTER:
      case GLFW_KEY_SPACE:
         if (action == GLFW_PRESS)
            is_next_case_request = true;
         break;
   }
}

bool displayHull(AppState* state)
{
   glfwPollEvents();
   if (glfwWindowShouldClose(state->window))
      return false;

   drawHull(state);
   is_next_case_request = false;
   is_redraw_request = false;

   do
   {
      glfwWaitEvents();
      if (is_next_case_request)
      {
         GL_CALL(glClear(GL_COLOR_BUFFER_BIT));
         glfwSwapBuffers(state->window);
         return true;
      }
      if (is_redraw_request)
         drawHull(state);
   } while (!glfwWindowShouldClose(state->window));

   return false;
}

void drawHull(AppState* state)
{
   GL_CALL(glClear(GL_COLOR_BUFFER_BIT));

   GL_CALL(glUniform1f(state->point_size_loc, NORMAL_POINT_SIZE));
   GL_CALL(glUniform4fv(state->color_loc, 1, NORMAL_POINT_COLOR));
   GL_CALL(glDrawArrays(GL_POINTS,
                       state->hull_count,
                       state->n_points - state->hull_count));
   GL_CALL(glUniform4fv(state->color_loc, 1, LINE_COLOR));
   GL_CALL(glDrawArrays(GL_LINE_LOOP, 0, state->hull_count));
   GL_CALL(glUniform1f(state->point_size_loc, HULL_POINT_SIZE));
   GL_CALL(glUniform4fv(state->color_loc, 1, HULL_POINT_COLOR));
   GL_CALL(glDrawArrays(GL_POINTS, 0, state->hull_count));

   glfwSwapBuffers(state->window);
}

