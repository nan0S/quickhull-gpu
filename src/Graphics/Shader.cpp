#include "Shader.h"

#include <fstream>
#include <sstream>

#include "Utils/Log.h"
#include "Graphics/Error.h"

namespace Graphics
{
   namespace
   {
      const char* shaderTypeToString(GLenum shader_type)
      {
         switch (shader_type)
         {
            case GL_VERTEX_SHADER:
               return "VERTEX";
            case GL_FRAGMENT_SHADER:
               return "FRAGMENT";
            default:
               return "UNKNOWN";
         }
      }

      GLuint compile(GLenum shader_type, const char* source)
      {
         GL_CALL(GLuint id = glCreateShader(shader_type));
         GL_CALL(glShaderSource(id, 1, &source, NULL));
         GL_CALL(glCompileShader(id));

         GLint success;
         GL_CALL(glGetShaderiv(id, GL_COMPILE_STATUS, &success));
         if (!success)
         {
            GLint length;
            GL_CALL(glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length));
            GLchar* msg = reinterpret_cast<GLchar*>(alloca(length * sizeof(GLchar)));
            GL_CALL(glGetShaderInfoLog(id, length, &length, msg));
            ERROR("Shader (", shaderTypeToString(shader_type), ") compilation error: '", msg, "'.");
         }

         return id;
      }
   }

   GLuint compileShader(const char* vertex_source, const char* fragment_source)
   {
      GLuint vs = compile(GL_VERTEX_SHADER, vertex_source);
      GLuint fs = compile(GL_FRAGMENT_SHADER, fragment_source);

      GL_CALL(GLuint program = glCreateProgram());
      GL_CALL(glAttachShader(program, vs));
      GL_CALL(glAttachShader(program, fs));
      GL_CALL(glLinkProgram(program));

      GLint success;
      GL_CALL(glGetProgramiv(program, GL_LINK_STATUS, &success));
      if (!success)
      {
         GLint length;
         GL_CALL(glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length));
         GLchar* msg = reinterpret_cast<GLchar*>(alloca(length * sizeof(GLchar)));
         GL_CALL(glGetProgramInfoLog(program, length, &length, msg));
         ERROR("Program link error: '{}'.", msg);
      }

      GL_CALL(glDeleteShader(vs));
      GL_CALL(glDeleteShader(fs));

      return program;
   }
}
