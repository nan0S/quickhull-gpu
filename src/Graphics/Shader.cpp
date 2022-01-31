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
         glCall(GLuint id = glCreateShader(shader_type));
         glCall(glShaderSource(id, 1, &source, NULL));
         glCall(glCompileShader(id));

         GLint success;
         glCall(glGetShaderiv(id, GL_COMPILE_STATUS, &success));
         if (!success)
         {
            GLint length;
            glCall(glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length));
            GLchar* msg = reinterpret_cast<GLchar*>(alloca(length * sizeof(GLchar)));
            glCall(glGetShaderInfoLog(id, length, &length, msg));
            ERROR("Shader (", shaderTypeToString(shader_type), ") compilation error: '", msg, "'.");
         }

         return id;
      }
   }

   GLuint compileShader(const char* vertex_source, const char* fragment_source)
   {
      GLuint vs = compile(GL_VERTEX_SHADER, vertex_source);
      GLuint fs = compile(GL_FRAGMENT_SHADER, fragment_source);

      glCall(GLuint program = glCreateProgram());
      glCall(glAttachShader(program, vs));
      glCall(glAttachShader(program, fs));
      glCall(glLinkProgram(program));

      GLint success;
      glCall(glGetProgramiv(program, GL_LINK_STATUS, &success));
      if (!success)
      {
         GLint length;
         glCall(glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length));
         GLchar* msg = reinterpret_cast<GLchar*>(alloca(length * sizeof(GLchar)));
         glCall(glGetProgramInfoLog(program, length, &length, msg));
         ERROR("Program link error: '{}'.", msg);
      }

      glCall(glDeleteShader(vs));
      glCall(glDeleteShader(fs));

      return program;
   }
}
