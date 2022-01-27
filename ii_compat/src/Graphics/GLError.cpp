#include "GLError.h"

#include <GL/glew.h>

#include "Debug/Logging.h"

void glClearError()
{
	while (glGetError() != GL_NO_ERROR);
}

bool glLogError(const char* call, const char* file, int line)
{
	GLenum errcode;
	bool good = true;
	while ((errcode = glGetError()) != GL_NO_ERROR)
	{
		const char* msg = reinterpret_cast<const char*>(
			gluErrorString(errcode));
		WARNING("[OpenGL Error] ", file, "::", line, " ", call, " '",
			msg, "' (", errcode, ")");
		good = false;
	}
	return good;
}
