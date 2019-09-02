#include "pch.h"

#include <iostream>
#include <fstream>

#include "tensorflow_lite_c_api.h"

#include <tensorflow/lite/delegates/gpu/gl_delegate.h>

#include "opencv_bridge.h"

#include "egl.h"

#include <util/EGLWindow.h>
#include <util/OSWindow.h>
#include <util/Event.h>
#include "util/shader_utils.h"
#include <common/system_utils.h>

class Win32Library : public angle::Library
{
public:
	Win32Library()
	{
		mModule = LoadLibrary(L"libEGL.dll");
	}

	~Win32Library() override
	{
		if (mModule)
		{
			FreeLibrary(mModule);
		}
	}

	void* getSymbol(const char* symbolName) override
	{
		if (!mModule)
		{
			return nullptr;
		}

		return reinterpret_cast<void*>(GetProcAddress(mModule, symbolName));
	}

	void* getNative() const override { return reinterpret_cast<void*>(mModule); }

private:

	HMODULE mModule = nullptr;
};

struct EGLWindowDeleter
{
	void operator()(void* d)
	{
		EGLWindow* p = reinterpret_cast<EGLWindow*>(d);
		EGLWindow::Delete(&p);
	}
};

struct OSWindowDeleter
{
	void operator()(void* d)
	{
		OSWindow* p = reinterpret_cast<OSWindow*>(d);
		OSWindow::Delete(&p);
	}
};

ConfigParameters make_config_parameters()
{
	ConfigParameters configParams;

	configParams.redBits = 8;
	configParams.greenBits = 8;
	configParams.blueBits = 8;
	configParams.alphaBits = 8;
	configParams.depthBits = 0;
	configParams.stencilBits = 0;

	return configParams;
}

EGLPlatformParameters make_platform_parameters()
{
	EGLPlatformParameters mPlatformParams;

	mPlatformParams.majorVersion = 3;
	mPlatformParams.minorVersion = 1;

	return mPlatformParams;
}

int32_t main(int32_t, char*[])
{ 
	std::unique_ptr<EGLWindow, EGLWindowDeleter> egl(EGLWindow::New(3, 1), EGLWindowDeleter());
	std::unique_ptr<OSWindow, OSWindowDeleter> w(OSWindow::New(), OSWindowDeleter());

	Win32Library library;

	w->initialize("main window", 1600, 900);

	bool init	= egl->initializeGL(w.get(), &library, make_platform_parameters(), make_config_parameters());

	constexpr char kVS[] = R"(attribute vec4 vPosition;
void main()
{
    gl_Position = vPosition;
})";

	constexpr char kFS[] = R"(precision mediump float;
void main()
{
    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
})";

	GLuint mProgram = CompileProgram(kVS, kFS);
	if (!mProgram)
	{
		return false;
	}


	if (init)
	{
		w->setVisible(true);

		bool running = true;
		while (running)
		{
			// Clear events that the application did not process from this frame
			Event event;
			while (w->popEvent(&event))
			{
				// If the application did not catch a close event, close now
				if (event.Type == Event::EVENT_CLOSED)
				{
					running = false;
				}
			}

			if (!running)
			{
				break;
			}

			GLfloat vertices[] = 
			{
				0.0f, 0.5f, 0.0f, -0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f,
			};

			glViewport(0, 0, 1600, 900);
			glClearColor(0.0, 0, 0, 0);
			glClear(GL_COLOR_BUFFER_BIT);

			// Use the program object
			glUseProgram(mProgram);

			// Load the vertex data
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, vertices);
			glEnableVertexAttribArray(0);

			glDrawArrays(GL_TRIANGLES, 0, 3);

			egl->swap();
			w->messageLoop();
		}

		w->destroy();
		return 0;
	}
	else
	{
		return 1;
	}
 }


    