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

int32_t main(int32_t, char*[])
{ 
	EGLWindow* egl = EGLWindow::New(3, 1);
	OSWindow* w = OSWindow::New();

	ConfigParameters configParams;

	configParams.redBits		= 8;
	configParams.greenBits		= 8;
	configParams.blueBits		= 8;
	configParams.alphaBits		= 8;
	configParams.depthBits		= 24;
	configParams.stencilBits	= 8;

	EGLPlatformParameters mPlatformParams;

	mPlatformParams.majorVersion = 3;
	mPlatformParams.minorVersion = 1;
	
	w->initialize("test", 1600, 900);
	w->setVisible(true);

	Win32Library library;

	bool init	= egl->initializeGL(w, &library, mPlatformParams, configParams);

	egl->setSwapInterval(0);
	egl->makeCurrent();

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

		glViewport(0, 0, 1600, 900);
		glClearColor(0.5, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT);

		egl->swap();
		w->messageLoop();

		//prevTime = elapsedTime;
	}

	EGLWindow::Delete(&egl);
	w->destroy();
	OSWindow::Delete(&w);

    return 0;
 }


    