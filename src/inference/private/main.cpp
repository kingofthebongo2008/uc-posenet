#include "pch.h"

#include <iostream>
#include <fstream>

#include "egl.h"

#include "model.h"
#include "opencv_bridge.h"

/*
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
*/


int32_t main(int32_t, char*[])
{ 
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		std::cerr << "Cannot open camera" << "\n";
		return -1;
	}

	cap.set(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH, 257);
	cap.set(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT, 353);
	cap.set(cv::VideoCaptureProperties::CAP_PROP_CONVERT_RGB, 1);

	auto m = make_model<test_model>("data/multi_person_mobilenet_v1_075_float.tflite");
	//auto m = make_model<test_model>("data/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite");

	auto inter = &m.m_interpreter;

	auto output = inter->get_output_tensor_count();
	auto intput = inter->get_input_tensor_count();
	auto image	= tensorflow_lite_c_api::make_input_tensor(inter, 0);

	auto heatmaps = tensorflow_lite_c_api::make_output_tensor(inter, 0);
	auto offsets = tensorflow_lite_c_api::make_output_tensor(inter, 1);
	auto displacements_forward = tensorflow_lite_c_api::make_output_tensor(inter, 2);
	auto displacements_backward = tensorflow_lite_c_api::make_output_tensor(inter, 3);

	auto dims0 = image.dim(0);
	auto dims1 = image.dim(1);
	auto dims2 = image.dim(2);
	auto dims3 = image.dim(3);

	for (;;)
	{
		cv::Mat frame;
		// Get a new frame from camera
		cap >> frame;

		//auto r = cv::imread("data/images/military_mechanic_257x257_2.png");

		auto r0 = opencv::resample(frame, 353, 257);
		auto img = opencv::normalize(r0); //stride 16
		auto w = opencv::width(img);
		auto h = opencv::height(img);
		auto s1 = opencv::byte_size(img);

		auto buffer = image.byte_size();

		image.copy_from_buffer(img.data, opencv::byte_size(img));
		inter->invoke();


		std::vector< uint8_t >      heatmaps_data = make_output_data(&heatmaps);
		tensor_view                 heatmaps_view = make_tensor_view(&heatmaps, reinterpret_cast<float*>(&heatmaps_data[0]));

		std::vector< uint8_t >      offsets_data = make_output_data(&offsets);
		tensor_view                 offsets_view = make_tensor_view(&offsets, reinterpret_cast<float*>(&offsets_data[0]));

		std::vector< uint8_t >      displacements_forward_data = make_output_data(&displacements_forward);
		tensor_view                 displacements_forward_view = make_tensor_view(&displacements_forward, reinterpret_cast<float*>(&displacements_forward_data[0]));

		std::vector< uint8_t >      displacements_backward_data = make_output_data(&displacements_backward);
		tensor_view                 displacements_backward_view = make_tensor_view(&displacements_backward, reinterpret_cast<float*>(&displacements_backward_data[0]));

		uint32_t keypoints = heatmaps.dim(3);
		std::vector<position>       positions(keypoints);
		std::vector<float>          sigs(keypoints);

		auto width = offsets_view.m_d2;
		auto height = offsets_view.m_d1;

		for (auto i = 0U; i < keypoints; ++i)
		{
			float       max_value = heatmaps_view.value(0, 0, 0, i);
			uint32_t    max_column = 0;
			uint32_t    max_row = 0;

			for (auto h = 0U; h < height; ++h)
			{
				for (auto w = 0U; w < width; w++)
				{
					float v = sigmoid(heatmaps_view.value(0, h, w, i));

					if (v > max_value)
					{
						max_value = v;
						max_column = w;
						max_row = h;
					}
				}
			}
			positions[i] = { max_row, max_column };
			sigs[i] = max_value;
		}

		std::vector<float2>     image_offsets(keypoints);
		std::vector<position>   image_positions(keypoints);
		std::vector<float>      confidence_scores(keypoints);

		for (auto i = 0U; i < keypoints; ++i)
		{
			auto p = positions[i];

			float2 offset;

			offset.m_y = offsets_view.value(0, p.m_y, p.m_x, i);
			offset.m_x = offsets_view.value(0, p.m_y, p.m_x, i + keypoints);

			image_offsets[i] = offset;

			image_positions[i].m_x = (p.m_x / (width - 1.0f)) * w + offset.m_x;
			image_positions[i].m_y = (p.m_y / (height - 1.0f)) * h + offset.m_y;
		}

		if (true)
		{
			auto&& p = image_positions;
			auto s = p.size();
			for (auto i = 0U; i < s; ++i)
			{
				cv::Scalar color(255, 255, 0);
				cv::circle(r0, cv::Point(p[i].m_x, p[i].m_y), 2, color);
			}
		}

		// Display images    
		imshow("frame", r0);
		if (cv::waitKey(30) >= 0) break;
	}
	// Deinitialize camera in the VideoCapture destructor
	return 0;
	
	//Visualization code
	/*
	std::unique_ptr<EGLWindow, EGLWindowDeleter> egl(EGLWindow::New(3, 1), EGLWindowDeleter());
	st8d::unique_ptr<OSWindow, OSWindowDeleter> w(OSWindow::New(), OSWindowDeleter());

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
	*/
 }


    