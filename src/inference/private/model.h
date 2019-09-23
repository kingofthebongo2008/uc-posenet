#pragma once

#include <assert.h>
#include <vector>
#include "tensorflow_lite_c_api.h"

struct model
{
    tensorflow_lite_c_api::model                m_model;
    tensorflow_lite_c_api::interpreter_options  m_options;
    tensorflow_lite_c_api::interpreter          m_interpreter;

    model(tensorflow_lite_c_api::model m
        , tensorflow_lite_c_api::interpreter_options  o
        , tensorflow_lite_c_api::interpreter          i) :

        m_model(std::move(m))
        , m_options(std::move(o))
        , m_interpreter(std::move(i))
    {

    }
};

struct test_model : public model
{
    test_model(tensorflow_lite_c_api::model m
        , tensorflow_lite_c_api::interpreter_options  o
        , tensorflow_lite_c_api::interpreter          i) :
        model(
            std::move(m)
            , std::move(o)
            , std::move(i))
    {

    }
};

template <typename t>
inline t make_model(const char* model_file, TfLiteDelegate* delegate = nullptr)
{
    tensorflow_lite_c_api::model                m(model_file);
    tensorflow_lite_c_api::interpreter_options  o;

    //o.set_num_threads(1);
    if (delegate)
    {
        TfLiteInterpreterOptionsAddDelegate(o, delegate);
    }
    tensorflow_lite_c_api::interpreter          i(m, o);
    i.allocate_tensors();
    return t(std::move(m), std::move(o), std::move(i));
}

enum class body_part : uint32_t
{
    nose,
    left_eye,
    right_eye,
    left_ear,
    right_ear,
    left_shoulder,
    right_shoulder,
    left_elbow,
    right_elbow,
    left_wrist,
    right_wrist,
    left_hip,
    right_hip,
    left_knee,
    right_knee,
    left_ankle,
    right_ankle
};


struct tensor_view
{
    float* m_data;

    uint32_t m_d0;
    uint32_t m_d1;
    uint32_t m_d2;
    uint32_t m_d3;

    tensor_view( float* d, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3) :
    m_data(d)
    , m_d0(d0)
    , m_d1(d1)
    , m_d2(d2)
    , m_d3(d3)
    {

    }

    size_t byte_size() const
    {
        return m_d0 * m_d1 * m_d2 * m_d3;
    }

    uint32_t offset(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3)
    {
        return a3 + m_d3 * (a2 + m_d2 * (a1 + m_d1 * (a0)));
    }

    float value(uint32_t x, uint32_t y, uint32_t z, uint32_t w)
    {
        uint32_t o = offset(x, y, z, w);
        assert(offset < byte_size());
        return *(m_data + o);
    }

    void set_value(uint32_t x, uint32_t y, uint32_t z, uint32_t w, float v)
    {
        uint32_t o = offset(x, y, z, w);
        assert(offset < byte_size());
        *(m_data + o) = v;
    }
};

tensor_view make_tensor_view( tensorflow_lite_c_api::output_tensor* d, float* data)
{
    auto dims0 = d->dim(0);
    auto dims1 = d->dim(1);
    auto dims2 = d->dim(2);
    auto dims3 = d->dim(3);
    return tensor_view(data, dims0, dims1, dims2, dims3);
}

std::vector< uint8_t > make_output_data(tensorflow_lite_c_api::output_tensor* d)
{
    std::vector<uint8_t> v(d->byte_size());
    d->copy_to_buffer(&v[0], d->byte_size());
    return v;
}

static float sigmoid(float x) 
{
    return (1.0f / (1.0f + expf(-x)));
}

struct position
{
    float m_y;
    float m_x;
};

struct float2
{
    float m_y;
    float m_x;
};

void detect()
{
    /*
    TfLiteDelegate* d;
    const TfLiteGpuDelegateOptions options = {nullptr, { 1, TFLITE_GL_OBJECT_TYPE_FASTEST, 0 }, };

    d = TfLiteGpuDelegateCreate(&options);
    
    //auto m = make_model<test_model>("data/multi_person_mobilenet_v1_075_float.tflite");
    auto m                      = make_model<test_model>("data/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite");
    
    auto inter                  = &m.m_interpreter;

    auto output                 = inter->get_output_tensor_count();
    auto intput                 = inter->get_input_tensor_count();

    auto heatmaps               = tensorflow_lite_c_api::make_output_tensor(inter, 0);
    auto offsets                = tensorflow_lite_c_api::make_output_tensor(inter, 1);
    auto displacements_forward   = tensorflow_lite_c_api::make_output_tensor(inter, 2);
    auto displacements_backward = tensorflow_lite_c_api::make_output_tensor(inter, 3);

    auto image                  = tensorflow_lite_c_api::make_input_tensor(inter, 0);
    auto dims0                  = image.dim(0);
    auto dims1                  = image.dim(1);
    auto dims2                  = image.dim(2);
    auto dims3                  = image.dim(3);

    //auto r                    = cv::imread("data/images/military_mechanic_257x353.png");
    auto r                      = cv::imread("data/images/military_mechanic_257x257_2.png");
    //auto r                        = cv::imread("data/images/posenet_sample.png");
    auto img                    = opencv::normalize(r); //stride 16
    auto w                      = opencv::width(img);
    auto h                      = opencv::height(img);
    auto s1                     = opencv::byte_size(img);
    auto buffer                 = image.byte_size();

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

    uint32_t keypoints          = heatmaps.dim(3);
    std::vector<position>       positions(keypoints);
    std::vector<float>          sigs(keypoints);

    auto width  = offsets_view.m_d2;
    auto height = offsets_view.m_d1;

    for (auto i = 0U; i < keypoints; ++i)
    {
        float       max_value   = heatmaps_view.value(0, 0, 0, i);
        uint32_t    max_column  = 0;
        uint32_t    max_row     = 0;

        for (auto h = 0U; h < height; ++h)
        {
            for (auto w = 0U; w < width; w++)
            {
                float v = sigmoid(heatmaps_view.value(0, h, w, i));
    
                if (v > max_value)
                {
                    max_value   = v;
                    max_column  = w;
                    max_row     = h;
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

        image_positions[i].m_x = (p.m_x / (width  - 1.0f)) * w + offset.m_x;
        image_positions[i].m_y = (p.m_y / (height - 1.0f)) * h + offset.m_y;
    }

    if (true)
    {
        auto&& p = image_positions;
        auto s = p.size();
        for (auto i = 0U; i < s; ++i)
        {
            cv::Scalar color(255, 255, 0);
            cv::circle(r, cv::Point(p[i].m_x, p[i].m_y), 2, color);
        }

        std::cout << "writing to result2.png" << std::endl;
        cv::imwrite("result2.png", r);
    }
	*/
}

