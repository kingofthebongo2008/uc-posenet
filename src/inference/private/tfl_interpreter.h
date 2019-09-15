#pragma once

#include "tfl_base.h"
#include "tfl_exception.h"

namespace tensorflow_lite_c_api
{
    class interpreter : private not_copyable
    {

    public:

        interpreter(TfLiteModel* m, TfLiteInterpreterOptions* o)
        {
            m_value = TfLiteInterpreterCreate(m, o);
            throw_if_failed_value(m_value);
        }

        ~interpreter()
        {
            if (m_value)
            {
                TfLiteInterpreterDelete(m_value);
            }
        }

        interpreter(interpreter&& o)
        {
            m_value = o.m_value;
            o.m_value = nullptr;
        }

        interpreter& operator=(interpreter&& o)
        {
            if (&o != this)
            {
                m_value = o.m_value;
                o.m_value = nullptr;
            }
        }

        int32_t get_input_tensor_count() const
        {
            return TfLiteInterpreterGetInputTensorCount(m_value);
        }

        TfLiteTensor* get_input_tensor(int32_t index) const
        {
            return TfLiteInterpreterGetInputTensor(m_value, index);
        }

        void resize_input_tensor( int32_t input_index, const int* input_dims, int32_t input_dims_size) const
        {
            throw_if_failed(TfLiteInterpreterResizeInputTensor(m_value, input_index, input_dims, input_dims_size));
        }

        void allocate_tensors() const
        {
            throw_if_failed(TfLiteInterpreterAllocateTensors(m_value));
        }

        void invoke() const
        {
            throw_if_failed(TfLiteInterpreterInvoke(m_value));
        }

        int32_t get_output_tensor_count() const
        {
            return TfLiteInterpreterGetOutputTensorCount(m_value);
        }

        const TfLiteTensor* get_output_tensor(int32_t index) const
        {
            return TfLiteInterpreterGetOutputTensor(m_value, index);
        }

        operator TfLiteInterpreter *() const
        {
            return m_value;
        }

    private:
        TfLiteInterpreter* m_value = nullptr;
    };
}
