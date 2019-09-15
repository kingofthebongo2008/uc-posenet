#pragma once

#include "tfl_base.h"
#include "tfl_exception.h"

namespace tensorflow_lite_c_api
{
    class interpreter_options : private not_copyable
    {

    public:

        interpreter_options()
        {
            m_value = TfLiteInterpreterOptionsCreate();
            throw_if_failed_value(m_value);
        }

        ~interpreter_options()
        {
            if (m_value)
            {
				TfLiteInterpreterOptionsDelete(m_value);
            }
        }

        interpreter_options(interpreter_options&& o)
        {
            m_value = o.m_value;
            o.m_value = nullptr;
        }

        interpreter_options& operator=(interpreter_options&& o)
        {
            if (&o != this)
            {
                m_value = o.m_value;
                o.m_value = nullptr;
            }
        }

        void set_num_threads(int32_t threads)
        {
			TfLiteInterpreterOptionsSetNumThreads(m_value, threads);
        }
        
        operator TfLiteInterpreterOptions*() const
        {
            return m_value;
        }

    private:
		TfLiteInterpreterOptions* m_value = nullptr;
    };
}
