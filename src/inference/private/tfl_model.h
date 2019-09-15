#pragma once

#include "tfl_base.h"
#include "tfl_exception.h"

namespace tensorflow_lite_c_api
{
    class model : private not_copyable
    {

    public:

        model(const void* d, size_t s)
        {
            m_value = TfLiteModelCreate(d, s);
            throw_if_failed_value(m_value);
        }

        model(const char* file_name)
        {
            m_value = TfLiteModelCreateFromFile(file_name);
            throw_if_failed_value(m_value);
        }

        ~model()
        {
            if (m_value)
            {
				TfLiteModelDelete(m_value);
            }
        }

        model(model&& o)
        {
            m_value = o.m_value;
            o.m_value = nullptr;
        }

        model& operator=(model&& o)
        {
            if (&o != this)
            {
                m_value = o.m_value;
                o.m_value = nullptr;
            }
        }

        operator TfLiteModel *() const
        {
            return m_value;
        }

    private:
		TfLiteModel* m_value = nullptr;
    };
}
