#pragma once

#include "tfl_base.h"


#include <exception>
#include <string>

namespace tensorflow_lite_c_api
{
    class exception : public std::exception
    {

    public:

    exception(const TfLiteStatus  s) : m_status(s)
    {

    }

    const char* what() const override
    {
        return "tensorflow lite exception";
    }

    private:
		TfLiteStatus m_status;
        
    };

    inline void throw_if_failed( const TfLiteStatus s )
    {
        if (s != kTfLiteOk)
        {
            throw exception(s);
        }
    }

    template <typename T>
    inline void throw_if_failed_value(const T* v)
    {
        if (v == nullptr)
        {
            throw exception(kTfLiteError);
        }
    }
}
