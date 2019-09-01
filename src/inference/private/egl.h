#pragma once

#include <EGL/egl.h>

namespace egl
{
    class exception : public std::exception
    {

    public:

        exception(const EGLint  s) : m_error(s)
        {

        }

        const char* what() const override
        {
            return "egl exception";
        }

    private:
        EGLint m_error;

    };

    inline void throw_if_failed(const EGLBoolean s)
    {
        if (s != EGL_TRUE)
        {
            throw exception(s);
        }
    }
}

