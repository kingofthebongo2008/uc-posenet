@echo off

echo ^<?xml version="1.0" encoding="utf-8"?^> > build_app_h.msbuild
echo ^<Project DefaultTargets="Build" ToolsVersion="15.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003"^> >> build_app_h.msbuild

ucdev_build_file_generator_r.exe --input ..\src\inference\private\ --mode h >> build_app_h.msbuild
ucdev_build_file_generator_r.exe --input ..\include\ --mode h >> build_app_h.msbuild

echo ^</Project^> >> build_app_h.msbuild

echo ^<?xml version="1.0" encoding="utf-8"?^> > build_app_cpp.msbuild
echo ^<Project DefaultTargets="Build" ToolsVersion="15.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003"^> >> build_app_cpp.msbuild

ucdev_build_file_generator_r.exe --input ..\src\inference\private\ --mode cpp >> build_app_cpp.msbuild

echo ^</Project^> >> build_app_cpp.msbuild

echo ^<?xml version="1.0" encoding="utf-8"?^> > build.vcxproj.filters
echo ^<Project DefaultTargets="Build" ToolsVersion="15.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003"^> >> build.vcxproj.filters

ucdev_build_file_generator_r.exe --type filters --input ..\src\inference\private\ --mode h >> build.vcxproj.filters
ucdev_build_file_generator_r.exe --type filters --input ..\include\ --mode h >> build.vcxproj.filters
ucdev_build_file_generator_r.exe --type filters --input ..\src\inference\private\ --mode cpp >> build.vcxproj.filters

echo ^</Project^> >> build.vcxproj.filters





