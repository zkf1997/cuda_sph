cmake_minimum_required( VERSION 3.10.0 )
project( sph LANGUAGES CXX CUDA)

# external glfw
set( GLFW_BUILD_DOCS OFF CACHE INTERNAL "" )
set( GLFW_BUILD_TESTS OFF CACHE INTERNAL "" )
set( GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "" )
add_subdirectory( external/glfw )

# make sure we have OpenMP C++ support
find_package( OpenMP REQUIRED CXX )

# external includes
include_directories( . )
include_directories( SYSTEM "external/glm" )
include_directories( SYSTEM "external/glad/include" )

# application code
# include_directories( "include" )
# add_executable(
#     sph-benchmark
#     "src/main.cpp"
#     "external/glad/src/glad.c" )
# target_compile_features(
#     sph-benchmark
#     PUBLIC cxx_std_11 )
# target_link_libraries(
#     sph-benchmark
#     PRIVATE glfw
#     PRIVATE OpenMP::OpenMP_CXX )

add_executable(
    sph-cuda
    "cuSPH-test.cu" 
    "external/glad/src/glad.c")
target_compile_features(
    sph-cuda
    PUBLIC cxx_std_11 )
target_link_libraries(
    sph-cuda
    PRIVATE glfw
    PRIVATE OpenMP::OpenMP_CXX)
set_target_properties(sph-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)