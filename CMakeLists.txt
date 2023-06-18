cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(SVM_cpp VERSION 1.0)

find_package(Python3 3.11 COMPONENTS Interpreter Development NumPy REQUIRED)
include_directories((PYTHON_INCLUDE_DIRS) include)
add_executable(svm_showcase src/main.cpp include/svm.h)
target_link_libraries(svm_showcase Python3::NumPy)