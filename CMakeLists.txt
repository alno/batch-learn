cmake_minimum_required(VERSION 2.6)
project(batch-learn)

find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED program_options)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

file(GLOB SRCS src/*.cpp src/**/*.cpp)
include_directories(include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -std=c++11 -march=native")

add_executable(${PROJECT_NAME} ${SRCS})
target_link_libraries(${PROJECT_NAME} boost_program_options)
