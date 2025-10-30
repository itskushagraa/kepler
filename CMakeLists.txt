cmake_minimum_required(VERSION 3.10)
project(Kepler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

file(GLOB SRC_FILES src/*.cpp)
add_executable(kepler ${SRC_FILES})
target_include_directories(kepler PRIVATE include)

add_executable(perft tests/perft.cpp src/position.cpp src/movegen.cpp src/eval.cpp src/search.cpp src/uci.cpp)
target_include_directories(perft PRIVATE include src)