cmake_minimum_required(VERSION 3.10)
project(Kepler LANGUAGES CXX)
include(CheckIPOSupported)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
check_ipo_supported(RESULT KEPLER_IPO_SUPPORTED OUTPUT KEPLER_IPO_ERROR)
if(KEPLER_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

file(GLOB SRC_FILES src/*.cpp)
list(REMOVE_ITEM SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/nnue_tool.cpp")
add_executable(kepler ${SRC_FILES})
target_include_directories(kepler PRIVATE include)

add_executable(perft tests/perft.cpp src/position.cpp src/movegen.cpp src/eval.cpp src/search.cpp src/uci.cpp src/zobrist.cpp src/nnue.cpp src/datagen.cpp)
target_include_directories(perft PRIVATE include src)

add_executable(nnue_tool src/nnue_tool.cpp)
target_include_directories(nnue_tool PRIVATE include src)
