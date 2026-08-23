cmake_minimum_required(VERSION 3.10)
project(Kepler LANGUAGES C CXX)
include(CheckIPOSupported)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
check_ipo_supported(RESULT KEPLER_IPO_SUPPORTED OUTPUT KEPLER_IPO_ERROR)
if(KEPLER_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

option(KEPLER_SYZYGY "Enable Syzygy tablebase probing (requires fathom source)" OFF)

set(KEPLER_SOURCES
    src/main.cpp
    src/position.cpp
    src/movegen.cpp
    src/eval.cpp
    src/search.cpp
    src/uci.cpp
    src/zobrist.cpp
    src/nnue.cpp
    src/datagen.cpp
    src/tb.cpp
)

if(KEPLER_SYZYGY)
    find_path(FATHOM_INCLUDE_DIR tbprobe.h
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/external/fathom"
        NO_DEFAULT_PATH)
    find_file(FATHOM_SOURCE tbprobe.c
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/external/fathom"
        NO_DEFAULT_PATH)
    if(NOT FATHOM_INCLUDE_DIR OR NOT FATHOM_SOURCE)
        message(FATAL_ERROR
            "KEPLER_SYZYGY=ON requires external/fathom/tbprobe.h and tbprobe.c")
    endif()
    list(APPEND KEPLER_SOURCES "${FATHOM_SOURCE}")
endif()

add_executable(kepler ${KEPLER_SOURCES})
target_include_directories(kepler PRIVATE src)
if(KEPLER_SYZYGY)
    target_compile_definitions(kepler PRIVATE KEPLER_SYZYGY)
    target_include_directories(kepler PRIVATE "${FATHOM_INCLUDE_DIR}")
endif()

add_executable(perft
    tests/perft.cpp
    src/position.cpp
    src/movegen.cpp
    src/zobrist.cpp
    src/nnue.cpp
)
target_include_directories(perft PRIVATE src)
enable_testing()
add_test(NAME perft_smoke COMMAND perft --smoke)

add_executable(nnue_tool src/nnue_tool.cpp)
target_include_directories(nnue_tool PRIVATE src)

add_executable(search_budget
    tests/search_budget.cpp
    src/position.cpp
    src/movegen.cpp
    src/eval.cpp
    src/search.cpp
    src/zobrist.cpp
    src/nnue.cpp
    src/tb.cpp
)
target_include_directories(search_budget PRIVATE src)
add_test(NAME search_node_budget COMMAND search_budget)

add_executable(eval_consistency
    tests/eval_consistency.cpp
    src/position.cpp
    src/movegen.cpp
    src/eval.cpp
    src/zobrist.cpp
    src/nnue.cpp
)
target_include_directories(eval_consistency PRIVATE src)
add_test(
    NAME eval_incremental_consistency
    COMMAND eval_consistency "${CMAKE_CURRENT_SOURCE_DIR}/models/kepler_baseline_pst_v1.nnue")

add_executable(eval_blend
    tests/eval_blend.cpp
    src/position.cpp
    src/movegen.cpp
    src/eval.cpp
    src/zobrist.cpp
    src/nnue.cpp
)
target_include_directories(eval_blend PRIVATE src)
add_test(
    NAME eval_blend_configuration
    COMMAND eval_blend "${CMAKE_CURRENT_SOURCE_DIR}/models/kepler_baseline_pst_v1.nnue")

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME uci_model_persistence
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/uci_model_persistence.py"
                "$<TARGET_FILE:kepler>"
                "$<TARGET_FILE:nnue_tool>"
                "${CMAKE_CURRENT_BINARY_DIR}/uci_persistence_test.nnue"
                "${CMAKE_CURRENT_SOURCE_DIR}/models/kepler_baseline_pst_v1.nnue")
endif()
