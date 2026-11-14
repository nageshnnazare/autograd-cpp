cmake_minimum_required(VERSION 3.20)
project(autograd_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(autograd
  main.cpp
)

target_include_directories(autograd
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Keep the binary next to the build dir root
set_target_properties(autograd
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ---------------------------------------------------------------------------
# clang-format targets
# ---------------------------------------------------------------------------
find_program(CLANG_FORMAT_EXE NAMES clang-format)

# Collect only project sources (excludes build dir)
file(GLOB_RECURSE ALL_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
)

if(CLANG_FORMAT_EXE)
  # format: rewrite files in-place
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXE} -i --style=file ${ALL_SOURCE_FILES}
    COMMENT "Running clang-format (in-place)..."
    VERBATIM
  )

  # format-check: dry-run, exits non-zero if any file would change (useful in CI)
  add_custom_target(format-check
    COMMAND ${CLANG_FORMAT_EXE} --dry-run --Werror --style=file ${ALL_SOURCE_FILES}
    COMMENT "Checking clang-format (dry-run)..."
    VERBATIM
  )
else()
  message(WARNING "clang-format not found — 'format' and 'format-check' targets are disabled.")
endif()

