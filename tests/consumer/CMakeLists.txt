cmake_minimum_required(VERSION 3.22)
project(tokenizers_cpp_consumer_smoke LANGUAGES CXX)

if(DEFINED TOKENIZERS_CPP_CONSUMER_SOURCE_TREE)
  set(TOKENIZERS_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  add_subdirectory(
    "${TOKENIZERS_CPP_CONSUMER_SOURCE_TREE}"
    "${CMAKE_CURRENT_BINARY_DIR}/tokenizers_cpp"
  )
elseif(DEFINED TOKENIZERS_CPP_CONSUMER_PACKAGE_PREFIX)
  list(PREPEND CMAKE_PREFIX_PATH "${TOKENIZERS_CPP_CONSUMER_PACKAGE_PREFIX}")
  find_package(tokenizers_cpp CONFIG REQUIRED)
else()
  message(FATAL_ERROR
    "set TOKENIZERS_CPP_CONSUMER_SOURCE_TREE or TOKENIZERS_CPP_CONSUMER_PACKAGE_PREFIX"
  )
endif()

if(NOT DEFINED TOKENIZERS_CPP_CONSUMER_TOKENIZER_JSON)
  message(FATAL_ERROR "TOKENIZERS_CPP_CONSUMER_TOKENIZER_JSON is required")
endif()

add_executable(tokenizers_cpp_consumer_smoke
  consumer_smoke.cpp
)
target_link_libraries(tokenizers_cpp_consumer_smoke
  PRIVATE
    tokenizers_cpp::tokenizers_cpp
)
target_compile_features(tokenizers_cpp_consumer_smoke PRIVATE cxx_std_17)
target_compile_definitions(tokenizers_cpp_consumer_smoke
  PRIVATE
    TOKENIZERS_CPP_CONSUMER_TOKENIZER_JSON="${TOKENIZERS_CPP_CONSUMER_TOKENIZER_JSON}"
)

enable_testing()
add_test(
  NAME tokenizers_cpp_consumer_smoke
  COMMAND tokenizers_cpp_consumer_smoke
)
