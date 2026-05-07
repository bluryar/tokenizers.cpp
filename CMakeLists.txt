cmake_minimum_required(VERSION 3.22)
project(tokenizers_cpp VERSION 0.1.0 LANGUAGES CXX)

option(TOKENIZERS_CPP_BUILD_TESTS "Build tokenizers.cpp tests" ON)
option(TOKENIZERS_CPP_BUILD_EXAMPLES "Build tokenizers.cpp examples" ${PROJECT_IS_TOP_LEVEL})
option(TOKENIZERS_CPP_BUILD_BENCHMARKS "Build tokenizers.cpp microbenchmarks" OFF)
option(TOKENIZERS_CPP_FETCH_DEPS "Fetch missing third-party dependencies" ON)
option(TOKENIZERS_CPP_ICU_VENDOR "Require ICU from a project-owned vendored install" ON)
option(TOKENIZERS_CPP_ICU_SHARED "Allow ICU shared libraries for development builds" OFF)
set(
  TOKENIZERS_CPP_ICU_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/icu4c-install"
  CACHE PATH
  "Path to a vendored ICU4C install prefix"
)
set(
  TOKENIZERS_CPP_HF_TEST_DATA_DIR_DEFAULT
  "${CMAKE_CURRENT_SOURCE_DIR}/hf-internal-testing/tokenizers-test-data"
)
set(
  TOKENIZERS_CPP_HF_TEST_DATA_DIR_OLD_DEFAULT
  "${CMAKE_CURRENT_SOURCE_DIR}/../../hf-internal-testing/tokenizers-test-data"
)
set(
  TOKENIZERS_CPP_HF_TEST_DATA_DIR
  "${TOKENIZERS_CPP_HF_TEST_DATA_DIR_DEFAULT}"
  CACHE PATH
  "Path to the local hf-internal-testing/tokenizers-test-data checkout"
)
if(
  TOKENIZERS_CPP_HF_TEST_DATA_DIR STREQUAL TOKENIZERS_CPP_HF_TEST_DATA_DIR_OLD_DEFAULT
  AND EXISTS "${TOKENIZERS_CPP_HF_TEST_DATA_DIR_DEFAULT}"
)
  set(
    TOKENIZERS_CPP_HF_TEST_DATA_DIR
    "${TOKENIZERS_CPP_HF_TEST_DATA_DIR_DEFAULT}"
    CACHE PATH
    "Path to the local hf-internal-testing/tokenizers-test-data checkout"
    FORCE
  )
endif()
set(TOKENIZERS_CPP_BUILD_HF_TEST_DATA_TESTS_DEFAULT OFF)
if(EXISTS "${TOKENIZERS_CPP_HF_TEST_DATA_DIR}")
  set(TOKENIZERS_CPP_BUILD_HF_TEST_DATA_TESTS_DEFAULT ON)
endif()
option(
  TOKENIZERS_CPP_BUILD_HF_TEST_DATA_TESTS
  "Build parity tests that require hf-internal-testing/tokenizers-test-data"
  ${TOKENIZERS_CPP_BUILD_HF_TEST_DATA_TESTS_DEFAULT}
)
if(TOKENIZERS_CPP_BUILD_HF_TEST_DATA_TESTS AND NOT EXISTS "${TOKENIZERS_CPP_HF_TEST_DATA_DIR}")
  message(FATAL_ERROR
    "TOKENIZERS_CPP_BUILD_HF_TEST_DATA_TESTS=ON requires "
    "TOKENIZERS_CPP_HF_TEST_DATA_DIR to point at a local "
    "hf-internal-testing/tokenizers-test-data checkout."
  )
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(DEFINED CACHE{TOKENIZERS_CPP_USE_ICU})
  if(NOT TOKENIZERS_CPP_USE_ICU)
    message(FATAL_ERROR
      "TOKENIZERS_CPP_USE_ICU=OFF has been retired from the supported build matrix. "
      "The build now requires the vendored ICU backend."
    )
  endif()
  unset(TOKENIZERS_CPP_USE_ICU CACHE)
endif()
if(DEFINED CACHE{TOKENIZERS_CPP_BOOTSTRAP_BUILTIN_UNICODE})
  if(TOKENIZERS_CPP_BOOTSTRAP_BUILTIN_UNICODE)
    message(FATAL_ERROR
      "TOKENIZERS_CPP_BOOTSTRAP_BUILTIN_UNICODE has been removed. "
      "The build now requires the vendored ICU backend."
    )
  endif()
  unset(TOKENIZERS_CPP_BOOTSTRAP_BUILTIN_UNICODE CACHE)
endif()

include(cmake/Dependencies.cmake)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Threads REQUIRED)

set(TOKENIZERS_CPP_SOURCES
  src/tokenizer.cpp
  src/unicode_backend_icu.cpp
)

add_library(tokenizers_cpp
  ${TOKENIZERS_CPP_SOURCES}
)

add_library(tokenizers_cpp::tokenizers_cpp ALIAS tokenizers_cpp)

target_include_directories(tokenizers_cpp
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/aho_corasick>
)

target_link_libraries(tokenizers_cpp
  PRIVATE
    Threads::Threads
    tokenizers_cpp_icu4c::i18n
    tokenizers_cpp_icu4c::uc
    tokenizers_cpp_icu4c::data
)
target_compile_definitions(tokenizers_cpp
  PRIVATE
    U_STATIC_IMPLEMENTATION
)

target_compile_features(tokenizers_cpp PUBLIC cxx_std_17)

get_filename_component(
  TOKENIZERS_CPP_ICU_UC_INSTALL_NAME
  "${TOKENIZERS_CPP_ICU_UC_LIBRARY}"
  NAME
)
get_filename_component(
  TOKENIZERS_CPP_ICU_I18N_INSTALL_NAME
  "${TOKENIZERS_CPP_ICU_I18N_LIBRARY}"
  NAME
)
get_filename_component(
  TOKENIZERS_CPP_ICU_DATA_INSTALL_NAME
  "${TOKENIZERS_CPP_ICU_DATA_LIBRARY}"
  NAME
)

set(TOKENIZERS_CPP_INSTALL_CMAKEDIR
  "${CMAKE_INSTALL_LIBDIR}/cmake/tokenizers_cpp"
)

install(
  TARGETS tokenizers_cpp
  EXPORT tokenizers_cppTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(
  DIRECTORY include/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(
  DIRECTORY "${TOKENIZERS_CPP_ICU_ROOT}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/tokenizers_cpp/icu"
)
install(
  FILES
    "${TOKENIZERS_CPP_ICU_UC_LIBRARY}"
    "${TOKENIZERS_CPP_ICU_I18N_LIBRARY}"
    "${TOKENIZERS_CPP_ICU_DATA_LIBRARY}"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/tokenizers_cpp/icu"
)
install(
  EXPORT tokenizers_cppTargets
  NAMESPACE tokenizers_cpp::
  DESTINATION "${TOKENIZERS_CPP_INSTALL_CMAKEDIR}"
)

configure_package_config_file(
  cmake/tokenizers_cppConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/tokenizers_cppConfig.cmake"
  INSTALL_DESTINATION "${TOKENIZERS_CPP_INSTALL_CMAKEDIR}"
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/tokenizers_cppConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/tokenizers_cppConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/tokenizers_cppConfigVersion.cmake"
  DESTINATION "${TOKENIZERS_CPP_INSTALL_CMAKEDIR}"
)
install(
  DIRECTORY examples/
  DESTINATION "${CMAKE_INSTALL_DOCDIR}/examples"
)
install(
  FILES
    README.md
    LICENSE
    NOTICE
    THIRD_PARTY_NOTICES.md
    CONTRIBUTING.md
    SECURITY.md
    docs/api-stability.md
    docs/integration.md
    docs/core-inference-api-translation-complete.md
    docs/open-source-checklist.md
  DESTINATION "${CMAKE_INSTALL_DOCDIR}"
)

if(TOKENIZERS_CPP_BUILD_EXAMPLES)
  add_executable(tokenizers_cpp_example_basic_encode_decode
    examples/basic_encode_decode.cpp
  )
  target_link_libraries(tokenizers_cpp_example_basic_encode_decode
    PRIVATE
      tokenizers_cpp::tokenizers_cpp
  )

  add_executable(tokenizers_cpp_example_batch_and_padding
    examples/batch_and_padding.cpp
  )
  target_link_libraries(tokenizers_cpp_example_batch_and_padding
    PRIVATE
      tokenizers_cpp::tokenizers_cpp
  )

  add_executable(tokenizers_cpp_example_stream_decode
    examples/stream_decode.cpp
  )
  target_link_libraries(tokenizers_cpp_example_stream_decode
    PRIVATE
      tokenizers_cpp::tokenizers_cpp
  )
endif()

if(TOKENIZERS_CPP_BUILD_BENCHMARKS)
  add_executable(tokenizers_cpp_added_token_matcher_benchmark
    benchmarks/added_token_matcher_benchmark.cpp
  )
  target_include_directories(tokenizers_cpp_added_token_matcher_benchmark
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/third_party/aho_corasick"
  )
endif()

if(TOKENIZERS_CPP_BUILD_TESTS)
  enable_testing()
  if(TOKENIZERS_CPP_BUILD_EXAMPLES)
    add_test(
      NAME tokenizers_cpp_example_basic_encode_decode
      COMMAND tokenizers_cpp_example_basic_encode_decode
    )
    add_test(
      NAME tokenizers_cpp_example_batch_and_padding
      COMMAND tokenizers_cpp_example_batch_and_padding
    )
    add_test(
      NAME tokenizers_cpp_example_stream_decode
      COMMAND tokenizers_cpp_example_stream_decode
    )
  endif()

  add_executable(tokenizers_cpp_api_smoke
    tests/parity/tokenizer_api_smoke.cpp
  )
  target_link_libraries(tokenizers_cpp_api_smoke PRIVATE tokenizers_cpp::tokenizers_cpp)
  add_test(NAME tokenizers_cpp_api_smoke COMMAND tokenizers_cpp_api_smoke)

  add_executable(tokenizers_cpp_generated_fixture_test
    tests/parity/generated_fixture_test.cpp
  )
  target_link_libraries(tokenizers_cpp_generated_fixture_test PRIVATE tokenizers_cpp::tokenizers_cpp)
  target_compile_definitions(tokenizers_cpp_generated_fixture_test
    PRIVATE
      TOKENIZERS_CPP_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/parity/fixtures"
  )
  add_test(NAME tokenizers_cpp_generated_fixture_test COMMAND tokenizers_cpp_generated_fixture_test)

  add_executable(tokenizers_cpp_unicode_backend_test
    tests/parity/unicode_backend_test.cpp
  )
  target_link_libraries(tokenizers_cpp_unicode_backend_test
    PRIVATE
      tokenizers_cpp::tokenizers_cpp
  )
  add_test(NAME tokenizers_cpp_unicode_backend_test COMMAND tokenizers_cpp_unicode_backend_test)
  if(NOT TOKENIZERS_CPP_ICU_SHARED AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(
      NAME tokenizers_cpp_no_shared_icu_audit
      COMMAND
        ${CMAKE_COMMAND}
        -DTOKENIZERS_CPP_AUDIT_BINARY=$<TARGET_FILE:tokenizers_cpp_unicode_backend_test>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckNoSharedIcu.cmake
    )
  endif()

  add_executable(tokenizers_cpp_json_wrapper_dispatch_test
    tests/parity/json_wrapper_dispatch_test.cpp
  )
  target_link_libraries(tokenizers_cpp_json_wrapper_dispatch_test PRIVATE tokenizers_cpp::tokenizers_cpp)
  add_test(NAME tokenizers_cpp_json_wrapper_dispatch_test COMMAND tokenizers_cpp_json_wrapper_dispatch_test)

  add_executable(tokenizers_cpp_added_tokens_test
    tests/parity/added_tokens_test.cpp
  )
  target_link_libraries(tokenizers_cpp_added_tokens_test PRIVATE tokenizers_cpp::tokenizers_cpp)
  add_test(NAME tokenizers_cpp_added_tokens_test COMMAND tokenizers_cpp_added_tokens_test)

  add_executable(tokenizers_cpp_wordlevel_test
    tests/parity/wordlevel_test.cpp
  )
  target_link_libraries(tokenizers_cpp_wordlevel_test PRIVATE tokenizers_cpp::tokenizers_cpp)
  add_test(NAME tokenizers_cpp_wordlevel_test COMMAND tokenizers_cpp_wordlevel_test)

  add_executable(tokenizers_cpp_split_pre_tokenizer_test
    tests/parity/split_pre_tokenizer_test.cpp
  )
  target_link_libraries(tokenizers_cpp_split_pre_tokenizer_test PRIVATE tokenizers_cpp::tokenizers_cpp)
  add_test(NAME tokenizers_cpp_split_pre_tokenizer_test COMMAND tokenizers_cpp_split_pre_tokenizer_test)

  if(TOKENIZERS_CPP_BUILD_HF_TEST_DATA_TESTS)
    add_executable(tokenizers_cpp_byte_level_bpe_test
      tests/parity/byte_level_bpe_test.cpp
    )
    target_link_libraries(tokenizers_cpp_byte_level_bpe_test PRIVATE tokenizers_cpp::tokenizers_cpp)
    target_compile_definitions(tokenizers_cpp_byte_level_bpe_test
      PRIVATE
        TOKENIZERS_CPP_HF_TEST_DATA_DIR="${TOKENIZERS_CPP_HF_TEST_DATA_DIR}"
    )
    add_test(NAME tokenizers_cpp_byte_level_bpe_test COMMAND tokenizers_cpp_byte_level_bpe_test)
  endif()

  add_executable(tokenizers_cpp_bpe_config_test
    tests/parity/bpe_config_test.cpp
  )
  target_link_libraries(tokenizers_cpp_bpe_config_test PRIVATE tokenizers_cpp::tokenizers_cpp)
  add_test(NAME tokenizers_cpp_bpe_config_test COMMAND tokenizers_cpp_bpe_config_test)

  if(TOKENIZERS_CPP_BUILD_HF_TEST_DATA_TESTS)
    add_executable(tokenizers_cpp_unigram_test
      tests/parity/unigram_test.cpp
    )
    target_link_libraries(tokenizers_cpp_unigram_test PRIVATE tokenizers_cpp::tokenizers_cpp)
    target_compile_definitions(tokenizers_cpp_unigram_test
      PRIVATE
        TOKENIZERS_CPP_HF_TEST_DATA_DIR="${TOKENIZERS_CPP_HF_TEST_DATA_DIR}"
    )
    add_test(NAME tokenizers_cpp_unigram_test COMMAND tokenizers_cpp_unigram_test)
  endif()

  add_executable(tokenizers_cpp_replace_normalizer_test
    tests/parity/replace_normalizer_test.cpp
  )
  target_link_libraries(tokenizers_cpp_replace_normalizer_test PRIVATE tokenizers_cpp::tokenizers_cpp)
  add_test(NAME tokenizers_cpp_replace_normalizer_test COMMAND tokenizers_cpp_replace_normalizer_test)

  if(TOKENIZERS_CPP_BUILD_HF_TEST_DATA_TESTS)
    add_executable(tokenizers_cpp_sentencepiece_albert_test
      tests/parity/sentencepiece_albert_test.cpp
    )
    target_link_libraries(tokenizers_cpp_sentencepiece_albert_test
      PRIVATE
        tokenizers_cpp::tokenizers_cpp
    )
    target_compile_definitions(tokenizers_cpp_sentencepiece_albert_test
      PRIVATE
        TOKENIZERS_CPP_HF_TEST_DATA_DIR="${TOKENIZERS_CPP_HF_TEST_DATA_DIR}"
    )
    add_test(
      NAME tokenizers_cpp_sentencepiece_albert_test
      COMMAND tokenizers_cpp_sentencepiece_albert_test
    )

    add_executable(tokenizers_cpp_serialization_model_load_test
      tests/parity/serialization_model_load_test.cpp
    )
    target_link_libraries(tokenizers_cpp_serialization_model_load_test PRIVATE tokenizers_cpp::tokenizers_cpp)
    target_compile_definitions(tokenizers_cpp_serialization_model_load_test
      PRIVATE
        TOKENIZERS_CPP_HF_TEST_DATA_DIR="${TOKENIZERS_CPP_HF_TEST_DATA_DIR}"
    )
    add_test(
      NAME tokenizers_cpp_serialization_model_load_test
      COMMAND tokenizers_cpp_serialization_model_load_test
    )

    add_executable(tokenizers_cpp_bert_wordpiece_added_tokens_test
      tests/parity/bert_wordpiece_added_tokens_test.cpp
    )
    target_link_libraries(tokenizers_cpp_bert_wordpiece_added_tokens_test
      PRIVATE
        tokenizers_cpp::tokenizers_cpp
    )
    target_compile_definitions(tokenizers_cpp_bert_wordpiece_added_tokens_test
      PRIVATE
        TOKENIZERS_CPP_HF_TEST_DATA_DIR="${TOKENIZERS_CPP_HF_TEST_DATA_DIR}"
    )
    add_test(
      NAME tokenizers_cpp_bert_wordpiece_added_tokens_test
      COMMAND tokenizers_cpp_bert_wordpiece_added_tokens_test
    )

    add_executable(tokenizers_cpp_documentation_pipeline_test
      tests/parity/documentation_pipeline_test.cpp
    )
    target_link_libraries(tokenizers_cpp_documentation_pipeline_test
      PRIVATE
        tokenizers_cpp::tokenizers_cpp
    )
    target_compile_definitions(tokenizers_cpp_documentation_pipeline_test
      PRIVATE
        TOKENIZERS_CPP_HF_TEST_DATA_DIR="${TOKENIZERS_CPP_HF_TEST_DATA_DIR}"
    )
    add_test(
      NAME tokenizers_cpp_documentation_pipeline_test
      COMMAND tokenizers_cpp_documentation_pipeline_test
    )

    add_executable(tokenizers_cpp_documentation_load_test
      tests/parity/documentation_load_test.cpp
    )
    target_link_libraries(tokenizers_cpp_documentation_load_test
      PRIVATE
        tokenizers_cpp::tokenizers_cpp
    )
    target_compile_definitions(tokenizers_cpp_documentation_load_test
      PRIVATE
        TOKENIZERS_CPP_HF_TEST_DATA_DIR="${TOKENIZERS_CPP_HF_TEST_DATA_DIR}"
    )
    add_test(
      NAME tokenizers_cpp_documentation_load_test
      COMMAND tokenizers_cpp_documentation_load_test
    )

    add_executable(tokenizers_cpp_stream_decode_test
      tests/parity/stream_decode_test.cpp
    )
    target_link_libraries(tokenizers_cpp_stream_decode_test
      PRIVATE
        tokenizers_cpp::tokenizers_cpp
    )
    target_compile_definitions(tokenizers_cpp_stream_decode_test
      PRIVATE
        TOKENIZERS_CPP_HF_TEST_DATA_DIR="${TOKENIZERS_CPP_HF_TEST_DATA_DIR}"
    )
    add_test(
      NAME tokenizers_cpp_stream_decode_test
      COMMAND tokenizers_cpp_stream_decode_test
    )
  endif()

  add_executable(tokenizers_cpp_decoder_sequence_test
    tests/parity/decoder_sequence_test.cpp
  )
  target_link_libraries(tokenizers_cpp_decoder_sequence_test
    PRIVATE
      tokenizers_cpp::tokenizers_cpp
  )
  add_test(
    NAME tokenizers_cpp_decoder_sequence_test
    COMMAND tokenizers_cpp_decoder_sequence_test
  )

  add_executable(tokenizers_cpp_template_processing_test
    tests/parity/template_processing_test.cpp
  )
  target_link_libraries(tokenizers_cpp_template_processing_test
    PRIVATE
      tokenizers_cpp::tokenizers_cpp
  )
  add_test(
    NAME tokenizers_cpp_template_processing_test
    COMMAND tokenizers_cpp_template_processing_test
  )

  add_executable(tokenizers_cpp_truncation_test
    tests/parity/truncation_test.cpp
  )
  target_link_libraries(tokenizers_cpp_truncation_test
    PRIVATE
      tokenizers_cpp::tokenizers_cpp
  )
  add_test(
    NAME tokenizers_cpp_truncation_test
    COMMAND tokenizers_cpp_truncation_test
  )

  add_executable(tokenizers_cpp_padding_test
    tests/parity/padding_test.cpp
  )
  target_link_libraries(tokenizers_cpp_padding_test
    PRIVATE
      tokenizers_cpp::tokenizers_cpp
  )
  add_test(
    NAME tokenizers_cpp_padding_test
    COMMAND tokenizers_cpp_padding_test
  )

  if(TOKENIZERS_CPP_BUILD_HF_TEST_DATA_TESTS)
    add_executable(tokenizers_cpp_roberta_processing_test
      tests/parity/roberta_processing_test.cpp
    )
    target_link_libraries(tokenizers_cpp_roberta_processing_test
      PRIVATE
        tokenizers_cpp::tokenizers_cpp
    )
    target_compile_definitions(tokenizers_cpp_roberta_processing_test
      PRIVATE
        TOKENIZERS_CPP_HF_TEST_DATA_DIR="${TOKENIZERS_CPP_HF_TEST_DATA_DIR}"
    )
    add_test(
      NAME tokenizers_cpp_roberta_processing_test
      COMMAND tokenizers_cpp_roberta_processing_test
    )

    add_executable(tokenizers_cpp_post_processor_sequence_test
      tests/parity/post_processor_sequence_test.cpp
    )
    target_link_libraries(tokenizers_cpp_post_processor_sequence_test
      PRIVATE
        tokenizers_cpp::tokenizers_cpp
    )
    target_compile_definitions(tokenizers_cpp_post_processor_sequence_test
      PRIVATE
        TOKENIZERS_CPP_HF_TEST_DATA_DIR="${TOKENIZERS_CPP_HF_TEST_DATA_DIR}"
    )
    add_test(
      NAME tokenizers_cpp_post_processor_sequence_test
      COMMAND tokenizers_cpp_post_processor_sequence_test
    )

    add_executable(tokenizers_cpp_real_tokenizer_smoke_test
      tests/parity/real_tokenizer_smoke_test.cpp
    )
    target_link_libraries(tokenizers_cpp_real_tokenizer_smoke_test
      PRIVATE
        tokenizers_cpp::tokenizers_cpp
    )
    target_compile_definitions(tokenizers_cpp_real_tokenizer_smoke_test
      PRIVATE
        TOKENIZERS_CPP_HF_TEST_DATA_DIR="${TOKENIZERS_CPP_HF_TEST_DATA_DIR}"
    )
    add_test(
      NAME tokenizers_cpp_real_tokenizer_smoke_test
      COMMAND tokenizers_cpp_real_tokenizer_smoke_test
    )

    add_executable(tokenizers_cpp_llama_encode_test
      tests/parity/llama_encode_test.cpp
    )
    target_link_libraries(tokenizers_cpp_llama_encode_test
      PRIVATE
        tokenizers_cpp::tokenizers_cpp
    )
    target_compile_definitions(tokenizers_cpp_llama_encode_test
      PRIVATE
        TOKENIZERS_CPP_HF_TEST_DATA_DIR="${TOKENIZERS_CPP_HF_TEST_DATA_DIR}"
    )
    add_test(
      NAME tokenizers_cpp_llama_encode_test
      COMMAND tokenizers_cpp_llama_encode_test
    )
  endif()

  add_test(
    NAME tokenizers_cpp_consumer_add_subdirectory_smoke
    COMMAND
      ${CMAKE_COMMAND}
      -DTOKENIZERS_CPP_CONSUMER_MODE=add_subdirectory
      -DTOKENIZERS_CPP_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -DTOKENIZERS_CPP_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
      -DTOKENIZERS_CPP_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/consumer
      -DTOKENIZERS_CPP_CONSUMER_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}/consumer-add-subdirectory-build
      -DTOKENIZERS_CPP_CONSUMER_TOKENIZER_JSON=${CMAKE_CURRENT_SOURCE_DIR}/tests/parity/fixtures/simple_wordlevel_tokenizer.json
      -DTOKENIZERS_CPP_AUDIT_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckNoSharedIcu.cmake
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunConsumerSmoke.cmake
  )
  add_test(
    NAME tokenizers_cpp_consumer_installed_package_smoke
    COMMAND
      ${CMAKE_COMMAND}
      -DTOKENIZERS_CPP_CONSUMER_MODE=installed_package
      -DTOKENIZERS_CPP_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -DTOKENIZERS_CPP_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
      -DTOKENIZERS_CPP_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/consumer-install
      -DTOKENIZERS_CPP_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/consumer
      -DTOKENIZERS_CPP_CONSUMER_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}/consumer-installed-package-build
      -DTOKENIZERS_CPP_CONSUMER_TOKENIZER_JSON=${CMAKE_CURRENT_SOURCE_DIR}/tests/parity/fixtures/simple_wordlevel_tokenizer.json
      -DTOKENIZERS_CPP_AUDIT_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckNoSharedIcu.cmake
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunConsumerSmoke.cmake
  )
endif()
