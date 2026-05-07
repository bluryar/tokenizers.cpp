set(_required_vars
  TOKENIZERS_CPP_CONSUMER_MODE
  TOKENIZERS_CPP_SOURCE_DIR
  TOKENIZERS_CPP_BUILD_DIR
  TOKENIZERS_CPP_CONSUMER_SOURCE_DIR
  TOKENIZERS_CPP_CONSUMER_BUILD_DIR
  TOKENIZERS_CPP_CONSUMER_TOKENIZER_JSON
)

foreach(_var IN LISTS _required_vars)
  if(NOT DEFINED "${_var}" OR "${${_var}}" STREQUAL "")
    message(FATAL_ERROR "${_var} is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${TOKENIZERS_CPP_CONSUMER_BUILD_DIR}")

set(_configure_args
  "-DTOKENIZERS_CPP_CONSUMER_TOKENIZER_JSON=${TOKENIZERS_CPP_CONSUMER_TOKENIZER_JSON}"
)

if(TOKENIZERS_CPP_CONSUMER_MODE STREQUAL "add_subdirectory")
  list(APPEND _configure_args
    "-DTOKENIZERS_CPP_CONSUMER_SOURCE_TREE=${TOKENIZERS_CPP_SOURCE_DIR}"
  )
elseif(TOKENIZERS_CPP_CONSUMER_MODE STREQUAL "installed_package")
  if(NOT DEFINED TOKENIZERS_CPP_INSTALL_PREFIX OR
      "${TOKENIZERS_CPP_INSTALL_PREFIX}" STREQUAL "")
    message(FATAL_ERROR "TOKENIZERS_CPP_INSTALL_PREFIX is required")
  endif()
  file(REMOVE_RECURSE "${TOKENIZERS_CPP_INSTALL_PREFIX}")
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}"
      --install "${TOKENIZERS_CPP_BUILD_DIR}"
      --prefix "${TOKENIZERS_CPP_INSTALL_PREFIX}"
    RESULT_VARIABLE _install_result
  )
  if(NOT _install_result EQUAL 0)
    message(FATAL_ERROR "tokenizers_cpp install failed")
  endif()
  list(APPEND _configure_args
    "-DTOKENIZERS_CPP_CONSUMER_PACKAGE_PREFIX=${TOKENIZERS_CPP_INSTALL_PREFIX}"
  )
else()
  message(FATAL_ERROR
    "unsupported TOKENIZERS_CPP_CONSUMER_MODE=${TOKENIZERS_CPP_CONSUMER_MODE}"
  )
endif()

execute_process(
  COMMAND
    "${CMAKE_COMMAND}"
    -S "${TOKENIZERS_CPP_CONSUMER_SOURCE_DIR}"
    -B "${TOKENIZERS_CPP_CONSUMER_BUILD_DIR}"
    ${_configure_args}
  RESULT_VARIABLE _configure_result
)
if(NOT _configure_result EQUAL 0)
  message(FATAL_ERROR
    "consumer configure failed for ${TOKENIZERS_CPP_CONSUMER_MODE}"
  )
endif()

execute_process(
  COMMAND
    "${CMAKE_COMMAND}"
    --build "${TOKENIZERS_CPP_CONSUMER_BUILD_DIR}"
  RESULT_VARIABLE _build_result
)
if(NOT _build_result EQUAL 0)
  message(FATAL_ERROR "consumer build failed for ${TOKENIZERS_CPP_CONSUMER_MODE}")
endif()

execute_process(
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
    --test-dir "${TOKENIZERS_CPP_CONSUMER_BUILD_DIR}"
    --output-on-failure
  RESULT_VARIABLE _test_result
)
if(NOT _test_result EQUAL 0)
  message(FATAL_ERROR "consumer test failed for ${TOKENIZERS_CPP_CONSUMER_MODE}")
endif()

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" AND
    DEFINED TOKENIZERS_CPP_AUDIT_SCRIPT AND
    NOT "${TOKENIZERS_CPP_AUDIT_SCRIPT}" STREQUAL "")
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}"
      "-DTOKENIZERS_CPP_AUDIT_BINARY=${TOKENIZERS_CPP_CONSUMER_BUILD_DIR}/tokenizers_cpp_consumer_smoke"
      -P "${TOKENIZERS_CPP_AUDIT_SCRIPT}"
    RESULT_VARIABLE _audit_result
  )
  if(NOT _audit_result EQUAL 0)
    message(FATAL_ERROR
      "consumer shared ICU audit failed for ${TOKENIZERS_CPP_CONSUMER_MODE}"
    )
  endif()
endif()
