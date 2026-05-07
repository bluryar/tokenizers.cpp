add_library(tokenizers_cpp_nlohmann_json INTERFACE)
add_library(nlohmann_json::nlohmann_json ALIAS tokenizers_cpp_nlohmann_json)
target_include_directories(tokenizers_cpp_nlohmann_json
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Full Unicode and regex parity dependencies must be added as project-owned
# vendored/static dependencies or explicit optional features. The default target
# must not rely on system dynamic ICU/RE2 libraries.

function(tokenizers_cpp_reject_shared_library library_path)
  if(NOT TOKENIZERS_CPP_ICU_SHARED AND library_path MATCHES "\\.(so|dylib|dll)$")
    message(FATAL_ERROR
      "TOKENIZERS_CPP_ICU_SHARED=OFF rejects shared ICU library: ${library_path}"
    )
  endif()
endfunction()

if(NOT TOKENIZERS_CPP_ICU_VENDOR)
  message(FATAL_ERROR
    "The supported tokenizers.cpp build requires TOKENIZERS_CPP_ICU_VENDOR=ON. "
    "The default build must not discover system ICU."
  )
endif()

set(_tokenizers_cpp_icu_include "${TOKENIZERS_CPP_ICU_ROOT}/include")
set(_tokenizers_cpp_icu_lib "${TOKENIZERS_CPP_ICU_ROOT}/lib")
if(NOT EXISTS "${_tokenizers_cpp_icu_include}/unicode/utypes.h")
  message(FATAL_ERROR
    "Vendored ICU headers not found at "
    "${_tokenizers_cpp_icu_include}/unicode/utypes.h. "
    "Build or install ICU4C into TOKENIZERS_CPP_ICU_ROOT."
  )
endif()

find_library(TOKENIZERS_CPP_ICU_UC_LIBRARY
  NAMES icuuc libicuuc
  PATHS "${_tokenizers_cpp_icu_lib}"
  NO_DEFAULT_PATH
  REQUIRED
)
find_library(TOKENIZERS_CPP_ICU_I18N_LIBRARY
  NAMES icui18n libicui18n
  PATHS "${_tokenizers_cpp_icu_lib}"
  NO_DEFAULT_PATH
  REQUIRED
)
find_library(TOKENIZERS_CPP_ICU_DATA_LIBRARY
  NAMES icudata libicudata
  PATHS "${_tokenizers_cpp_icu_lib}"
  NO_DEFAULT_PATH
  REQUIRED
)

tokenizers_cpp_reject_shared_library("${TOKENIZERS_CPP_ICU_UC_LIBRARY}")
tokenizers_cpp_reject_shared_library("${TOKENIZERS_CPP_ICU_I18N_LIBRARY}")
tokenizers_cpp_reject_shared_library("${TOKENIZERS_CPP_ICU_DATA_LIBRARY}")

add_library(tokenizers_cpp_icu4c_uc UNKNOWN IMPORTED)
add_library(tokenizers_cpp_icu4c::uc ALIAS tokenizers_cpp_icu4c_uc)
set_target_properties(tokenizers_cpp_icu4c_uc PROPERTIES
  IMPORTED_LOCATION "${TOKENIZERS_CPP_ICU_UC_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${_tokenizers_cpp_icu_include}"
)

add_library(tokenizers_cpp_icu4c_i18n UNKNOWN IMPORTED)
add_library(tokenizers_cpp_icu4c::i18n ALIAS tokenizers_cpp_icu4c_i18n)
set_target_properties(tokenizers_cpp_icu4c_i18n PROPERTIES
  IMPORTED_LOCATION "${TOKENIZERS_CPP_ICU_I18N_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${_tokenizers_cpp_icu_include}"
)

add_library(tokenizers_cpp_icu4c_data UNKNOWN IMPORTED)
add_library(tokenizers_cpp_icu4c::data ALIAS tokenizers_cpp_icu4c_data)
set_target_properties(tokenizers_cpp_icu4c_data PROPERTIES
  IMPORTED_LOCATION "${TOKENIZERS_CPP_ICU_DATA_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${_tokenizers_cpp_icu_include}"
)
