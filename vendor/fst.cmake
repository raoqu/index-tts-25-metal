set(FST_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/vendor/openfst/src)
add_library(itts25_fst STATIC
  ${FST_ROOT}/lib/compat.cc ${FST_ROOT}/lib/flags.cc ${FST_ROOT}/lib/fst.cc
  ${FST_ROOT}/lib/properties.cc ${FST_ROOT}/lib/symbol-table.cc ${FST_ROOT}/lib/util.cc
  ${FST_ROOT}/lib/symbol-table-ops.cc ${FST_ROOT}/lib/mapped-file.cc ${FST_ROOT}/lib/weight.cc)
target_include_directories(itts25_fst PUBLIC ${FST_ROOT}/include)
target_compile_options(itts25_fst PRIVATE -w)
set(KF ${CMAKE_CURRENT_SOURCE_DIR}/vendor/kaldifst/csrc)
add_library(itts25_kaldifst STATIC
  ${KF}/context-fst.cc ${KF}/kaldi-fst-io.cc ${KF}/kaldi-holder.cc ${KF}/kaldi-io.cc
  ${KF}/kaldi-math.cc ${KF}/kaldi-semaphore.cc ${KF}/kaldi-table.cc ${KF}/parse-options.cc
  ${KF}/text-normalizer.cc ${KF}/text-utils.cc)
target_include_directories(itts25_kaldifst PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor)
target_link_libraries(itts25_kaldifst PUBLIC itts25_fst)
target_compile_options(itts25_kaldifst PRIVATE -w)
