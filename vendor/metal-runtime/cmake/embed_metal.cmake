# Converts kernels.metal into a C++ byte-array header so the Metal source is
# embedded in the binary (no runtime file dependency).
# Usage: cmake -DINPUT=... -DOUTPUT=... -P embed_metal.cmake
file(READ ${INPUT} content HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes ${content})
file(WRITE ${OUTPUT} "// Generated from metal/kernels.metal by embed_metal.cmake — do not edit.
namespace mit2 {
const unsigned char kMetalKernelsSourceBytes[] = {${bytes}0x00};
const char* const kMetalKernelsSource = reinterpret_cast<const char*>(kMetalKernelsSourceBytes);
}  // namespace mit2
")
