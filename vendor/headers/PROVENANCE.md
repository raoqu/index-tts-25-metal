# Native protocol headers

Copied from local llama.cpp commit `6d05498314db1b57f81c271080018aa2d0b89be9`:

- `vendor/nlohmann/json.hpp`: nlohmann/json, MIT license embedded in header.
- `vendor/cpp-httplib/httplib.h`: cpp-httplib, MIT license embedded in header.

These are protocol/data utilities only. No llama.cpp model code is used.

The same checkout supplies split implementation `httplib.cpp` and
`HTTPLIB_LICENSE`; both are required by the vendored 0.53.1 HTTP API.
