# Native text normalization dependencies

Copied from `~/mylab/index-tts2-metal/` commit
`eda855b2e9d7cfaa4269380f42880d41b060c4d2` (reference checkout unchanged):

- OpenFST core headers and library sources: sherpa-onnx-2024-06-13, Apache-2.0.
- kaldifst C++ sources: v1.7.17, Apache-2.0.
- `src/text_processor.cpp` UTF-8 helpers, punctuation maps and FST token parser
  adapted from reference `runtime/impl/text_frontend.cpp`; the SentencePiece
  model and tokenizer are deliberately excluded. 2.5 protection/glossary,
  pronunciation annotations, Japanese tokenization and splitting are ported
  from this repository's `indextts/utils/front.py` and `infer_v2_5.py`.

`vendor/fst.cmake` builds only the native libraries. It never downloads sources
or invokes Python. FST grammars, UniDic dictionary and native MeCab dylib are
copied into ignored frontend resources by the offline exporter, retaining the
UniDic licenses. MeCab dylib is the native library supplied with the installed
fugashi wheel; it links only macOS system libraries, not Python.
