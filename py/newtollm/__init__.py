"""NewToLLM: a GPT built twice - PyTorch here, LibTorch in ``cpp/newtollm``.

Every module in this package has a C++ twin with the same file name, the
same class names and the same function names. Blocks marked ``# @twin <name>``
are aligned with the matching ``// @twin <name>`` block on the C++ side.
"""
