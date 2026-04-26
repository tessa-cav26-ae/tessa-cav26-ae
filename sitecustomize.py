import ctypes
import os

if os.name == "posix":
    try:
        ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        for base in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
            candidate = os.path.join(base, "libcuda.so.1")
            if os.path.exists(candidate):
                try:
                    ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break
