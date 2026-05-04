import ctypes
import os

if os.name == "posix":
    try:
        ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        for base in (
            "/usr/lib/x86_64-linux-gnu",   # Debian/Ubuntu multiarch
            "/usr/lib64",                  # RHEL/Fedora/SUSE
            "/usr/lib/wsl/lib",            # WSL2 GPU passthrough
            "/usr/lib",                    # Arch/Gentoo/non-multiarch
        ):
            candidate = os.path.join(base, "libcuda.so.1")
            if os.path.exists(candidate):
                try:
                    ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break
