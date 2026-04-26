from __future__ import annotations

import importlib
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class BackendSpec:
    raw: str
    kind: str
    platform: str | None = None
    device_index: int | None = None


@dataclass(frozen=True)
class BackendRuntime:
    spec: BackendSpec
    array: Any
    lax: Any
    dtype_name: str
    float_dtype: Any
    jax_module: Any | None = None
    device: Any | None = None

    def __repr__(self) -> str:
        if self.spec.kind == "explicit":
            return "explicit"
        if self.jax_module is None:
            return "numpy"
        return str(self.jax_module.default_backend())

    def put(self, value):
        if self.jax_module is None or self.device is None:
            return value
        return self.jax_module.device_put(value, self.device)

    def device_get(self, value):
        if self.jax_module is None:
            return value
        return self.jax_module.device_get(value)

    def block_until_ready(self, value):
        """Block on a backend future until the underlying computation finishes.

        Required for accurate wall-clock timing of JAX runs, whose operations
        are asynchronous: ``run()`` returns immediately with a future while the
        actual work proceeds on the device. For non-JAX backends this is a
        no-op because numpy / explicit / stormpy return fully materialized
        values.
        """
        if self.jax_module is None:
            return value
        # ``jax.block_until_ready`` accepts a pytree and blocks all leaves; it
        # returns the (still-on-device) value without copying to host.
        return self.jax_module.block_until_ready(value)

    def scatter_add(self, tensor, indices: tuple[Any, ...], values):
        if self.jax_module is not None:
            return tensor.at[indices].add(values)
        np.add.at(tensor, tuple(indices), values)
        return tensor

    def set_index(self, tensor, indices: tuple[int, ...], value):
        if self.jax_module is not None:
            return tensor.at[indices].set(value)
        tensor[indices] = value
        return tensor

    def to_python_bool(self, value) -> bool:
        host_value = self.device_get(value)
        if hasattr(host_value, "item"):
            return bool(host_value.item())
        return bool(host_value)

    def to_python_int(self, value) -> int:
        host_value = self.device_get(value)
        if hasattr(host_value, "item"):
            return int(host_value.item())
        return int(host_value)


class NumpyLaxCompat:
    @staticmethod
    def fori_loop(lower, upper, body_fun, init_val):
        lower_index = int(np.asarray(lower).item())
        upper_index = int(np.asarray(upper).item())
        carry = init_val
        for index in range(lower_index, upper_index):
            carry = body_fun(index, carry)
        return carry


def parse_backend(raw_backend: str | None = None) -> BackendSpec:
    if raw_backend is None:
        return BackendSpec(raw="jax:cpu", kind="jax", platform="cpu")

    normalized = raw_backend.strip().lower()
    if normalized == "numpy":
        return BackendSpec(raw="numpy", kind="numpy")
    if normalized in ("explicit", "storm"):
        return BackendSpec(raw="explicit", kind="explicit")
    if normalized == "jax:cpu":
        return BackendSpec(raw="jax:cpu", kind="jax", platform="cpu")

    cuda_match = re.fullmatch(r"jax:cuda:(\d+)", normalized)
    if cuda_match is not None:
        return BackendSpec(
            raw=normalized,
            kind="jax",
            platform="cuda",
            device_index=int(cuda_match.group(1)),
        )

    raise ValueError("Invalid backend. Expected one of: numpy, explicit, jax:cpu, jax:cuda:N")


def parse_dtype(raw_dtype: str | None = None) -> str:
    if raw_dtype is None:
        return "float32"

    normalized = raw_dtype.strip().lower()
    if normalized in {"float32", "float64"}:
        return normalized

    raise ValueError("Invalid dtype. Expected one of: float32, float64")


def jit(backend: BackendRuntime, **jit_kwargs):
    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        if backend.jax_module is None:
            return function
        return backend.jax_module.jit(function, **jit_kwargs)

    return decorator


def build_backend(backend: str | BackendSpec | None, *, dtype: str | None = None) -> BackendRuntime:
    spec = backend if isinstance(backend, BackendSpec) else parse_backend(backend)
    dtype_name = parse_dtype(dtype)
    if spec.kind == "numpy":
        return BackendRuntime(
            spec=spec,
            array=np,
            lax=NumpyLaxCompat(),
            dtype_name=dtype_name,
            float_dtype=getattr(np, dtype_name),
        )
    if spec.kind == "explicit":
        # Delegated to Storm via stormpy (no Tessa array pipeline).
        return BackendRuntime(
            spec=spec,
            array=np,
            lax=NumpyLaxCompat(),
            dtype_name=dtype_name,
            float_dtype=getattr(np, dtype_name),
        )
    return _build_jax_backend(spec, dtype_name=dtype_name)


def _build_jax_backend(spec: BackendSpec, *, dtype_name: str) -> BackendRuntime:
    platform_value = spec.platform
    assert platform_value is not None

    if "jax" not in sys.modules:
        os.environ["JAX_PLATFORMS"] = platform_value
        if dtype_name == "float64":
            os.environ["JAX_ENABLE_X64"] = "true"
    else:
        current_platforms = os.environ.get("JAX_PLATFORMS")
        if current_platforms not in (None, platform_value):
            raise RuntimeError(
                "JAX has already been imported with a different platform selection. "
                "Start a fresh process to switch backends."
            )

    try:
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")
    except ImportError as exc:
        raise RuntimeError(
            f"Backend '{spec.raw}' requires JAX. Install project dependencies or use the repo's Nix environment."
        ) from exc

    if dtype_name == "float64":
        jax.config.update("jax_enable_x64", True)

    try:
        devices = jax.devices(platform_value)
    except Exception as exc:
        raise RuntimeError(f"Backend '{spec.raw}' could not initialize JAX platform '{platform_value}': {exc}") from exc

    if platform_value == "cpu":
        if not devices:
            raise RuntimeError("JAX CPU backend is unavailable")
        device = devices[0]
    else:
        if spec.device_index is None:
            raise RuntimeError("Internal error: missing CUDA device index")
        if spec.device_index >= len(devices):
            raise ValueError(
                f"Requested CUDA device {spec.device_index}, but only {len(devices)} CUDA device(s) are available"
            )
        device = devices[spec.device_index]

    return BackendRuntime(
        spec=spec,
        array=jnp,
        lax=jax.lax,
        dtype_name=dtype_name,
        float_dtype=getattr(jnp, dtype_name),
        jax_module=jax,
        device=device,
    )
