# Artifact: Tessa

This repository contains the tool source code, benchmarks, and instructions to reproduce the results in the paper.

## Table of Contents

* [Overview](#overview)
* [Platform Requirements](#platform-requirements)
* [Getting Started](#getting-started)
* [Experimental Evaluation](#experimental-evaluation)
* [Parametric Example: Knuth-Yao Die](#parametric-example-knuth-yao-die)
* [Troubleshooting](#troubleshooting)
* [Reproducing External Tool Numbers](#reproducing-external-tool-numbers)

## Overview

`tessa` is a GPU-accelerated bounded reachability engine for discrete-time Markov chains (DTMCs).
It loads PRISM and JANI models, compiles the transition structure into batched tensor operations, and evaluates them on NVIDIA GPUs via JAX.

```console
> tessa --help
usage: tessa [-h] [--type {jani,prism}] [--const KEY=VALUE]
             [--property PROPERTY] [--horizon HORIZON]
             [--backend {numpy,jax:cpu,jax:cuda:0,...}]
             [--dtype {float32,float64}]
             [--timed_run_num N] [--time-log FILE]
             model

positional arguments:
  model                 Path to a PRISM (.prism/.pm/.nm) or JANI (.jani) model

options:
  --property PROPERTY   Reachability property to evaluate
  --horizon HORIZON     Bounded reachability horizon
  --backend BACKEND     Computation backend (default: jax:cpu)
  --dtype DTYPE         Floating-point precision (default: float64)
  --timed_run_num N     Number of timed runs after warmup
  --time-log FILE       Append timing records as JSONL
```

## Platform Requirements

### Minimum requirements

- **NVIDIA GPU** with a driver supporting CUDA 12.x (check with `nvidia-smi`)
- **Linux** (x86_64)
- **Nix** package manager (recommended — provides Storm, JAX with CUDA, and all other dependencies via `flake.nix`), *or* Python 3.12 with pip (see the [pip alternative](#alternative-pip-install-with-jaxcuda) below)

### Choose your GPU

Tessa runs on any NVIDIA GPU with **CUDA compute capability ≥ 7.5** (Turing or newer). The pip path works on every supported architecture without recompilation.

| Architecture | Compute cap | Example cards             | pip path                        | Nix path                          |
| ------------ | ----------- | ------------------------- | ------------------------------- | --------------------------------- |
| Turing       | 7.5         | RTX 2080, RTX 2080 Ti, T4 | works                           | works out of the box              |
| Ampere       | 8.0 / 8.6   | A100, RTX 3090, RTX 3080  | works                           | works out of the box              |
| Ada          | 8.9         | RTX 4090, L40             | works                           | works out of the box              |
| Hopper       | 9.0         | H100                      | works                           | works out of the box              |
| Blackwell    | 10.0 / 12.0 | B200, RTX 5090            | works (with a recent JAX wheel) | works out of the box              |

Look up your card's compute capability at <https://developer.nvidia.com/cuda-gpus>.

### Verified host

The Nix development shell in `flake.nix` is tested on the machine below.
`flake.nix` pins `cudaPackages_12_8` and builds JAX with `cudaSupport = true` for compute capabilities 7.5–12.0 (Turing through Blackwell), with `cudaForwardCompat = true`. The 12.8 build runs on any 12.x NVIDIA driver (≥ 525) via CUDA minor-version compatibility, including the host's driver 535.

| Component        | Value                                                        |
| ---------------- | ------------------------------------------------------------ |
| OS               | Ubuntu 24.04.4 LTS (Noble Numbat)                            |
| Kernel           | Linux 6.8.0-107-generic                                      |
| CPU              | Intel(R) Core(TM) i7-7820X @ 3.60 GHz (8 cores / 16 threads) |
| RAM              | 128 GB                                                       |
| GPU              | NVIDIA GeForce RTX 2080 (8 GB) + RTX 2080 Ti (11 GB)         |
| NVIDIA driver    | 535.288.01  (as reported by `nvidia-smi`)                    |
| CUDA runtime     | 12.2 (as reported by `nvidia-smi`)                           |

The published wall-clock numbers in [Experimental Evaluation](#experimental-evaluation) were captured against the previous CUDA 12.2 build; expect small drift on a 12.8 toolkit.

Absolute wall-clock times depend on hardware; expect similar relative speedups and scaling trends on different machines.

## Getting Started

### Install the NVIDIA driver

All three install paths below need a working NVIDIA **driver** (recent enough for CUDA 12.x — anything ≥ 525). The Nix and pip paths bundle their own CUDA + cuDNN runtime, so a system-wide CUDA toolkit is **not** required.

Follow the official NVIDIA installer at <https://developer.nvidia.com/cuda-downloads>.

`nvidia-smi` should print a table with a non-empty **Driver Version** (≥ 525), a **CUDA Version** column showing `12.x`, and your GPU listed in the device table. If `nvidia-smi: command not found` or the device table is empty, the driver install did not complete — fix that before proceeding.

### Set up the environment

Install Nix by running:
```
sh <(curl -L https://nixos.org/nix/install) --daemon
```
Please refer to https://nixos.org/download for more info.

The flake builds JAX for compute capabilities 7.5–12.0 (Turing through Blackwell), so every shipped NVIDIA card works without editing `flake.nix`.

Enter the Nix development shell:
```
nix --experimental-features 'nix-command flakes' develop
```

The first build compiles JAX (and stormpy, if you use the included benchmarks) from source — expect ~30 minutes on a fast machine.

Verify the shell is working:
```console
> tessa examples/weather_factory_3.prism --property allStrike --horizon 10
```

### Alternative: Docker

If you'd rather not install Nix on the host, the repository ships a `Dockerfile` that builds Storm, `stormpy`, and `tessa` inside a `nixos/nix` image.

Prerequisites:
- [Docker](https://docs.docker.com/get-docker/).
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) — required for `--gpus all` passthrough. Tessa's default backend is CUDA-JAX.

Build the image:
```shell
> docker build -t tessa .
[+] Building 2854.1s (12/12) FINISHED                                                                              docker:default
 => [internal] load build definition from Dockerfile                                                                         0.1s
 => => transferring dockerfile: 803B                                                                                         0.0s
 => [internal] load metadata for docker.io/nixos/nix:latest                                                                  0.3s
 => [internal] load .dockerignore                                                                                            0.1s
 => => transferring context: 2B                                                                                              0.0s
 => [1/7] FROM docker.io/nixos/nix:latest@sha256:e2fe74e96e965653c7b8f16ac64d1e56581c63c84d7fa07fb0692fd055cd06b0            0.0s
 => [internal] load build context                                                                                            0.3s
 => => transferring context: 149.38kB                                                                                        0.2s
 => CACHED [2/7] RUN mkdir -p /etc/nix &&  echo "experimental-features = nix-command flakes" > /etc/nix/nix.conf  && ..      0.0s
 => CACHED [3/7] COPY . /tessa                                                                                               0.0s
 => CACHED [4/7] WORKDIR /tessa                                                                                              0.0s
 => [5/7] RUN nix build .#storm --cores 8 --print-build-logs                                                               680.8s
 => [6/7] RUN nix build .#stormpy --cores 8 --print-build-logs                                                             204.0s
 => [7/7] RUN nix build .#tessa --cores 8 --print-build-logs                                                              1894.2s
 => exporting to image                                                                                                      73.6s
 => => exporting layers                                                                                                     73.3s
 => => writing image sha256:6c0d6015a599d1de460eb56edecf8d0d7530cbbc1823acfd8ac8038c416b004f                                 0.0s
 => => naming to docker.io/library/tessa     
```
The first build takes roughly 60 minutes because Storm, `stormpy`, and CUDA-enabled JAX are compiled from scratch inside the container (no host Nix store to reuse). Subsequent builds reuse Docker's layer cache.

Run the image — the `Dockerfile`'s `CMD` is `nix develop`, so this drops straight into the Nix shell. `--rm` removes the container on exit so they don't pile up; drop `--rm` if you want the stopped container to be recoverable later via `docker start`.
```shell
> docker run -it --rm --gpus all tessa
```

Or start the container detached and re-enter it later. `--rm` is fine for the detach/re-enter loop because `ctrl-p ctrl-q` only detaches (the container keeps running) — drop `--rm` if you want the container to survive an `exit` of the inner shell so you can `docker start` it back up across sessions:
```shell
> docker run -it --rm --gpus all tessa #(and then detach by ctrl-p ctrl-q)
> docker ps
CONTAINER ID    IMAGE    COMMAND    CREATED    STATUS    PORTS    NAMES
3ac16580a16e    tessa    "nix …"    ....       ....      ....     ....
> docker exec -it 3ac1 bash -c "cd /tessa && nix develop"
```

Inside the container, verify the GPU is visible:
```console
> python -c "import jax; print(jax.devices())"
[CudaDevice(id=0), ...]
```

Then run `tessa --help` or any of the Makefile benchmarks described in [Experimental Evaluation](#experimental-evaluation).

(Optional) Export the built image for distribution:
```shell
> docker save tessa > tessa.tar
```

### Alternative: pip install with jax[cuda]

If you'd rather manage your own Python/CUDA toolchain instead of using Nix, you can install `tessa` with pip using JAX's official CUDA wheels.

Prerequisites:
- Python **3.12** (hard pin in `pyproject.toml`: `requires-python = ">=3.12,<3.13"`).
- An NVIDIA driver compatible with the JAX CUDA wheel you install (verify with `nvidia-smi`; see [Install the NVIDIA driver](#install-the-nvidia-driver)).

JAX's `jax[cuda]` wheel bundles the matching CUDA and cuDNN runtime, so a system-wide CUDA toolkit is **not** required — only a recent enough NVIDIA driver. Pick the `jax[cuda...]` extra that matches your driver (see https://jax.readthedocs.io/en/latest/installation.html for the current list of supported CUDA versions).

```shell
pip install -U "jax[cuda]"
pip install -e .
```

Sanity-check that JAX sees your GPU:
```console
> python -c "import jax; print(jax.devices())"
[CudaDevice(id=0), ...]
```

Note: the pip path installs JAX + `tessa`, but it does **not** install Storm or `stormpy`. PRISM model loading and the Storm comparison in `reproduce.mk`won't be available. If you need them for the paper reproduction, you can pull them in from the flake while keeping the rest of your pip workflow — see [Hybrid: pip-installed Tessa with Storm/stormpy from Nix](#hybrid-pip-installed-tessa-with-stormstormpy-from-nix) below, or install [storm](https://www.stormchecker.org/getting-started.html) and (stormpy)[https://stormchecker.github.io/stormpy/installation.html] seperately.

> **End-to-end smoke test on the pip path.** The example models bundled in `examples/` (`weather_factory_3.prism`, `complex_multi_action.prism`) are PRISM format and require **stormpy** for parsing. With JAX-only pip Tessa you can use the hybrid setup below to bring in stormpy from Nix.

#### Hybrid: pip-installed Tessa with Storm/stormpy from Nix

If you want PRISM parsing and the Storm comparison, but you'd rather not have Nix build Tessa itself from source (that step is the slow part of the dev shell), comment out the `tessa` entry in `flake.nix`'s `devShells.default.buildInputs` and enter the shell:

```diff
 buildInputs = (with pkgs; [ … python312 ])
   ++ (with python312.pkgs; [ jax jaxlib numpy ])
   ++ [
     plot
     storm
     stormpy
-    tessa
+    # tessa   # provided by the pip editable install below
   ];
```

```shell
nix --experimental-features 'nix-command flakes' develop
```

Inside the shell, install Tessa as an editable package that inherits Nix's `python312` site-packages so it can `import stormpy` / `import jax`:

```shell
pip install -e .
tessa --help           # verify the CLI is on PATH
python -c "import stormpy, jax; print(jax.devices())"
```

You now have pip-editable Tessa sharing a single Python 3.12 interpreter with the Nix-built stormpy and JAX, so `make -f reproduce.mk` works end-to-end. Mixing Nix's stormpy from a *non-Nix* Python (e.g., system Python) can be problematic, and we don not recommand it.

## Experimental Evaluation

**Performance might be different on different hardware configurations, but the scaling trends should be similar.**

Benchmark scripts to reproduce the paper figures are driven by `reproduce.mk`.
Three tools are compared: **Tessa** (JAX CUDA), **Storm ADD** (MTBDD engine), and **Storm SPM** (Sparse engine).

| Benchmark        | Size scaling                                                                                                                                                            | Horizon scaling                                                                                                                                                         |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Meeting          | [Figure 1(c) - left](reproduced-april/meeting/testn/meeting-testn.png)                                                                                                  | [Figure 1(c) - right](reproduced-april/meeting/testh/meeting-testh.png)                                                                                                 |
| ParQueues        | [Figure 5(a)](reproduced-april/parqueues/testq/parqueues-testq.png)             • [Figure 5(b)](reproduced-april/parqueues/testq/parqueues-testq-tessa.png)             | [Figure 5(c)](reproduced-april/parqueues/testh/parqueues-testh.png)             • [Figure 5(d)](reproduced-april/parqueues/testh/parqueues-testh-tessa.png)             |
| Weather Factory  | [Figure 6(a)](reproduced-april/weather-factory/testn/weather-factory-testn.png) • [Figure 6(b)](reproduced-april/weather-factory/testn/weather-factory-testn-tessa.png) | [Figure 6(c)](reproduced-april/weather-factory/testh/weather-factory-testh.png) • [Figure 6(d)](reproduced-april/weather-factory/testh/weather-factory-testh-tessa.png) |
| Herman           | [Figure 7(a)](reproduced-april/herman/testn/herman-testn.png)                   • [Figure 7(b)](reproduced-april/herman/testn/herman-testn-tessa.png)                   | [Figure 7(c)](reproduced-april/herman/testh/herman-testh.png)                   • [Figure 7(d)](reproduced-april/herman/testh/herman-testh-tessa.png)                   |

### Wall-clock time for the full reproduction

Approximate wall-clock time spent on each `bench.suite.tool` slice on the verified host (taken from the first→last timestamps in `reproduced/<suite>/<test>/*.log`).
Tessa totals include warmup plus the configured number of timed runs; Storm totals are sequential subprocess invocations, one per parameter point.
Cells where Storm ADD exits `FAILED` on the first parameter point (meeting.testh, weather-factory.testh) finish quickly but produce no useful data beyond H ≤ 2 — flagged explicitly.

| Suite           | Test  |  Tessa  |  Storm ADD  |  Storm SPM |
| --------------- | ----- | ------: | ----------: | ---------: |
| Herman          | testn |      2m |         47m |    25m     |
| Herman          | testh |      5m |      2h 48m | 1h 57m     |
| Meeting         | testn |      2m |          9m |     5m     |
| Meeting         | testh |      6m |          3m | 1h 59m     |
| Weather Factory | testn |      2m |          7m |    46m     |
| Weather Factory | testh |      7m |          4m | 1h 42m     |
| ParQueues       | testq |      2m |         27m |    24m     |
| ParQueues       | testh |      4m |      6h 28m |    15m     |

Totals: Tessa ≈ 27 min, Storm ADD ≈ 10.8 h, Storm SPM ≈ 7.5 h.
Per-point timeout is `TO=1260` s (see top of `reproduce.mk`); parameter points that hit the timeout are recorded as `TIMEOUT` in the CSVs.

### Quick Smoke Test

Runs a minimal parameter subset (~10 minutes) to verify the pipeline:

```shell
make -f reproduce.mk SMOKE=1
```

### Full Reproduction

Runs all benchmarks with the full parameter ranges from the paper (~18.7 hours):

```shell
make -f reproduce.mk
```

### Per-Tool Commands

To rerun all suites against just one tool — useful when only one tool's CSVs need refreshing — use the tool-aggregate targets:

```shell
make -f reproduce.mk tessa      # ~27 minutes -  Tessa across all suites
make -f reproduce.mk storm-add  # ~10.8 hours   -  Storm ADD across all suites
make -f reproduce.mk storm-spm  # ~7.5 hours   -  Storm SPM across all suites
make -f reproduce.mk            # reuses the CSVs produced by the per-tool commands above; verifies Tessa probabilities against Storm and emits PNG scaling plots
```

Finer-grained slices follow the pattern `<suite>.<test>.<tool>`, e.g. `herman.testn.tessa` or `parqueues.testh.storm-spm`.

Each run target has a matching `clean.*` target that drops only that target's CSV and log. Use these to force a rerun of one slice without nuking the rest of the output tree:

```shell
make -f reproduce.mk clean.tessa                # drop all Tessa CSVs/logs
make -f reproduce.mk clean.storm-add            # drop all Storm ADD CSVs/logs
make -f reproduce.mk clean.storm-spm            # drop all Storm SPM CSVs/logs
```

For iterating on Tessa without re-running Storm baselines, use `tessa.all` — it runs the Tessa sweep, then re-runs verification and plotting against the existing `storm.add.csv` / `storm.spm.csv` on disk:

```shell
make -f reproduce.mk tessa.all                  # ~27 minutes
```

### Results

Results are written to `reproduced/<benchmark>/<test>/` (or `smoke/` with `SMOKE=1`).
Each sub-test directory contains:
- `tessa.csv`, `storm.add.csv`, `storm.spm.csv` — timing data
- `*.png` — scaling plots comparing the three tools
- `verify.log` — correctness check (Tessa vs Storm probabilities)

Existing result files are reused automatically. Delete the output directory to force a rerun.

## Parametric Example: Knuth-Yao Die

`benchmarks/kydice/` exercises Tessa's parametric mode: the Knuth-Yao algorithm samples a fair six-sided die from coin flips, and the model leaves the per-flip biases `x` and `y` symbolic. Reachability for each die face is compiled once with `defer_constants=["x", "y"]` (PRISM) or symbolic constants (JANI), and the resulting kernels are differentiable through JAX. The script trains `(x, y)` with `optax.adam` to minimise KL divergence between the induced face distribution and uniform, then plots the trajectory and the KL landscape.

```shell
benchmarks/kydice/run.sh                       # uses benchmarks/kydice/kydice.jani
benchmarks/kydice/run.sh --backend jax:cpu     # CPU-only
```

Outputs land next to the script: `loss.csv`, `loss.png`, `landscape.png`. With the default 100 steps, training converges to `(x, y) ≈ (0.5, 0.5)` and a final KL on the order of `1e-5`.

## Troubleshooting

- **`python -c "import jax; print(jax.devices())"` returns `[CpuDevice(id=0)]` only.** Either the driver is too old, or `jax[cuda]` was not installed (you got the CPU-only wheel). Run `nvidia-smi` to confirm the driver works in this shell, then `pip install -U --force-reinstall "jax[cuda]"`.

- **`RuntimeError: jaxlib was unable to initialize the CUDA runtime`.** Driver / runtime mismatch, or `LD_LIBRARY_PATH` is shadowing the bundled CUDA. Check `nvidia-smi` works, then unset `LD_LIBRARY_PATH` and `CUDA_HOME` for the venv shell and retry.

- **`RuntimeError: jaxlib was unable to initialize the CUDA runtime` *and* `nvidia-smi` works (Nix path).** The repo's `sitecustomize.py` preloads `libcuda.so.1` for jaxlib by first asking the dynamic linker (`ld.so.cache`) and falling back to `/usr/lib/x86_64-linux-gnu/` and `/usr/lib64/`. If your host installed libcuda outside those locations and did not register it in `ld.so.cache` (e.g., a custom NVIDIA install at `/opt/cuda/lib64/`), the preload silently misses. Confirm with `find / -name 'libcuda.so.1' 2>/dev/null` and either run `sudo ldconfig` after adding the directory to `/etc/ld.so.conf.d/`, or add the directory to the fallback tuple in `sitecustomize.py`.

- **`CUDA_ERROR_NO_DEVICE`.** No GPU is visible to the process. Run `nvidia-smi` in the same shell; inside Docker you likely forgot `--gpus all`.

- **`CUDA driver version is insufficient for CUDA runtime version` (Nix path).** Driver predates CUDA 12 (< 525). Upgrade to any 12.x driver, or install `cuda-compat-12-8` (Ubuntu: `sudo apt install cuda-compat-12-8`; see <https://docs.nvidia.com/deploy/cuda-compatibility/>).

- **`no kernel image is available for execution on the device` (Nix path).** Your card's compute capability is not in `cudaCapabilities` and `cudaForwardCompat` PTX did not produce a runnable kernel. Add the capability to the `cudaCapabilities` list in `flake.nix` and re-enter the dev shell.

- **`ImportError: stormpy` (pip path).** Expected — the pip path does not install stormpy. Switch to the Nix path, or use a JANI model (the pip path can load `.jani` without stormpy).

- **`ValueError: Could not infer model type from file extension`.** Model file does not end in `.prism`, `.pm`, `.nm`, or `.jani`. Pass `--type jani` or `--type prism` explicitly.

- **`nvidia-smi: command not found`.** Driver not installed (or PATH issue). Install the NVIDIA driver from your distro's package manager or <https://developer.nvidia.com/cuda-downloads> (driver-only is sufficient).

If you hit something not listed above, please open an issue with the output of `nvidia-smi`, `python -c "import jax; print(jax.devices(), jax.__version__)"`, and the failing command's full traceback.

## Reproducing External Tool Numbers

### Rubicon (Dice)

To reproduce the Rubicon/Dice numbers reported in the paper, please use the Rubicon artifact:

> https://github.com/sjunges/rubicon

Follow the instructions in that repository to set up and run the Dice benchmarks.

### Geni

To reproduce the Geni numbers reported in the paper, please use the Geni artifact:

> https://github.com/geni-icfp25-ae/geni-icfp25-ae/tree/main/bench/ICFP/weather-factory

Follow the instructions in that repository to set up and run the Geni benchmarks.

## Notes

- PRISM loading uses `stormpy` when available.
- In this repo, the intended environment is the Linux/Nix setup from `flake.nix`.
- On machines without `stormpy`, the JANI loader still works and PRISM tests are skipped.
- **Dev-shell `PYTHONPATH` / `sitecustomize.py` hack.** `flake.nix`'s `shellHook` prepends `$PWD` to `PYTHONPATH` so the repo's `sitecustomize.py` is auto-imported at Python startup; that file `ctypes`-preloads `libcuda.so.1` so jaxlib can find the host's CUDA driver. Side effect: this `sitecustomize.py` then shadows the one that ships inside each nix Python env, which is the file responsible for processing `NIX_PYTHONPATH` and adding the env's `site-packages` to `sys.path`. As a result, a nix-built Python helper launched from the dev shell can hit `ModuleNotFoundError` for a library that is in fact present in its env but was never registered on `sys.path`. `nix/plot.nix` works around this by invoking its python with `-I` (isolated mode), which drops `PYTHON*` env vars (so `PYTHONPATH=$PWD` no longer wins) while preserving `NIX_PYTHONPATH` (it does not start with `PYTHON`). Any new nix-built Python helper added to the flake should follow the same pattern, or it will silently break under `nix develop`.
