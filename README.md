CAV 2026 Artifact
=======================================
Paper title: Tensor Probabilistic Model Checking of Finite-Horizon Markov Chains

Claimed badges: Available + Functional + Reusable

Justification for the badges:

* Functional:

  The artifact replicates all of the experimental figures in the paper. 

  It builds Tessa (our tool), Storm (used as the baseline for both the MTBDD and sparse engines), and Rubicon (a PRISM → Dice compiler plus the Dice inference backend), and runs the two scaling sweeps (`testn` size scaling, `testh` horizon scaling) for each of the four benchmark families.
  
  In addition, the kydice parametric workload exercises Tessa's parameter synthesis functionality and produces Figures 8(b) and 8(c).
  
  The end-to-end driver `reproduce.mk` runs Tessa, Storm-ADD, Storm-SPM, and Rubicon (via the bundled PRISM → Dice compiler), cross-checks Tessa's reachability probabilities against Storm and Rubicon (`src.postprocess verify`, tolerance `atol=1e-5`, `rtol=1e-4`), and emits the scaling plots used in the paper.
  
  Pre-shipped reference outputs are included so reviewers can diff their own runs against ours.

  - replicated:

      * [Figure 1(c) left](reproduced-0429/meeting/testn/meeting-testn.png)
      * [Figure 1(c) right](reproduced-0429/meeting/testh/meeting-testh.png)
      * [Figure 5(a)](reproduced-0429/parqueues/testq/parqueues-testq.png)
      * [Figure 5(b)](reproduced-0429/parqueues/testq/parqueues-testq-tessa.png)
      * [Figure 5(c)](reproduced-0429/parqueues/testh/parqueues-testh.png)
      * [Figure 5(d)](reproduced-0429/parqueues/testh/parqueues-testh-tessa.png)
      * [Figure 6(a)](reproduced-0429/weather-factory/testn/weather-factory-testn.png)
      * [Figure 6(b)](reproduced-0429/weather-factory/testn/weather-factory-testn-tessa.png)
      * [Figure 6(c)](reproduced-0429/weather-factory/testh/weather-factory-testh.png)
      * [Figure 6(d)](reproduced-0429/weather-factory/testh/weather-factory-testh-tessa.png)
      * [Figure 7(a)](reproduced-0429/herman/testn/herman-testn.png)
      * [Figure 7(b)](reproduced-0429/herman/testn/herman-testn-tessa.png)
      * [Figure 7(c)](reproduced-0429/herman/testh/herman-testh.png)
      * [Figure 7(d)](reproduced-0429/herman/testh/herman-testh-tessa.png)
      * [Figure 8(b)](benchmarks/kydice/loss.png)
      * [Figure 8(c)](benchmarks/kydice/landscape.png)
      * Cross-tool correctness: `reproduced-0429/<suite>/<test>/verify.csv` (Tessa vs Storm-ADD, Storm-SPM, and Rubicon probabilities)
      * Rubicon/Dice numbers (PRISM → Dice transpilation + Dice inference): `reproduced-0429/<suite>/<test>/rubicon.csv`, reproduced by `make -f reproduce.mk rubicon`. Rubicon and Dice are bundled by the flake (`nix/rubicon.nix`, `nix/dice.nix`) and exposed via `rubicon-shell` (and transitively `tessa-shell`).
      * Geni numbers (gennifer interpreting a generated `.gir` program) for every suite in the paper: bundled for `weather-factory` (`reproduced-0429/weather-factory/<test>/geni.csv`, reproduced by `make -f reproduce.mk geni` or `make -f reproduce.mk weather-factory`).

* Reusable: 

  The artifact ships under the MIT License (see `LICENSE`).
  Tessa's source lives under `src/`; benchmark models under `benchmarks/`; a regression test suite under `tests/` (`pytest`).
  
  The build is fully pinned: `flake.nix` + `flake.lock` reproduce the exact toolchain (Storm, stormpy, Rubicon, Dice, Gennifer, JAX with CUDA, Python 3.12), and the `Dockerfile` wraps the same flake in a `nixos/nix` image for hosts without Nix. The flake exposes `rubicon`, `dice`, and `gennifer` as packages alongside `storm`/`stormpy`/`tessa`, so the external comparison points reproduce inside the same toolchain.
  
  Examples in (`examples`) e.g. (`examples/complex_multi_action.prism` and `tessa/examples/complex_multi_action.jani`) demonstrates Tessa's usage beyond the paper figures.

Machine Configuration and Time Consumption:

  * RAM: 128 GB 
  * CPU cores: 8 
  * GPU: NVIDIA 2080 Ti
  * Time (installation): ~1.5 h
  * Time (smoke test): ~20 m
  * Time (full review): ~50 h

external connectivity: NO

  The core reproduction does not need network access at run time. 
  
  Network is required only during the one-time install (the `Dockerfile` / `flake.nix` build pulls Nix substituters and source tarballs for Storm, JAX, and CUDA).


## Table of Contents

This repository contains the tool source code, benchmarks, and instructions to reproduce the results in the paper.

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
- **Nix** package manager (recommended — provides Storm, JAX with CUDA, and all other dependencies via `flake.nix`), *or* Python 3.12 with pip (see [Option 3](#option-3-nix-storm--pip-tessa) below)

### Choose your GPU

The Nix path bundles its own CUDA runtime for compute capabilities 6.0–12.0 (Pascal through Blackwell). 

Tessa is tested and experimental resultes are reproduced on a **Turing host (CUDA compute capability 7.5)**.


| Architecture | Compute cap | Example cards               | pip path                        | Nix path                          |
| ------------ | ----------- | --------------------------- | ------------------------------- | --------------------------------- |
| Pascal       | 6.0 / 6.1   | P100, GTX 1080, GTX 1080 Ti | works                           | works out of the box              |
| Volta        | 7.0         | V100, Titan V               | works                           | works out of the box              |
| Turing       | 7.5         | RTX 2080, RTX 2080 Ti, T4   | works                           | works out of the box              |
| Ampere       | 8.0 / 8.6   | A100, RTX 3090, RTX 3080    | works                           | works out of the box              |
| Ada          | 8.9         | RTX 4090, L40, L4           | works                           | works out of the box              |
| Hopper       | 9.0         | H100                        | works                           | works out of the box              |
| Blackwell    | 10.0 / 12.0 | B200, RTX 5090              | works (with a recent JAX wheel) | works out of the box              |

Look up your card's compute capability at <https://developer.nvidia.com/cuda-gpus>.

The Nix-based paths (Options 1 and 2) ship a default `cudaCapabilities` list in `flake.nix` (line 14) covering every card in the table above. **The first build compiles JAX/NCCL kernels for every entry in the list**, so narrowing it to your card's capability before `nix develop` (Option 1) or `docker build` (Option 2) noticeably shortens the build:

```diff
 # flake.nix (around line 14)
 config = {
   allowUnfree = true;
-  cudaCapabilities = [ "6.0" "6.1" "7.0" "7.5" "8.0" "8.6" "8.9" "9.0" "10.0" "12.0" ];
+  cudaCapabilities = [ "7.5" ];  # your card's compute capability (e.g. RTX 2080 Ti = 7.5)
   cudaForwardCompat = true;
   allowUnsupportedSystem = true;
   allowBroken = true;
 };
```

Default ships all 10 archs for portability.

### Verified host

The Nix development shell in `flake.nix` is tested on the machine below.
`flake.nix` pins `cudaPackages_12_8` and builds JAX with `cudaSupport = true` for compute capabilities 6.0–12.0 (Pascal through Blackwell), with `cudaForwardCompat = true`. The 12.8 build runs on any 12.x NVIDIA driver (≥ 525) via CUDA minor-version compatibility, including the host's driver 535.

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

### Pick an install path

Three supported paths. All three need the NVIDIA driver above; pick one for the rest of the toolchain.

| Path                                                                  | What you get                                                                | When to use                                                                                                                  |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| [Option 1: Nix (recommended)](#option-1-nix-recommended)              | `tessa-shell`: Tessa, Storm, `stormpy`, JAX/CUDA, `plot`                    | Default. Full paper reproduction on a host where installing Nix is acceptable.                                               |
| [Option 2: Docker](#option-2-docker)                                  | Same shell as Option 1, inside a `nixos/nix` container                      | You'd rather not install Nix on the host but have Docker + NVIDIA Container Toolkit.                                         |
| [Option 3: Nix (Storm) + pip (Tessa)](#option-3-nix-storm--pip-tessa) | `storm-shell` (Storm CLI + `plot`) plus a pip venv with JAX + Tessa         | You'd rather manage Tessa/JAX yourself with `pip` and only need Nix for the Storm (no `stormpy`) baseline. JANI models only. |

### Option 1: Nix (recommended)

Install Nix by running:
```
sh <(curl -L https://nixos.org/nix/install) --daemon
```
Please refer to https://nixos.org/download for more info.

The flake builds JAX for compute capabilities 6.0–12.0 (Pascal through Blackwell), so every shipped NVIDIA card works without editing `flake.nix`. For a faster first build, narrow `cudaCapabilities` in `flake.nix` (line 14) to just your card's capability — see [Choose your GPU](#choose-your-gpu).

The flake exposes four development shells, each layered on top of the previous (`storm-shell` ⊆ `geni-shell` ⊆ `rubicon-shell` ⊆ `tessa-shell`):

| Shell                   | Contents                                                                              | When to use                                                                                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `storm-shell`           | `storm` CLI + `plot`                                                                  | Lightweight — you bring Tessa via `pip` and use `.jani` models so `stormpy` isn't needed (see [Option 3](#option-3-nix-storm--pip-tessa)) |
| `geni-shell`            | `storm-shell` + `gennifer`                                                            | Adds the Geni compiler                                                                                                                                 |
| `rubicon-shell`         | `geni-shell` + `rubicon` + `dice`                                                     | Adds the Rubicon (PRISM → Dice) baseline used by `reproduce.mk`                                                                                          |
| `tessa-shell` (default) | `rubicon-shell` + `stormpy`, `tessa`, Python 3.12 + JAX/CUDA                          | Full environment — paper reproduction (all four tools), PRISM models, interactive Tessa work                                                             |

Enter the full development shell (this is the default):
```
nix --experimental-features 'nix-command flakes' develop -c fish
# equivalent to:
nix --experimental-features 'nix-command flakes' develop .#tessa-shell -c fish
```

Or enter one of the lighter shells:
```
nix --experimental-features 'nix-command flakes' develop .#rubicon-shell -c fish
nix --experimental-features 'nix-command flakes' develop .#geni-shell    -c fish
nix --experimental-features 'nix-command flakes' develop .#storm-shell   -c fish
```

The `-c fish` flag launches fish inside the dev shell — fish is shipped in `common-shell-inputs` of every shell variant (`flake.nix` line 47), and the rest of this README's command examples assume fish. Drop `-c fish` if you prefer bash; `nix develop` defaults to bash without it.

The first build of `tessa-shell` compiles Storm, stormpy, Rubicon, Dice, JAX, CUDA from source, which can take hours. `rubicon-shell` builds Storm, Rubicon, Dice, and Gennifer but skips JAX/CUDA, so its first build is meaningfully shorter; `storm-shell` only builds Storm and is shortest.

Verify `tessa-shell` is working:
```console
> tessa examples/weather_factory_3.jani --property allStrike --horizon 10
```

### Option 2: Docker

If you'd rather not install Nix on the host, the repository ships a `Dockerfile` that builds Storm, `stormpy`, and `tessa` inside a `nixos/nix` image.

Prerequisites:
- [Docker](https://docs.docker.com/get-docker/).
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) — required for `--gpus all` passthrough. Tessa's default backend is CUDA-JAX.

Build the image:
```shell
> docker build -t tessa .
[+] Building 2854.1s (12/12) FINISHED                    docker:default
 => [internal] load build definition from Dockerfile               0.1s
 => => transferring dockerfile: 803B                               0.0s
 => [internal] load metadata for docker.io/nixos/nix:latest        0.3s
 => [internal] load .dockerignore                                  0.1s
 => => transferring context: 2B                                    0.0s
 => [1/7] FROM docker.io/nixos/nix:latest@sha256:e2fe74            0.0s
 => [internal] load build context                                  0.3s
 => => transferring context: 149.38kB                              0.2s
 => [2/7] RUN mkdir -p /etc/nix &&                                 5.2s
 => [3/7] COPY . /tessa                                            1.5s
 => [4/7] WORKDIR /tessa                                           0.4s
 => [5/7] RUN nix build .#storm --cores 4 --print-build-logs    1167.9s
 => [6/7] RUN nix build .#stormpy --cores 4 --print-build-logs   352.3s
 => [7/7] RUN nix build .#tessa --cores 4 --print-build-logs    3693.5s
 => exporting to image                                            72.6s
 => => exporting layers                                           72.3s
 => => writing image sha256:3d3291                                 0.0s
 => => naming to docker.io/library/tessa                           0.1s
```
The build takes ~1.5h because Storm, `stormpy`, and CUDA-enabled JAX are compiled from scratch inside the container (no host Nix store to reuse). Subsequent builds reuse Docker's layer cache.

The shipped `Dockerfile` only pre-builds `storm`, `stormpy`, and `tessa`. Rubicon, Dice, and Gennifer are pulled in by the `tessa-shell` `buildInputs` and so build the first time you run `nix develop` (or `nix develop .#rubicon-shell`) inside the container. To pre-bake them into the image, add `RUN nix build .#rubicon .#dice .#gennifer --cores 4 --print-build-logs` to the `Dockerfile` next to the existing `nix build` lines.

**Tune `--cores N` in the `Dockerfile` (lines 12–14).** The shipped value is `4`. NCCL (pulled in by the `tessa` step via JAX) compiles each `.cu` for all 10 CUDA archs in one NVCC call, so parallel jobs are memory-heavy. Raise it on a beefier host; if the `tessa` step OOMs (`builder failed due to signal 9` inside `all_reduce_*.cu`), drop to `2` or `1`. `storm` and `stormpy` don't hit NCCL and can stay higher.

**Narrow `cudaCapabilities` in `flake.nix` (line 14)** to your card's capability before `docker build` for a faster build — see [Choose your GPU](#choose-your-gpu).

Run the image — the `Dockerfile`'s `CMD` is `nix develop`, so this drops straight into the Nix shell. `--rm` removes the container on exit so they don't pile up; drop `--rm` if you want the stopped container to be recoverable later via `docker start`.
```shell
> mkdir -p smoke reproduced
> docker run -it --rm --gpus all \
    -v "$(pwd)/smoke:/tessa/smoke" \
    -v "$(pwd)/reproduced:/tessa/reproduced" \
    tessa
```

The two `-v` mounts expose the in-container output directories `/tessa/smoke` and `/tessa/reproduced` to host paths of the same name, so `smoke/parqueues/testq/tessa.csv` and friends are readable from the host as soon as `make` writes them. The `mkdir -p` line runs first because docker would otherwise create the host paths as root, which then fights with `make` runs from the host shell. Drop a mount if you only plan to run one of the two flows.

Or start the container detached and re-enter it later. `--rm` is fine for the detach/re-enter loop because `ctrl-p ctrl-q` only detaches (the container keeps running) — drop `--rm` if you want the container to survive an `exit` of the inner shell so you can `docker start` it back up across sessions:
```shell
> docker run -it --rm --gpus all \
    -v "$(pwd)/smoke:/tessa/smoke" \
    -v "$(pwd)/reproduced:/tessa/reproduced" \
    tessa #(and then detach by ctrl-p ctrl-q)
> docker ps
CONTAINER ID    IMAGE    COMMAND    CREATED    STATUS    PORTS    NAMES
3ac16580a16e    tessa    "nix …"    ....       ....      ....     ....
> docker exec -it 3ac1 bash -c "cd /tessa && nix develop -c fish"
```

The `Dockerfile`'s `CMD` is `nix develop -c fish`, so the interactive `docker run` lands in fish; `docker exec` re-entries should pass `-c fish` to match (commands in the rest of this README assume the fish shell).

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

### Option 3: Nix (Storm) + pip (Tessa)

Manage Tessa and JAX yourself in a pip venv outside Nix, and pull only the `storm` CLI in from the flake's lightweight `storm-shell` for the baseline used by `reproduce.mk`. This path does **not** install `stormpy`, so Tessa runs against `.jani` models only — PRISM loading is unavailable. Every benchmark suite (and both files in `examples/`) ships a `.jani` alongside the `.prism`, and `reproduce.mk` already passes `--model-type jani` to Tessa.

Prerequisites:
- Python **3.12** (hard pin in `pyproject.toml`: `requires-python = ">=3.12,<3.13"`).
- An NVIDIA driver compatible with the JAX CUDA wheel you install (verify with `nvidia-smi`; see [Install the NVIDIA driver](#install-the-nvidia-driver)). JAX's `jax[cuda]` wheel bundles the matching CUDA and cuDNN runtime, so a system-wide CUDA toolkit is **not** required. Pick the `jax[cuda...]` extra that matches your driver (see <https://jax.readthedocs.io/en/latest/installation.html> for the current list of supported CUDA versions).

**Step A — Tessa side (pip venv).** In a venv outside Nix, install JAX and Tessa:
```shell
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U "jax[cuda]"
pip install -e .
tessa --help                                  # verify the CLI is on PATH
python -c "import jax; print(jax.devices())"  # should list CudaDevice(...)
```

**Step B — Storm side (Nix).** Enter the lightweight `storm-shell` — no Tessa, no `stormpy`, no JAX, no Python toolchain, just the `storm` CLI and `plot`:
```shell
nix --experimental-features 'nix-command flakes' develop .#storm-shell
storm --version                               # verify the CLI is on PATH
```

If you only need Tessa itself and not the Storm baseline, skip Step B.

Smoke-test the JANI workflow from the pip venv:
```shell
tessa examples/weather_factory_3.jani --property allStrike --horizon 10
```

For the full paper reproduction, run `make -f reproduce.mk` from a shell where both `tessa` (pip venv) and `storm`/`plot` (`storm-shell`) are on `$PATH`. The simplest way is to enter `storm-shell` first, then `source .venv/bin/activate` inside it. Because `storm-shell` does not put its own Python on `$PATH`, the pip venv's Python wins and there is no Nix-Python ↔ pip-Python collision.

## Experimental Evaluation

**Performance might be different on different hardware configurations, but the scaling trends should be similar.**

Benchmark scripts to reproduce the paper figures are driven by `reproduce.mk`.
Four tools are compared on every suite: **Tessa** (JAX CUDA), **Storm ADD** (MTBDD engine), **Storm SPM** (Sparse engine), and **Rubicon** (PRISM → Dice compiler + Dice inference). **Geni** (gennifer interpreting a generated `.gir` program) is additionally included for the **weather-factory** suite.

| Benchmark        | Size scaling                                                                                                                                                            | Horizon scaling                                                                                                                                                         |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Meeting          | [Figure 1(c) - left](reproduced-0429/meeting/testn/meeting-testn.png)                                                                                                  | [Figure 1(c) - right](reproduced-0429/meeting/testh/meeting-testh.png)                                                                                                 |
| ParQueues        | [Figure 5(a)](reproduced-0429/parqueues/testq/parqueues-testq.png)             • [Figure 5(b)](reproduced-0429/parqueues/testq/parqueues-testq-tessa.png)             | [Figure 5(c)](reproduced-0429/parqueues/testh/parqueues-testh.png)             • [Figure 5(d)](reproduced-0429/parqueues/testh/parqueues-testh-tessa.png)             |
| Weather Factory  | [Figure 6(a)](reproduced-0429/weather-factory/testn/weather-factory-testn.png) • [Figure 6(b)](reproduced-0429/weather-factory/testn/weather-factory-testn-tessa.png) | [Figure 6(c)](reproduced-0429/weather-factory/testh/weather-factory-testh.png) • [Figure 6(d)](reproduced-0429/weather-factory/testh/weather-factory-testh-tessa.png) |
| Herman           | [Figure 7(a)](reproduced-0429/herman/testn/herman-testn.png)                   • [Figure 7(b)](reproduced-0429/herman/testn/herman-testn-tessa.png)                   | [Figure 7(c)](reproduced-0429/herman/testh/herman-testh.png)                   • [Figure 7(d)](reproduced-0429/herman/testh/herman-testh-tessa.png)                   |

### Wall-clock time for the full reproduction

Approximate wall-clock time spent on each `bench.suite.tool` slice on the verified host (taken from the first→last timestamps in `reproduced/<suite>/<test>/*.log`).
Tessa totals include warmup plus the configured number of timed runs; Storm totals are sequential subprocess invocations, one per parameter point.
Cells where Storm ADD exits `FAILED` on the first parameter point (meeting.testh, weather-factory.testh) finish quickly but produce no useful data beyond H ≤ 2 — flagged explicitly.

| Suite           | Test  |  Tessa  | Storm ADD| Storm SPM |  Rubicon  |   Geni   |
| --------------- | ----- | ------: | -------: | --------: | --------: | -------: |
| Herman          | testn |      1m |   1h 34m |       30m |     1h 6m |    —     |
| Herman          | testh |      4m |   6h 32m |    5h  9m |    2h 53m |    —     |
| Meeting         | testn |      1m |      20m |       11m |       18m |    —     |
| Meeting         | testh |      5m |       1m |    5h 51m |    1h 34m |    —     |
| Weather Factory | testn |      1m |      14m |       52m |       26m |    30m   |
| Weather Factory | testh |      6m |       3m |    4h 59m |    2h 22m |  4h 0m   |
| ParQueues       | testq |      1m |      58m |       28m |       47m |    —     |
| ParQueues       | testh |      4m |   4h 24m |       41m |     2h 7m |    —     |

Totals: Tessa ≈ 0.5 h, Storm ADD ≈ 14.2 h, Storm SPM ≈ 18.7 h, Rubicon ≈ 11.6 h, Geni ≈ 4.5 h (weather-factory only).
Per-point timeout is `TO=1260` s (see top of `reproduce.mk`); parameter points that hit the timeout are recorded as `TIMEOUT` in the CSVs.

### Quick Smoke Test

Runs a minimal parameter subset (~20 minutes) to verify the pipeline:

```shell
make -f reproduce.mk SMOKE=1
```

A reference smoke output is committed at [`smoke-0504/`](smoke-0504/) (119 files, 8 leaf directories under `herman/`, `meeting/`, `weather-factory/`, `parqueues/`). Compare your `smoke/` against it:

- `diff <(cd smoke && find . | sort) <(cd smoke-0504 && find . | sort)` — same tree, same filenames.
- `cut -d, -f1-13 smoke/parqueues/testq/tessa.csv` matches the same columns of `smoke-0504/parqueues/testq/tessa.csv` (timing columns will differ by host; identifiers, parameters, `status`, and `probability` should not).
- All `verify.csv` rows show `status=ok` (probabilities agree across tools within `atol=1e-5, rtol=1e-4`).

`status=TIMEOUT` / `status=FAILED` rows or `verify.csv` mismatches mean the pipeline ran but the result is not clean — usually a host slower than the smoke `TO=60` s budget, or a tool's nix shell missing from `PATH`.

### Full Reproduction

Runs all benchmarks with the full parameter ranges from the paper (~50 hours):

```shell
make -f reproduce.mk
```

### Per-Tool Commands

To rerun all suites against just one tool — useful when only one tool's CSVs need refreshing — use the tool-aggregate targets:

```shell
make -f reproduce.mk tessa      # ~26 minutes -  Tessa across all suites
make -f reproduce.mk storm-add  # ~14.2 hours -  Storm ADD across all suites
make -f reproduce.mk storm-spm  # ~18.7 hours -  Storm SPM across all suites
make -f reproduce.mk rubicon    # ~11.6 hours -  Rubicon across all suites
make -f reproduce.mk geni       # ~4.5 hours  -  Geni across the weather-factory suite (the only suite Geni runs against in this artifact)
make -f reproduce.mk            # reuses the CSVs produced by the per-tool commands above; verifies Tessa probabilities against Storm, Rubicon, and Geni and emits PNG scaling plots
```

Finer-grained slices follow the pattern `<suite>.<test>.<tool>`, e.g. `herman.testn.tessa`, `parqueues.testh.storm-spm`, or `meeting.testh.rubicon`. The eight Rubicon slices are `herman.test{n,h}.rubicon`, `meeting.test{n,h}.rubicon`, `weather-factory.test{n,h}.rubicon`, and `parqueues.test{q,h}.rubicon`.

Each run target has a matching `clean.*` target that drops only that target's CSV and log. Use these to force a rerun of one slice without nuking the rest of the output tree:

```shell
make -f reproduce.mk clean.tessa                # drop all Tessa CSVs/logs
make -f reproduce.mk clean.storm-add            # drop all Storm ADD CSVs/logs
make -f reproduce.mk clean.storm-spm            # drop all Storm SPM CSVs/logs
make -f reproduce.mk clean.rubicon              # drop all Rubicon CSVs/logs
make -f reproduce.mk clean.geni                 # drop the weather-factory Geni CSVs/logs/cache
```

For iterating on Tessa without re-running Storm baselines, use `tessa.all` — it runs the Tessa sweep, then re-runs verification and plotting against the existing `storm.add.csv` / `storm.spm.csv` on disk:

```shell
make -f reproduce.mk tessa.all                  # ~27 minutes
```

### Results

Results are written to `reproduced/<benchmark>/<test>/` (or `smoke/` with `SMOKE=1`).
Each sub-test directory contains:
- `tessa.csv`, `storm.add.csv`, `storm.spm.csv`, `rubicon.csv` — timing data (plus `geni.csv` for `weather-factory`)
- `tessa.log`, `storm.add.log`, `storm.spm.log`, `rubicon.log` — per-tool run logs (plus `geni.log` for `weather-factory`)
- `*.png` — scaling plots comparing the tools
- `verify.csv`, `verify.log` — correctness check (Tessa vs Storm, Rubicon, and — for `weather-factory` — Geni probabilities)

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

- **`ImportError: stormpy` (Option 3).** Expected — Option 3 does not install `stormpy`. Either switch to `tessa-shell` (Option 1, `nix develop`), or feed Tessa a `.jani` model (the JANI loader is pure Python and works without `stormpy`).

- **`ValueError: Could not infer model type from file extension`.** Model file does not end in `.prism`, `.pm`, `.nm`, or `.jani`. Pass `--type jani` or `--type prism` explicitly.

- **`nvidia-smi: command not found`.** Driver not installed (or PATH issue). Install the NVIDIA driver from your distro's package manager or <https://developer.nvidia.com/cuda-downloads> (driver-only is sufficient).

If you hit something not listed above, please open an issue with the output of `nvidia-smi`, `python -c "import jax; print(jax.devices(), jax.__version__)"`, and the failing command's full traceback.


## Notes

- PRISM loading uses `stormpy` when available.
- In this repo, the intended environment is the Linux/Nix setup from `flake.nix`.
- On machines without `stormpy`, the JANI loader still works and PRISM tests are skipped.
- **Dev-shell `PYTHONPATH` / `sitecustomize.py` hack.** `flake.nix`'s `shellHook` prepends `$PWD` to `PYTHONPATH` so the repo's `sitecustomize.py` is auto-imported at Python startup; that file `ctypes`-preloads `libcuda.so.1` so jaxlib can find the host's CUDA driver. Side effect: this `sitecustomize.py` then shadows the one that ships inside each nix Python env, which is the file responsible for processing `NIX_PYTHONPATH` and adding the env's `site-packages` to `sys.path`. As a result, a nix-built Python helper launched from the dev shell can hit `ModuleNotFoundError` for a library that is in fact present in its env but was never registered on `sys.path`. `nix/plot.nix` works around this by invoking its python with `-I` (isolated mode), which drops `PYTHON*` env vars (so `PYTHONPATH=$PWD` no longer wins) while preserving `NIX_PYTHONPATH` (it does not start with `PYTHON`). Any new nix-built Python helper added to the flake should follow the same pattern, or it will silently break under `nix develop`.
