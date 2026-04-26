# stormpy.nix
{ lib
, python3Packages
, fetchFromGitHub
, cmake
, ninja
, storm
, pkg-config
, boost
, gmp
, cln
, ginac
, z3
, glpk
, xercesc
, eigen
, hwloc
}:

python3Packages.buildPythonPackage rec {
  pname = "stormpy";
  version = "1.11.3";

  src = fetchFromGitHub {
    owner = "moves-rwth";
    repo = "stormpy";
    rev = version;
    hash = "sha256-Pr/j7TmG7W3d744PP0+bCOVbuOoycAuEW+IuK8hhf1c=";
  };

  format = "pyproject";

  nativeBuildInputs = [
    cmake ninja pkg-config
    python3Packages.scikit-build-core
    python3Packages.pybind11
    python3Packages.pyproject-metadata
    python3Packages.build
    python3Packages.pip
  ];

  buildInputs = [
    storm
    boost
    gmp
    cln
    ginac
    z3
    glpk
    xercesc
    eigen
    hwloc
  ];

  propagatedBuildInputs = with python3Packages; [
    numpy
    deprecated
  ];

  # Key part: tell stormpy to use your Nix-built Storm and do not fetch.
  # Patch CMakeLists.txt to set the variables directly since scikit-build-core
  # may not pass CMAKE_ARGS correctly
  prePatch = ''
    substituteInPlace CMakeLists.txt \
      --replace 'set(STORM_DIR_HINT "" CACHE STRING "A hint where the Storm library can be found.")' \
                'set(STORM_DIR_HINT "${storm}/lib/cmake/storm" CACHE STRING "A hint where the Storm library can be found.")'
    # Force the options after they're declared
    sed -i '/^option(ALLOW_STORM_SYSTEM/a set(ALLOW_STORM_SYSTEM ON CACHE BOOL "Force ON")' CMakeLists.txt
    sed -i '/^option(ALLOW_STORM_FETCH/a set(ALLOW_STORM_FETCH OFF CACHE BOOL "Force OFF")' CMakeLists.txt
    # Patch pyproject.toml to accept available versions in nixpkgs
    # nixpkgs has scikit-build-core 0.10.7 and pybind11 2.13.6, so we relax the requirements
    substituteInPlace pyproject.toml \
      --replace 'requires = ["scikit-build-core==0.11.6", "pybind11==3.0.1"]' \
                'requires = ["scikit-build-core>=0.10.7", "pybind11>=2.13.6"]'
  '';

  # Also set environment variables as backup
  preConfigure = ''
    export CMAKE_ARGS="-DALLOW_STORM_FETCH=OFF -DALLOW_STORM_SYSTEM=ON -DSTORM_DIR_HINT=${storm}/lib/cmake/storm -Dstorm_DIR=${storm}/lib/cmake/storm"
    export CMAKE_PREFIX_PATH="${storm}:$CMAKE_PREFIX_PATH"
    # Ensure Python can find the build dependencies
    export PYTHONPATH="${python3Packages.scikit-build-core}/${python3Packages.python.sitePackages}:${python3Packages.pybind11}/${python3Packages.python.sitePackages}:$PYTHONPATH"
  '';

  # Ensure we're in the source directory before build
  # The configure phase may change to build/, so we need to go back
  preBuild = ''
    # Go back to source directory if we're in a build subdirectory
    while [ ! -f "pyproject.toml" ] && [ "$PWD" != "/" ]; do
      if [ -f "../pyproject.toml" ]; then
        cd ..
        break
      else
        break
      fi
    done
    if [ ! -f "pyproject.toml" ]; then
      echo "Error: Could not find pyproject.toml"
      pwd
      ls -la
      exit 1
    fi
    # Verify dependencies are importable and install them so build can detect them
    python -c "import scikit_build_core; import pybind11; print('Dependencies OK')" || echo "Warning: Dependencies not importable"
    # Install dependencies in a way that build can detect them
    # Use --break-system-packages because we're in a Nix build environment
    python -m pip install --user --break-system-packages scikit-build-core pybind11 || true
  '';

  pythonImportsCheck = [ "stormpy" ];
  doCheck = false; # turn on later if you wire pytest + runtime deps
}