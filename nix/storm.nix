# storm.nix
{ stdenv
, lib
, fetchFromGitHub
, fetchFromGitLab
, fetchurl
, cmake
, git
, python3
, boost
, gmp
, cln
, ginac
, z3
, glpk
, xercesc
, eigen
, hwloc
, autoconf
, automake
, libtool
, curl
, cacert
, pkg-config
, gnumake
, gcc
, which
}:

let
  # Pre-fetch all git dependencies
  carl-storm = fetchFromGitHub {
    owner = "moves-rwth";
    repo = "carl-storm";
    rev = "14.33";
    sha256 = "sha256-hE/cuvLV7HuDOQ2e7rBNx2S15KnSwQambKou/HRpW/8=";
    fetchSubmodules = false;
  };

  l3pp = fetchFromGitHub {
    owner = "hbruintjes";
    repo = "l3pp";
    rev = "513c7589232396e4b8975113b7709a8f39b46c85";
    sha256 = "sha256-f3PpH+nKv5eVOMSGySSCLg+5uWQ520oS3oaxuon8DwU=";
  };

  eigen-src = fetchFromGitLab {
    owner = "libeigen";
    repo = "eigen";
    rev = "bae907b8f6078b1df290729eef946360315bd312";
    sha256 = "sha256-hqrrBOZ1MxTMKyXyu5CxrZc8m/cisr+fJSEQwHWqQLA=";
    domain = "gitlab.com";
  };

  gmm = fetchurl {
    url = "https://download-mirror.savannah.gnu.org/releases/getfem/stable/gmm-5.4.4.tar.gz";
    sha256 = "sha256-FesZQwEbkmZaqzsC7PPO3hz4nqFakAb4HyuizWYqoCs=";
  };
in

stdenv.mkDerivation rec {
  pname = "storm";
  version = "1.11.1";

  # https://github.com/moves-rwth/storm
  src = fetchFromGitHub {
    owner = "moves-rwth";
    repo = "storm";
    rev = version;
    sha256 = "sha256-2ClVJt/Pj1Oatr9muep+QD/R+LUmmsRPDCFY4Ec0UTg=";
  };

  nativeBuildInputs = [
    cmake
    git
    python3
    autoconf
    automake
    libtool
    pkg-config
    gnumake
    which
  ];

  buildInputs = [
    boost
    gmp
    cln
    ginac
    z3
    glpk
    xercesc
    eigen
    hwloc
    curl
    cacert
  ];

  # Patch CMake to use pre-fetched sources
  prePatch = ''
    l3pp_src="${l3pp}"
    eigen_src="${eigen-src}"
    gmm_tar="${gmm}"
    
    # Patch carl FetchContent to use local source
    # Replace the entire FETCHCONTENT_DECLARE block
    substituteInPlace resources/3rdparty/CMakeLists.txt \
      --replace \
      'FETCHCONTENT_DECLARE(
        carl
        GIT_REPOSITORY ''${STORM_CARL_GIT_REPO}
        GIT_TAG ''${STORM_CARL_GIT_TAG}
)' \
      'FETCHCONTENT_DECLARE(
        carl
        SOURCE_DIR "$ENV{NIX_BUILD_TOP}/carl-storm-source"
)'
    
    # Patch l3pp - replace GIT_REPOSITORY and GIT_TAG with DOWNLOAD_COMMAND
    sed -i \
      -e '/GIT_REPOSITORY https:\/\/github.com\/hbruintjes\/l3pp.git/,/GIT_TAG 513c7589232396e4b8975113b7709a8f39b46c85/c\
        DOWNLOAD_COMMAND ''${CMAKE_COMMAND} -E copy_directory "'"$l3pp_src"'" ''${STORM_3RDPARTY_BINARY_DIR}/l3pp' \
      resources/3rdparty/CMakeLists.txt
    
    # Create a CMake script for gmm download (expand shell variable)
    gmm_tar_expanded="$gmm_tar"
    cat > resources/download-gmm.cmake <<EOF
execute_process(
  COMMAND ''${CMAKE_COMMAND} -E make_directory ''${STORM_3RDPARTY_BINARY_DIR}/gmm
  COMMAND ''${CMAKE_COMMAND} -E tar xzf "$gmm_tar_expanded" --format=gnutar -C ''${STORM_3RDPARTY_BINARY_DIR}/gmm --strip-components=1
)
EOF
    
    # Patch gmm - replace URL with DOWNLOAD_COMMAND  
    sed -i \
      -e 's|URL https://download-mirror.savannah.gnu.org/releases/getfem/stable/gmm-''${GMM_VERSION}.tar.gz|DOWNLOAD_COMMAND ''${CMAKE_COMMAND} -P ''${PROJECT_SOURCE_DIR}/resources/download-gmm.cmake|' \
      resources/3rdparty/CMakeLists.txt
    
    # Patch eigen - replace GIT_REPOSITORY and GIT_TAG
    sed -i \
      -e '/GIT_REPOSITORY https:\/\/gitlab.com\/libeigen\/eigen.git/,/GIT_TAG bae907b8f6078b1df290729eef946360315bd312/c\
        DOWNLOAD_COMMAND ''${CMAKE_COMMAND} -E copy_directory "'"$eigen_src"'" ''${STORM_3RDPARTY_BINARY_DIR}/StormEigen' \
      resources/3rdparty/CMakeLists.txt

    # Drop the "Current working directory: <abs path>" banner line that Storm's
    # printHeader() emits unconditionally on every invocation. Storm provides
    # no flag to suppress it, and the absolute-cwd leak ends up in every
    # captured stdout (harness logs, transpiler-generated .prism.py comments,
    # etc.). The deletion targets the single STORM_PRINT in print.cpp and
    # leaves the rest of the banner (Storm version, Date, Command line) intact.
    sed -i '/STORM_PRINT("Current working directory:/d' \
      src/storm-cli-utilities/print.cpp
  '';

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DSTORM_PORTABLE=OFF"
    "-DSTORM_BUILD_TESTS=OFF"
    "-DSTORM_BUILD_EXECUTABLES=ON"
    "-DSTORM_DISABLE_SPOT=ON"
    "-DSTORM_DISABLE_GMM=ON"
    "-DSTORM_USE_CLN_EA=OFF"
    "-DSTORM_USE_CLN_RF=ON"
    "-DSTORM_CARL_GIT_TAG=14.33"
    "-DSTORM_RESOURCES_BUILD_JOBCOUNT=1"
  ];

  GIT_SSL_CAINFO = "${cacert}/etc/ssl/certs/ca-bundle.crt";
  GIT_TERMINAL_PROMPT = "0";

  enableParallelBuilding = true;
    dontFixCmake = true;

  # Copy carl-storm to a writable location (CMake needs to write configure files)
  # Also set up eigen for carl-storm and patch its CMakeLists.txt
  preConfigure = ''
    mkdir -p $NIX_BUILD_TOP/carl-storm-source
    cp -r ${carl-storm}/* $NIX_BUILD_TOP/carl-storm-source/
    chmod -R u+w $NIX_BUILD_TOP/carl-storm-source
    
    # Copy eigen for carl-storm (carl-storm also fetches eigen)
    mkdir -p $NIX_BUILD_TOP/eigen-carl-source
    cp -r ${eigen-src}/* $NIX_BUILD_TOP/eigen-carl-source/
    chmod -R u+w $NIX_BUILD_TOP/eigen-carl-source
    
    # Patch carl-storm's eigen3.cmake file to use local eigen source
    if [ -f "$NIX_BUILD_TOP/carl-storm-source/resources/eigen3.cmake" ]; then
      # Replace GIT_REPOSITORY and GIT_TAG with DOWNLOAD_COMMAND that copies from our source
      # Note: CMAKE_COMMAND and PROJECT_BINARY_DIR are CMake variables, so they need to be in CMake syntax
      sed -i \
        -e 's|GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git|DOWNLOAD_COMMAND ''${CMAKE_COMMAND} -E copy_directory "$ENV{NIX_BUILD_TOP}/eigen-carl-source" ''${PROJECT_BINARY_DIR}/resources/Eigen|g' \
        -e '/GIT_TAG.*bae907b8f6078b1df290729eef946360315bd312/d' \
        "$NIX_BUILD_TOP/carl-storm-source/resources/eigen3.cmake"
    fi
    export FETCHCONTENT_SOURCE_DIR_eigen_carl_src=$NIX_BUILD_TOP/eigen-carl-source
  '';

  meta = with lib; {
    description = "A Modern Probabilistic Model Checker";
    homepage = "https://www.stormchecker.org/";
    license = licenses.gpl3Plus;
    platforms = platforms.linux;
    maintainers = [ ];
  };
}