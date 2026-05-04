{ lib, gnum4, stdenv, fetchFromGitHub, ocamlPackages }:

stdenv.mkDerivation rec {
  pname = "cudd";
  version = "1.0";

 src = fetchFromGitHub {
    owner = "SHoltzen";
    repo = "mlcuddidil";
    rev = "f102c9ac23a3ded8e20b911f27125d111509757a";
    sha256 = "sha256-1nT1tb2pKfu2xm0S8JsJ/Au7Ng8lyrwWbPN50dz5tpA=";
  };

  createFindlibDestdir = true;

  postPatch = ''
    cat >> sedscript_c <<'EOF'
s/exn = NULL;/exn = Val_unit;/;
EOF
  '';

  nativeBuildInputs = [
    ocamlPackages.ocaml
    ocamlPackages.camlidl
    ocamlPackages.findlib
    gnum4
  ];
}
