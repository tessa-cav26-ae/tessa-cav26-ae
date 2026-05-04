{ lib, fetchFromGitHub, pkgs, callPackage, ocaml-ng }:

let 
  # Dice uses OCaml 4.09. Since we only need the `dice` 
  # executable, we use our own OCaml version and libraries 
  # without interfering with Mappl's OCaml environment.
  ocamlPackages = ocaml-ng.ocamlPackages_4_09; 
in

let 
  cudd = callPackage ./mlcuddidl.nix { ocamlPackages = ocamlPackages; };  
in 

ocamlPackages.buildDunePackage rec {
  pname = "dice";
  version = "0ea228edac87f3ef7b0785c23786a3696b912c55";

  src = fetchFromGitHub {
    owner = "SHoltzen";
    repo = "dice";
    rev = version;
    hash = "sha256-mQsvsLCxq4nrA/J6yXjun/2wICy289saclOyQe6W/iQ=";
  };

  patches = [ ./dice.patch ];

  nativeBuildInputs = with ocamlPackages; [
    menhir
  ] ++ [
    pkgs.makeWrapper
  ];

  propagatedBuildInputs = with ocamlPackages; [
    menhir
    core
    ounit2
    ppx_sexp_conv
    sexplib
    core_bench
    ppx_deriving
    yojson
    ctypes
    bignum
    menhirLib
    ppx_jane
  ] ++ [ cudd ];

  # set stack size to 64 GB 
  postInstall = ''
      wrapProgram $out/bin/dice --run "ulimit -s ${builtins.toString (64 * 1024 * 1024 * 1024)}"
  '';

}
