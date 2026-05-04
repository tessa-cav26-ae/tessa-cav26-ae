{ lib, fetchFromGitHub, rustPlatform, pkgs, applyPatches}:

rustPlatform.buildRustPackage rec {

  pname = "gennifer";
  version = "9e5ffb25548258849ce58105feef395514c8df94";

  # https://github.com/geni-icfp25-ae/geni-icfp25-ae/commit/9e5ffb25548258849ce58105feef395514c8df94
  src = fetchFromGitHub {
    owner = "geni-icfp25-ae";
    repo = "geni-icfp25-ae";
    rev =  version;
    hash = "sha256-lDOxBpoTq6n/1R1LIB+LeiHtGKpHsufOKvt30nPgOgI=";
  };

  cargoHash = "sha256-W9YAXdMxARa5mQ+WGg/UhtRlTeGG56MP9LMJwxwhmak=";
  
  nativeBuildInputs = with pkgs; [
    gnum4
  ];
  
  meta = {
    license = lib.licenses.mit;
    maintainers = [ ];
  };
}
