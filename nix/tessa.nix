{ lib, python312, stormpy ? null }:

python312.pkgs.buildPythonPackage rec {
  pname = "tessa";
  version = "0.1.0";

  pyproject = true;
  src = ../.;

  build-system = with python312.pkgs; [
    setuptools
    wheel
  ];

  propagatedBuildInputs = with python312.pkgs; [
    click
    jax
    jaxlib
    numpy
    optax
  ] ++ lib.optionals (stormpy != null) [
    stormpy
  ];

  pythonImportsCheck = [ "tessa" "tessa.cli" ];
}
