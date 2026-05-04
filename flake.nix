{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaCapabilities = [ "6.0" "6.1" "7.0" "7.5" "8.0" "8.6" "8.9" "9.0" "10.0" "12.0" ];
            cudaForwardCompat = true;
            allowUnsupportedSystem = true;
            allowBroken = true;
          };
          overlays = [
            (final: prev: { cudaPackages = prev.cudaPackages_12_8; })
          ];
        };
        plot = pkgs.callPackage ./nix/plot.nix { };

        jax = (pkgs.python312.pkgs.jax.override { cudaSupport = true; }).overridePythonAttrs (_: { doCheck = false; });
        python312 = pkgs.python312.override {
          packageOverrides = final: prev: { jax = jax; };
        };

        storm = pkgs.callPackage ./nix/storm.nix { };
        stormpy = pkgs.callPackage ./nix/stormpy.nix { storm = storm; };
        tessa = pkgs.callPackage ./nix/tessa.nix {
          python312 = python312;
          stormpy = stormpy;
        };

        dice = pkgs.callPackage ./nix/dice.nix { };
        gennifer = pkgs.callPackage ./nix/gennifer.nix { };
        rubicon = pkgs.callPackage ./nix/rubicon.nix {
          python3Packages = python312.pkgs;
          stormpy = stormpy;
        };

        common-shell-inputs = (with pkgs; [
          which
          bash
          fish
          zsh
          nano
          vim
          tree
          htop
        ]) ++ [
          plot
        ];

        storm-shell-inputs = common-shell-inputs ++ [ storm ];

        geni-shell-inputs = storm-shell-inputs ++ [ gennifer ];

        rubicon-shell-inputs = geni-shell-inputs ++ [
          rubicon
          dice
        ];

        tessa-shell-inputs = rubicon-shell-inputs
          ++ [ python312 stormpy tessa ]
          ++ (with python312.pkgs; [
            jax
            jaxlib
            numpy
            matplotlib
            optax
          ]);

        storm-shell = pkgs.mkShell {
          shellHook = ''
              echo "storm-shell: storm CLI is on PATH. Bring your own Tessa via pip and feed it .jani models."
          '';
          buildInputs = storm-shell-inputs;
        };

        geni-shell = pkgs.mkShell {
          shellHook = ''
              echo "geni-shell: storm CLI + gennifer are on PATH."
          '';
          buildInputs = geni-shell-inputs;
        };

        rubicon-shell = pkgs.mkShell {
          shellHook = ''
              echo "rubicon-shell: storm CLI + gennifer + rubicon + dice are on PATH."
          '';
          buildInputs = rubicon-shell-inputs;
        };

        tessa-shell = pkgs.mkShell {
          shellHook = ''
              echo "NIX_LD=$NIX_LD"
              echo "NIX_LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH"
              echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
              # Avoid nix-ld overriding Nix's libstdc++ inside the dev shell.
              unset NIX_LD
              unset NIX_LD_LIBRARY_PATH
              unset LD_LIBRARY_PATH
              # Putting $PWD first on PYTHONPATH also makes sitecustomize.py
              # (in the repo root) auto-run at interpreter startup, which
              # ctypes-preloads libcuda.so.1 so jaxlib finds it.
              export PYTHONPATH="$PWD''${PYTHONPATH:+:}$PYTHONPATH"
          '';
          buildInputs = tessa-shell-inputs;
        };

      in
      {
        packages = {
          tessa = tessa;
          storm = storm;
          stormpy = stormpy;
          rubicon = rubicon;
          gennifer = gennifer;
          dice = dice;
        };

        devShells = {
          inherit tessa-shell rubicon-shell geni-shell storm-shell;
          default = tessa-shell;
        };
      }
    );
}
