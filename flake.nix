{
  description = "ad";

  inputs = {
    nixpkgs = {
      url = "github:nixos/nixpkgs?ref=release-22.05";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
    };
  };

  outputs = { self, fenix, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
          };
        in
        rec {
          devShell = pkgs.mkShell
            rec {
              buildInputs = with pkgs; [
                blas
              ];
              #LD_LIBRARY_PATH = "${pkgs.openblas}:$LD_LIBRARY_PATH"; 
              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs; 
            };
        }
      );
}
