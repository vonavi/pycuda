{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
  flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      pycuda = pkgs.callPackage ./pycuda.nix {
        inherit (pkgs) cudaPackages;
      };
    in {
      packages.default = pycuda;

      devShells.default = pkgs.mkShell {
        inputsFrom = [ pycuda ];
        buildInputs = [ pkgs.clang-tools ]; # N.B. clang-tools for clangd
      };
  });
}
