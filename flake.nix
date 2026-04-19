{
  description = "LocalVQE — CPU inference build environment (cmake + gcc + libsndfile)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.cmake
          pkgs.gcc
          pkgs.pkg-config
          pkgs.libsndfile
          # Optional: used when building with -DLOCALVQE_VULKAN=ON
          pkgs.vulkan-loader
          pkgs.vulkan-headers
          pkgs.shaderc
        ];
      };
    };
}
