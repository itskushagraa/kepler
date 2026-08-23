{
  description = "Reproducible development environments for the Kepler chess engine";

  inputs.nixpkgs.url =
    "github:NixOS/nixpkgs/d1bc25d401f5569983ad773f2525045af734de1b";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      # This project-local CUDA package set matches the NixOS-CUDA Hydra job
      # and includes sm_86. TORCH_CUDA_ARCH_LIST below narrows project builds.
      cudaPkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };

      commonPackages = with pkgs; [
        cmake
        gcc
        git
        htop
        ninja
        pkg-config
        stockfish
        tmux
      ];

      pythonPackages = python: with python; [
        chess
        numpy
        pyyaml
      ];

      defaultPython = pkgs.python313.withPackages pythonPackages;
      cudaPython = cudaPkgs.python313.withPackages (python:
        pythonPackages python ++ [ python.torch ]);

      cudaToolkit = cudaPkgs.cudaPackages.cudatoolkit;
      driverLibraryPath = "/run/opengl-driver/lib";
    in
    {
      devShells.${system} = {
        default = pkgs.mkShell {
          packages = commonPackages ++ [ defaultPython ];

          shellHook = ''
            echo "Kepler development shell (CPU)"
          '';
        };

        cuda = pkgs.mkShell {
          packages = commonPackages ++ [
            cudaPython
            cudaToolkit
            pkgs.nvitop
          ];

          CUDA_PATH = cudaToolkit;
          CUDA_HOME = cudaToolkit;
          TORCH_CUDA_ARCH_LIST = "8.6";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc
            cudaToolkit
          ] + ":${driverLibraryPath}";

          shellHook = ''
            echo "Kepler development shell (CUDA, sm_86)"
            echo "NVIDIA driver libraries: ${driverLibraryPath}"
          '';
        };
      };
    };
}
