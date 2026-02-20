{
  stdenv,

  # Native build inputs
  autoAddDriverRunpath,
  cmake,
  cudaPackages
}:

stdenv.mkDerivation {
  pname = "pycuda";
  version = "0.0.1";

  src = ./.;

  nativeBuildInputs = [
    autoAddDriverRunpath
    cmake
    (with cudaPackages; [
      cuda_cudart
      cuda_nvcc
    ])
  ];
}
