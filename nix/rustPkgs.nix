{ nixpkgsMoz, pkgs }:
let
  mozPkgs = import "${nixpkgsMoz}/package-set.nix" {
    inherit pkgs;
  };

  channel = mozPkgs.rustChannelOf {
    rustToolchain = ../rust-toolchain;
    sha256 = "sha256-g0vzQzexOfgXeu7nm8AFJJxmRCDpZXksw3H58a156+0=";
  };
in
channel // {
  rust = channel.rust.override { extensions = [ "rust-src" ]; };
}
