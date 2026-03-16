{
  description = "Approval autopilot for Claude Code — auto-approve safe commands, prompt for destructive ones";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      forAllSystems =
        f:
        nixpkgs.lib.genAttrs
          [
            "x86_64-linux"
            "aarch64-linux"
            "x86_64-darwin"
            "aarch64-darwin"
          ]
          (
            system:
            f {
              pkgs = nixpkgs.legacyPackages.${system};
            }
          );
    in
    {
      packages = forAllSystems (
        { pkgs }:
        let
          dippy = pkgs.python3Packages.buildPythonApplication {
            pname = "dippy";
            version = "0.2.6";
            pyproject = true;

            src = ./.;

            postPatch = ''
              substituteInPlace pyproject.toml \
                --replace-fail 'requires = ["uv_build>=0.7.19"]' 'requires = ["setuptools"]' \
                --replace-fail 'build-backend = "uv_build"' 'build-backend = "setuptools.build_meta"'
            '';

            build-system = [ pkgs.python3Packages.setuptools ];

            pythonImportsCheck = [ "dippy" ];

            meta = {
              description = "Approval autopilot for Claude Code";
              homepage = "https://github.com/ldayton/Dippy";
              license = pkgs.lib.licenses.mit;
              mainProgram = "dippy";
            };
          };
        in
        {
          default = dippy;
          inherit dippy;
        }
      );
    };
}
