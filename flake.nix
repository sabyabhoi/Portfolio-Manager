{
  inputs = { nixpkgs.url = "github:nixos/nixpkgs"; };

  outputs = { self, nixpkgs }:
    let pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in {
      devShell.x86_64-linux =
        pkgs.mkShell { 
          name = "python-devel";
          venvDir = "venv";
          buildInputs = with pkgs.python311Packages; [
            pandas
            numpy
            scipy
            jupyter
            yfinance
            python-dotenv
            statsmodels
            icecream
            venvShellHook
            seaborn
            babel
            pyarrow
          ]; 
        };
   };
}
