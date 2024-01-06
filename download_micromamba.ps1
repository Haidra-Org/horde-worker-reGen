# Download micromamba if it doesn't exist, suppressing any errors
if (!(Test-Path "micromamba.exe")) {
    Write-Output "* Downloading micromamba"
    Invoke-Webrequest -URI https://micro.mamba.pm/api/micromamba/win-64/latest -OutFile micromamba.tar.bz2
    tar xf micromamba.tar.bz2

    Move-Item -Force Library\bin\micromamba.exe micromamba.exe

    Remove-Item micromamba.tar.bz2
    Remove-Item -r ./Library/
}
