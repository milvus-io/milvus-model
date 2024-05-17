import subprocess

def prompt_install(package: str, warn: bool = False):  # pragma: no cover
    cmd = f"pip install -q {package}"
    try:
        if warn and input(f"Install {package}? Y/n: ") != "Y":
            raise ModuleNotFoundError(f"No module named {package}")
        print(f"start to install package: {package}")
        subprocess.check_call(cmd, shell=True)
        print(f"successfully installed package: {package}")
    except subprocess.CalledProcessError as e:
        raise ValueError(f"install error {e}")
