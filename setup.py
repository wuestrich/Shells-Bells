#! /usr/bin/env python3.8

import sys
import subprocess
import asyncio

def check_program_exists(program):
    res = subprocess.run(
        ["dpkg","-s", program], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return res.returncode == 0

def check_dependencies():
    print("Checking dependencies...")
    not_existing = []
    DEPENDENCIES = [
        "liblzma-dev",
        "portaudio19-dev",
        "python3-pyaudio",
        "python3-tk",
        "sox",
        "python3-virtualenv",
        "mysql-server",
        "libpq-dev",
        "ffmpeg"
    ]
    for program in DEPENDENCIES:
        if not check_program_exists(program):
            not_existing.append(program)
    
    if not_existing:
        print("Missing dependencies: {}".format(", ".join(not_existing)))
        return False
    print("All dependencies installed")
    return True

def check_python_versions():
    print("Checking installed python versions...")
    missing_ver = []
    PYTHON_VERSIONS = [
        "3.8",
        "3.7",
    ]
    res = subprocess.run(
        ["whereis", "python"], capture_output=True, text=True
    )
    for version in PYTHON_VERSIONS:
        if "python{}".format(version) not in res.stdout:
            missing_ver.append("python{}".format(version))

    if missing_ver:
        print("Missing python installations: {}".format(", ".join(missing_ver)))
        return False
    print("All required python versions installed")
    return True

def main(args):
    assert check_dependencies()
    assert check_python_versions()
    pass


if __name__ == "__main__":
    main(sys.argv)