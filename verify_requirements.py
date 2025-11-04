import subprocess
import sys
import pkg_resources

# Define required packages and versions
REQUIRED = {
    "fastapi": "0.104.1",
    "uvicorn": "0.23.2",
    "opencv-python": "4.8.1.78",
    "numpy": "1.24.4",
    "sqlmodel": "0.0.8",
    "pydantic": "1.10.8",
    "scipy": "1.10.1"
}

def install(package, version):
    """Install or upgrade a specific package version."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])

def main():
    print("🔍 Checking Python dependencies...\n")
    for package, required_version in REQUIRED.items():
        try:
            installed_version = pkg_resources.get_distribution(package).version
            if installed_version != required_version:
                print(f"⚠️ {package} {installed_version} found, upgrading to {required_version}...")
                install(package, required_version)
            else:
                print(f"✅ {package} {installed_version} is up to date.")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package} not found. Installing {required_version}...")
            install(package, required_version)
    print("\n🎉 All requirements verified successfully!")

if __name__ == "__main__":
    main()
