import subprocess
import sys
import importlib
import os
import json
LOG_FILE_PATH = '../assests/logmodules.json'


# Version-locked package map
required_modules = {
    'onnxruntime': '1.22.0',
    'numpy': '2.3.1',
    'transformers': '4.53.0',
    'faiss': '1.11.0',
    'requests': '2.32.4',
    'pillow': '11.2.1',
    'scipy': '1.15.3',
    'sklearn': '1.7.0',
    'tqdm': '4.67.1'
}
def load_log():
    if not os.path.exists(LOG_FILE_PATH):
        return {}
    with open(LOG_FILE_PATH, 'r') as f:
        return json.load(f)
    
def update_log(module):
    log = load_log()
    log[module] = True
    with open(LOG_FILE_PATH, 'w') as f:
        json.dump(log, f)

# pip package names (if different from import names)
module_to_package = {
    'faiss': 'faiss-cpu',
    'sklearn': 'scikit-learn',
    'sentence_transformers': 'sentence-transformers'
}


def install_module(module_name, package_name, version):
    """Install module using pip with version lock."""
    print(f" Installing {module_name}=={version}...")
    sys.stdout.flush()
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package_name}=={version}"])
    print(f" {module_name} installed successfully.")
    sys.stdout.flush()


def ensure_module_installed(module_name):
    """Check if module is installed, if not download/install it."""
    package_name = module_to_package.get(module_name, module_name)
    version = required_modules[module_name]
    try:
        importlib.import_module(module_name)
        print(f" {module_name} is already installed.")
    except ImportError:
        print(f" {module_name} not found.")
        install_module(module_name, package_name, version)
          

# Start Checking All Required Modules
print("Checking required modules...")
for module in required_modules:
    ensure_module_installed(module)
    update_log(module)

# This should be OUTSIDE the loop
print("ALL_DOWNLOADS_COMPLETED")
sys.stdout.flush()