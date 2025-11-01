import importlib
import sys
import os
import subprocess
import json

# Expected versions
required = {
    "tensorflow": "2.15.0",
    "keras": "2.15.0",
    "keras_rl2": "1.0.4",
    "gymnasium": "0.29.1",
    "numpy": "1.25.2",
    "PIL": "10.3.0",
    "h5py": "3.11.0",
}

def check_package(pkg, expected_version):
    try:
        module = importlib.import_module(pkg)
        version = getattr(module, "__version__", "unknown")
        ok = version.startswith(expected_version)
        print(f"[{'OK' if ok else 'WARN'}] {pkg} version {version} (expected {expected_version})")
    except ImportError:
        print(f"[MISSING] {pkg} not installed!")
        return False
    return True

print("🔍 Checking Python environment...\n")

# Show current interpreter
python_path = sys.executable
print(f"🧠 Current Python interpreter: {python_path}")

# Detect Pylance interpreter (if VS Code settings exist)
settings_path = os.path.join(".vscode", "settings.json")
if os.path.exists(settings_path):
    with open(settings_path, "r", encoding="utf-8") as f:
        try:
            settings = json.load(f)
            pylance_path = settings.get("python.defaultInterpreterPath") or settings.get("python.pythonPath")
            if pylance_path:
                same = os.path.normpath(pylance_path) == os.path.normpath(python_path)
                print(f"🧩 VS Code (Pylance) interpreter: {pylance_path}")
                print(f"➡️  {'Same environment ✅' if same else 'DIFFERENT interpreter ⚠️'}")
            else:
                print("⚠️ VS Code settings found, but no python interpreter defined.")
        except Exception:
            print("⚠️ Could not parse .vscode/settings.json")
else:
    print("ℹ️ No .vscode/settings.json found — VS Code may be using default settings.")

# Check Python version
if not (sys.version_info.major == 3 and sys.version_info.minor in (10, 11)):
    print(f"[WARN] Python {sys.version_info.major}.{sys.version_info.minor} detected — recommended: 3.10 or 3.11")

print("\n📦 Checking dependencies...\n")

ok = True
for pkg, ver in required.items():
    if not check_package(pkg, ver):
        ok = False

print("\n🎮 Checking Atari environment availability...\n")
try:
    import gymnasium as gym
    env = gym.make("ALE/Breakout-v5")
    print("[OK] Atari Breakout environment loaded successfully!")
    env.close()
except Exception as e:
    ok = False
    print(f"[ERROR] Could not load ALE/Breakout-v5: {e}")
    print("Try running: AutoROM --accept-license")

if ok:
    print("\n✅ All dependencies and environment look good!")
else:
    print("\n⚠️ Some issues were found. Check the warnings above.")
