"""Run the full smoke test suite."""
import subprocess, sys, os

print("Installing dependencies...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "ccxt", "httpx", "pandas", "numpy"],
    capture_output=True, text=True, cwd="/home/zer0null/algo-trading-bot",
)
if result.returncode != 0:
    print(f"pip install stderr: {result.stderr}")

print("\nRunning smoke tests...\n")
result = subprocess.run(
    [sys.executable, "tests/test_smoke.py"],
    capture_output=True, text=True, cwd="/home/zer0null/algo-trading-bot",
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
sys.exit(result.returncode)
