import os
import subprocess
import fnmatch
from codecarbon import EmissionsTracker

# Get the current directory
current_dir = os.getcwd()

tracker = EmissionsTracker()
tracker.start()
# Loop through all subdirectories
for root, dirs, files in os.walk(current_dir):
    # Find files that match either 'main_*.py' or 'test_*.py' pattern
    for filename in files:
        if fnmatch.fnmatch(filename, 'main_*.py') or fnmatch.fnmatch(filename, 'test_*.py'):
            file_path = os.path.join(root, filename)
            print(f"Running {file_path} in directory {root}")
            
            # Run each selected file in its respective directory
            subprocess.run(['python3', filename], cwd=root, check=True)

emissions : float = tracker.stop()
print(f"Total emissions for this run: {round(emissions * 1_000_000, 5)} mg CO2")
