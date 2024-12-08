import os
import glob

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the path for the "../figures" folder
figures_directory = os.path.join(current_directory, '..', 'figures')

# Check if the figures folder exists, if not, create it
if not os.path.exists(figures_directory):
    os.makedirs(figures_directory)
    print(f"Created folder: {figures_directory}")

# Find all Python files that start with 'evaluation' in the current directory
evaluation_files = glob.glob(os.path.join(current_directory, 'evaluation*.py'))

# Run each file
for file in evaluation_files:
    print(f"Running {file}...")
    with open(file) as f:
        code = f.read()
        exec(code)
