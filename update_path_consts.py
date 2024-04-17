import fileinput
import sys
import os

# Assuming the script is executed in or below the 'image-worker' directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up the directories to the 'image-worker', if not already there
while os.path.basename(base_dir) != 'image-worker' and os.path.basename(base_dir) != '':
    base_dir = os.path.dirname(base_dir)

# Construct the path to the file we need to edit
file_path = os.path.join(base_dir, 'conda', 'envs', 'windows', 'lib', 'site-packages', 'horde_model_reference', 'path_consts.py')

owner_pattern = 'GITHUB_REPO_OWNER = "Haidra-Org"'
new_owner = 'GITHUB_REPO_OWNER = "AIPowergrid"'
repo_pattern = 'GITHUB_REPO_NAME = "AI-Horde-image-model-reference"'
new_repo = 'GITHUB_REPO_NAME = "grid-image-model-reference"'

with fileinput.FileInput(file_path, inplace=True, backup='.bak') as file:
    for line in file:
        line = line.replace(owner_pattern, new_owner)
        line = line.replace(repo_pattern, new_repo)
        sys.stdout.write(line)