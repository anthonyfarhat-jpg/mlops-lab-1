# Lab 1 summary

## Solution used
I kept the existing DVC remote configuration in the repo and did not switch to a local folder remote. The repo already had a DagsHub remote configured under `.dvc/config`, so the default DVC remote stayed as `origin`.

## Git and DVC answers

1. `uv init` creates a Python project skeleton including `pyproject.toml`, a `.python-version` file, and a package folder layout. The project file defines the package metadata and dependencies, while the Python version file pins the interpreter for the workspace.
2. `dvc init` creates the `.dvc/` directory and `.dvcignore`. The folder stores DVC internals such as the cache and config, while `.dvcignore` tells DVC what files to ignore. Only the config and metadata that are meant to be version-controlled should be pushed to Git; the actual data cache stays out of Git.
3. The credentials for DVC are stored in the local or global DVC config, not in the repository itself. `--global` writes them to the user-level config, while `--local` writes to the repo-local config. The credentials should not be pushed to GitHub.
4. After `dvc add data`, the `.gitignore` file is updated so the tracked data directory is ignored by Git. The pointer file `data.dvc` stores the hash metadata for the data directory and is the part that Git tracks.
5. The DVC metadata file `data.dvc` contains the hashed reference to the data directory and its path, file count, and size. It is the pointer Git stores, while the actual binary data remains in the DVC cache.
6. GitHub shows the code and the pointer file; the actual raw data is not stored directly there. DagsHub shows the data objects after `dvc push` because the remote is configured to receive the dataset contents.
7. A fresh clone of the GitHub repo does not contain the large dataset files unless they are checked out with DVC. The command is `dvc checkout` (or `dvc pull` if the remote is being fetched for the first time).
8. After switching to an earlier commit and running `dvc checkout`, the old directory state is restored; after switching back to `main`, `dvc checkout` restores the newer processed dataset state again.

## Data preparation result
The project now includes a preprocessing script at `src/food11/data.py` that reads the raw Food-11 images, resizes them to 128x128, and creates:
- `data/food11_processed`
- `data/food11_processed_mini`

The processed data is organized by split and class folder name, matching the expected ResNet-style structure.
