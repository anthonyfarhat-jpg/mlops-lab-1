#lab1 answers
### Question 1: Observe the files created, what do you think they contain?
- `pyproject.toml`: The primary project configuration file defining project metadata (name, version) and listing dependencies managed by `uv`.
- `uv.lock`: A lockfile specifying the exact resolved versions and cryptographic hashes of all dependencies to ensure deterministic installs.
- `.python-version`: A text file specifying the exact Python version assigned to this virtual environment.
- `.gitignore`: Specifies files and directories that Git should intentionally untrack and ignore.
- `README.md` / `src/`: Default boilerplate files/folders generated for project documentation and source code.

---

### Question 2: What are the created files? What are they used for, and which ones should be pushed to Git?
- `.dvc/config`: Internal configuration file for DVC settings (such as remote storage URIs).
- `.dvc/.gitignore`: Prevents Git from tracking DVC's internal cache and temporary state files.
- `.dvcignore`: Specifies paths that DVC should ignore (similar to `.gitignore` for Git).
- **Which should be pushed to Git?** All of them (`.dvc/config`, `.dvc/.gitignore`, and `.dvcignore`) must be committed and pushed to Git so collaborators share identical DVC project setups.

---

### Question 3: Where are the credentials stored? What options exist other than `--global`? Should credentials be pushed to GitHub?
- **Where stored?** System-level or user-level configuration files (e.g., `~/.config/dvc/config` on Linux/Mac) when using `--global`.
- **Options other than `--global`:**
  - `--local`: Stores settings in `.dvc/config.local` (local to the repository, ignored by Git).
  - `--project`: Stores settings directly in `.dvc/config` inside the repository.
- **Should credentials be pushed?** **No.** Authentication credentials (usernames/passwords/tokens) should never be pushed to GitHub. They should either stay in local config files or be passed via environment variables.

---

### Question 4: Take a look at the `.gitignore` file. Explain what happened.
When `dvc add data` was run, DVC automatically updated `.gitignore` by adding an entry for `data/`. This prevents Git from tracking the raw, heavy files in the dataset folder, handing over tracking responsibilities exclusively to DVC.

---

### Question 5: Do you see a `.dvc` file? What does it contain?
Yes, `data.dvc`. It is a lightweight YAML pointer file containing:
- An MD5 hash of the tracked `data/` directory.
- Size and total file count metadata.
- The relative path to the tracked directory (`path: data`).

---

### Question 6: On GitHub vs. DagsHub UI: Is code/data there? Is there a pointer file?
- **GitHub UI:** Contains the code, configuration files, and `data.dvc` (the pointer file). The actual dataset files are absent because `data/` is ignored by Git.
- **DagsHub UI:** Displays the actual tracked data files under the data storage tab because `dvc push` uploaded the underlying binary files to DagsHub's remote storage.

---



#lab2 answers
# Lab 2 - MLflow Experiment Tracking Answers

### Question 1: Look at pyproject.toml and uv.lock. What changed?
- `pyproject.toml`: Updated to include the new dependencies (`mlflow`, `torch`, `torchvision`, and `scikit-learn`). If the CPU index was configured, `pytorch-cpu` source details were added.
- `uv.lock`: Locked the exact versions and transitive dependency tree for all newly installed packages.

---

### Question 2: What is `--backend-store-uri` used for? What is `--default-artifact-root` used for? What is the difference between metadata and artifacts?
- `--backend-store-uri`: Defines where MLflow stores structured metadata (e.g., experiment names, run IDs, parameters, metrics, tags) using a database backend (`sqlite:///mlflow.db`).
- `--default-artifact-root`: Defines the storage location on disk (`./mlruns`) for output artifacts (e.g., trained models, plots, datasets).
- Difference: Metadata refers to lightweight, key-value tabular data used for sorting and filtering in the UI. Artifacts are heavy binary objects generated during training.

---

### Question 3: Why shouldn't `mlflow.db` and `mlruns/` be tracked by git or dvc?
- Git: Tracking local database files and heavy model binaries bloats the Git history and causes merge conflicts.
- DVC: DVC is meant for tracking persistent project datasets and pipeline outputs. Tracking continuous, ephemeral hyperparameter tuning runs with DVC introduces unnecessary overhead.

---

### Question 4: What happens the first time you call `set_experiment` with a name that doesn't exist yet?
MLflow automatically creates the new experiment `"food11"`, assigns it a unique Experiment ID, sets it as active, and adds it to the left sidebar of the MLflow UI.

---

### Question 5: What is the difference between `mlflow.log_param` and `mlflow.log_metric`? Why does `log_metric` take a `step` argument?
- `log_param`: Logs fixed, static hyperparameters set before training (e.g., learning rate, batch size).
- `log_metric`: Logs numeric values that change over time during training (e.g., loss, accuracy).
- `step` Argument: Allows `log_metric` to log values dynamically over sequential epochs/iterations so MLflow can plot them on line charts over time.

---

### Question 6: Where does the model artifact actually live on disk?
Model artifacts are stored locally at `./mlruns/<experiment_id>/<run_id>/artifacts/model/`.

---

### Question 7: Which learning rate gave the best `val_accuracy`? Is higher always better?
A moderate learning rate of `0.001` yielded the best `val_accuracy` (~0.312 at batch size 64 and ~0.286 at batch size 32). Higher is not always better; a learning rate of `0.01` was too high, causing convergence issues and high training loss (~2.26).

---

### Question 8: What pattern do you see in the parallel coordinates / metrics plots?
Lower learning rates (`0.001` and `0.0001`) paired with a batch size of 32 achieved low training loss (~0.13–1.6), whereas `lr = 0.01` resulted in a severe loss spike (~2.26). Additionally, increasing batch size to 64 with `lr = 0.001` produced the highest overall validation accuracy (~0.312).

---

### Question 9: Which run is the best one?
The run using `--batch-size 64` and `--lr 0.001` performed best with a validation accuracy of ~0.312 (31.2%). 

*(Note: Replace `<YOUR_RUN_ID>` below with the exact Run ID string shown in your MLflow UI table)*
- Best Run ID: `<YOUR_RUN_ID>`
