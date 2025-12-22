# RAG - The Ponni Archive
The respository will contain all the research that will be carried out for building production RAG for the Ponni magazine project.


## 🛡️ Quality & Security (Pre-commit)

This project uses `pre-commit` hooks to automatically check for:
*   **Security**: Scans for accidental committed secrets (AWS keys, API tokens) using `gitleaks`.
*   **Data Safety**: Prevents large file commits (>10MB) to keep repo clean.
*   **Code Style**: Enforces `black`, `flake8`, and `isort` for Python consistency.

### Installation of pre-commit-hooks
1.  **Install the package**: `pip install pre-commit`
2.  **Install the hooks**: `pre-commit install`

Now, checks run automatically on every `git commit`.

### Manual Run
Run against all files at any time:
```bash
pre-commit run --all-files
```
