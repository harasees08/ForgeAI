# ForgeAI release progress

The release patch targets commit `161841e`. It fixes stale reports in Engineering Workspace and extends the Streamlit regression. The release files add the runnable flagship walkthrough, benchmark methodology, product one-pager, demo script, application bullets and validation record. See [validation](validation.md) for measured results and remaining limitations.

## Resume from the repository root

```powershell
git rev-parse --short HEAD
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m forgeai evaluate-retrieval --device cuda:0
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 --server.headless true
```

Run the full Streamlit AppTest in a regular PowerShell window on the laptop. In Engineering Workspace, run the calculation, select **Document question**, and confirm the calculation and its download disappear. Return to **Engineering calculation** with the original prompt to see the saved result, then edit the prompt to hide it until the next run.

GitHub publication remains pending. Review the files and repository visibility before pushing. The original supplier PDFs, local environment, model weights and private experiment runs stay outside the release package.
