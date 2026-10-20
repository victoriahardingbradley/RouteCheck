# Upload RouteCheck to GitHub

Upload the contents of this folder to the root of a website repository (keep the directory structure). Set Streamlit's main file to app.py and Python to 3.12. All files fit GitHub's browser upload limits. Original historical trial files are preserved inside experiments/study_evidence.zip and the website reads individual JSON/Markdown records from it.

Add your Neon URL only to Streamlit's Advanced settings > Secrets:

    ROUTECHECK_DATABASE_URL = "your entire pooled Neon connection string"

No real credentials are included. Follow DEPLOYMENT.md. This is the runnable website; do not substitute the teammate's model-only repository. The full source, tests and uncompressed original evidence are retained in the separate RouteCheck-cloud package.
