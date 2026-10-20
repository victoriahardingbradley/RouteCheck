# Deploy RouteCheck with Streamlit and Neon

Use the **RouteCheck-cloud** folder as the root of a new GitHub repository. It contains app.py, all Python modules, requirements.txt, trained weights, the original study evidence, and a compact excerpt of four real held-out routes. The Desktop application still reads your full local data and uses SQLite unless you configure PostgreSQL. No raw multi-gigabyte files are needed for hosting.

## 1. Create Neon

1. Sign in at https://console.neon.tech and create a project named RouteCheck. Use the default PostgreSQL database and choose a region near your app.
2. Select **Connect** and copy the PostgreSQL connection string, preferably the pooled connection. Keep its `sslmode=require` parameter and any other Neon parameters.
3. Treat this URL as a password. Do not upload it to GitHub or paste it in app inputs.

The app automatically creates `routecheck_runs`, `routecheck_reviews`, and `routecheck_artifacts` in the selected database. Use a dedicated database/role for RouteCheck. It needs permission to create and write these tables. No SQL setup is required.

## 2. Upload the deployment folder

Create a GitHub repository and upload the **contents** of RouteCheck-cloud so app.py is at the repository root. Include the hidden .streamlit folder, bundled_data, models and experiments. Do not upload .venv, secrets.toml, SQLite databases, Desktop raw data, or your actual connection URL. The example secrets file is safe to include.

## 3. Deploy

In https://share.streamlit.io choose **Create app**, select the repository/branch, and set the main file to `app.py`. In Advanced settings select Python **3.12**, and paste this into Secrets, replacing the example with the entire real Neon URL:

```toml
ROUTECHECK_DATABASE_URL = "postgresql://ROLE:PASSWORD@YOUR-NEON-HOST/neondb?sslmode=require"
```

Deploy, then load a bundled route and click Run comparison. In My comparisons, download its trial JSON files. Add a review, reboot the app through its management controls, and confirm the comparison, downloads and review remain available.

Without a connection string, the cloud package shows the study and route data but disables new comparisons. Connection failures show a sanitized message; the app never silently switches cloud runs to ephemeral SQLite.

## What is saved

Each button click saves a paired comparison, both individual trial JSON files, both readable summaries, the full modification log and all model/route hashes. PostgreSQL commits the run and evidence in one transaction. Reviews append an audit record and update the displayed decision in a transaction. Downloads retrieve cloud evidence directly from PostgreSQL, even when temporary local files disappear. Streamlit rerenders do not launch new comparisons.

The historical 576-trial study stays in the repository; new interactive trials go to Neon. Existing Desktop SQLite runs remain local and are not migrated automatically. Keep that database and original evidence files as your local archive.

This is a shared team workspace: names are self-reported and all viewers can see and review shared runs. Use Streamlit's private-app sharing controls when available for your account, and invite only teammates. There is no user-account system in this prototype. Repository dataset excerpts and offline trial files are visible to anyone with repository access. Database hosting persists data across app restarts but is not a substitute for an independent export/backup.

## Run the cloud package locally

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For read-only preview, leave Secrets unset. To test cloud writes, create `.streamlit/secrets.toml` from the example and add your real URL; this file is ignored by Git. Locally, you can alternatively set `ROUTECHECK_DATABASE_URL` as an environment variable. The ordinary RouteCheck folder works with SQLite without Neon.

## Data and limits

The four bundled routes are complete normalized test routes from your supplied Amazon Last Mile Routing Research Challenge dataset, not invented demo routes. cloud_bundle.json lists route IDs and SHA-256 checksums; dataset_hash values match selected_routes.csv. Historical driver sequences are used for training/evaluation, not planner inputs. Defects are controlled simulations. Estimates are historical, not live traffic, verified arrivals or guarantees of optimal routing.

Source: Amazon Last Mile Routing Research Challenge (2021), https://registry.opendata.aws/amazon-last-mile-challenges/. Dataset license: Creative Commons Attribution-NonCommercial 4.0, https://creativecommons.org/licenses/by-nc/4.0/. This package is for the noncommercial college research project; retain attribution with copies. Trained weights are the project's small NumPy neural constructive policy; no cloud training is required.

Official setup references: https://neon.com/docs/get-started/connect-neon and https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management.

## Verification

See CLOUD_VERIFICATION.md for tests run. Real Neon credentials and a hosted Streamlit deployment must still be configured by the account owner; testing used a disposable local PostgreSQL 16 server, not a live Neon account.

## Teammate model integration

Upload the teammate project's actual metrics.csv or pilot-output ZIP under **Teammate model: pilot results**. For individual stop orders, use **Load results from your teammate's model** after loading the matching route. See MODEL_INTEGRATION.md for the prepared exporter and live callable option. Source-code ZIPs and mock API fixtures are not measured results. The supplied teammate model is CatBoost repair; the original 576-trial neural study remains separate.
