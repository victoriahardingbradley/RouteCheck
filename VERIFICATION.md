# Verification record

Verified October 20, 2026 on macOS arm64 with Python 3.12.7, Streamlit 1.50.0, NumPy 2.5.4, pandas 2.3.3 and Plotly 6.3.1.

- Final installed application: **50 automated tests passed**, using the virtual environment inside `Desktop/caseathon/website/routecheck`.
- The one-screen interface has no sidebar or page navigation. Tests verify that changing settings does not run an experiment, explicit runs save records, repeat rendering does not create duplicate runs, and the review form validates input.
- Browser verification: read the supplied local JSON folder, found 6,112 routes, loaded 118 stops and 276 packages for one real route, ran reference and combined comparisons, inspected before/current values, explanatory text, study curves, individual trial downloads and the review form.
- Real-file importer checks also loaded three separate complete routes. Full travel-time files are not loaded into memory.
- Trained neural policy and neural travel-time estimator on 15 complete training routes, using 5 distinct later validation routes. Evaluated 4 still-later full test routes.
- Completed 576 held-out runs: 4 conditions × 6 simulated stages × 2 countermeasure modes × 3 seeds × 4 test routes.
- All 576 trials have individual JSON records and readable Markdown summaries, plus shared exact-value modification logs and preserved per-study snapshots of the model/results. Tests verify every trial file and chronological split boundaries.
- Learning/adapter checks: neural loss improvement, weight restoration, frozen inference, no driver-order leakage into live payloads, repeatable corruption, matching-pair evaluation arithmetic and invalid-output rejection.
- Oversight checks: persistent audit history, complete stop-order override validation and separate optional self-reported human review duration. No real-person review or usability study is claimed.
- Local server launch and HTTP health endpoint returned success. The Desktop project has its own installed dependencies and double-click launcher.
- Raw JSON file sizes and modification times remain identical to the study manifest. Existing top-level website files and the original case PDF were preserved.
- Source brief inspected: the actual Desktop `Case-a-thon Prompt.pdf`, especially the requirements on pages 3, 4, 7 and 8. No deck was prepared, at the user's explicit request.

Scope limits: the study uses only four independent test routes from a local split of the supplied training corpus. The separate official evaluation corpus was not supplied. Degradation is simulated; historical travel-time estimates are not fresh ground truth. The model is a small supervised neural constructive policy, not a pointer-network or reinforcement-learning reproduction. Windows/Linux commands were not tested. No vehicle, dispatch service or external AI inference service is connected.

The supplied fallback is a 36-second silent GIF assembled from actual browser captures of the app and evidence viewer, with recording notes. It is not a slide deck.
