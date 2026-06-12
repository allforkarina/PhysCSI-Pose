# Repository Bootstrap Design

## Goal

Initialize the local `PhysCSI_Pose` workspace as the code-only home for a WiFi CSI based human pose recognition project and connect it to the GitHub remote `git@github.com:allforkarina/PhysCSI-Pose.git`.

## Approved Constraints

- The current repository links to `git@github.com:allforkarina/PhysCSI-Pose.git`.
- Local work is limited to code authoring and code-correctness testing while real model inference data is unavailable.
- Every completed modification is committed and pushed to GitHub.
- GitHub content is restricted to code scripts, tests, lightweight configuration, and explanatory documentation. Data, model weights, experiment outputs, generated artifacts, caches, and local environment files are excluded.

## Architecture

The bootstrap layer consists of two repository-root files and two planning documents:

- `AGENTS.md` defines project-level instructions for future agent work.
- `.gitignore` prevents accidental staging of datasets, checkpoints, generated outputs, caches, local environments, and editor metadata.
- This design document records the agreed repository constraints.
- The implementation plan documents the concrete execution steps used for this bootstrap.

## Data And Inference Boundary

Until the user provides usable WiFi CSI data and explicitly requests inference or evaluation, code validation must rely on static checks, unit tests, synthetic tensors, mocks, or deterministic lightweight fixtures. No accuracy, quality, or runtime claims should be made without a real evaluation run.

## Testing

This bootstrap is verified through Git state checks, staged-file inspection, and remote configuration checks. Model-code tests will be introduced when the first source modules are implemented.
