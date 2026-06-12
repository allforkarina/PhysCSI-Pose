# Repository Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Initialize the repository with code-only collaboration constraints and connect it to the approved GitHub remote.

**Architecture:** Keep the bootstrap minimal: root-level agent instructions, root-level ignore rules, and lightweight design/plan documentation. No model package is scaffolded in this step because the architecture for the WiFi CSI pose model has not yet been specified.

**Tech Stack:** Git, Markdown, Python-oriented ignore rules.

---

### Task 1: Repository Policy Files

**Files:**
- Create: `AGENTS.md`
- Create: `.gitignore`
- Create: `docs/superpowers/specs/2026-06-12-repository-bootstrap-design.md`
- Create: `docs/superpowers/plans/2026-06-12-repository-bootstrap.md`

- [x] **Step 1: Create the policy files**

  Add `AGENTS.md` with the remote, data, push, and allowed-content constraints. Add `.gitignore` entries for Python cache files, virtual environments, data directories, experiment outputs, checkpoints, logs, generated reports, and model/data artifact extensions. Add this design and plan documentation under `docs/superpowers/`.

- [ ] **Step 2: Inspect the created files**

  Run: `rg --files`

  Expected: only `.gitignore`, `AGENTS.md`, and the two `docs/superpowers/...` markdown files are listed.

### Task 2: Git Initialization And Remote

**Files:**
- No source files modified.

- [ ] **Step 1: Initialize Git**

  Run: `git init`

  Expected: Git creates `.git/` in the current workspace.

- [ ] **Step 2: Configure the main branch**

  Run: `git branch -M main`

  Expected: the current branch is `main`.

- [ ] **Step 3: Add the approved remote**

  Run: `git remote add origin git@github.com:allforkarina/PhysCSI-Pose.git`

  Expected: `git remote -v` shows `origin` using `git@github.com:allforkarina/PhysCSI-Pose.git` for fetch and push.

### Task 3: Commit And Push

**Files:**
- Stage: `.gitignore`
- Stage: `AGENTS.md`
- Stage: `docs/superpowers/specs/2026-06-12-repository-bootstrap-design.md`
- Stage: `docs/superpowers/plans/2026-06-12-repository-bootstrap.md`

- [ ] **Step 1: Stage only allowed files**

  Run: `git add .gitignore AGENTS.md docs/superpowers/specs/2026-06-12-repository-bootstrap-design.md docs/superpowers/plans/2026-06-12-repository-bootstrap.md`

  Expected: `git status --short` shows only the four intended files staged.

- [ ] **Step 2: Commit**

  Run: `git commit -m "chore: initialize repository policy"`

  Expected: Git creates the initial commit.

- [ ] **Step 3: Push**

  Run: `git push -u origin main`

  Expected: the `main` branch is pushed to GitHub and tracks `origin/main`.

### Task 4: Final Verification

**Files:**
- No source files modified.

- [ ] **Step 1: Verify cleanliness and remote**

  Run: `git status --short`

  Expected: no output.

  Run: `git remote -v`

  Expected: both fetch and push URLs are `git@github.com:allforkarina/PhysCSI-Pose.git`.
