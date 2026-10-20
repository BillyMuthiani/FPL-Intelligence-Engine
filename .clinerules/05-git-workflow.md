# Git and Change Management

## Protect Existing Work

* Inspect `git status --short --branch` before making changes.
* Review relevant diffs before editing.
* Preserve all user-created, untracked, modified, and staged work.
* Never assume untracked files are disposable.
* Do not overwrite or discard changes simply because they appear unrelated.

## Scope Control

* Make only changes required for the approved task.
* Do not perform unrelated refactors or mass formatting.
* Avoid modifying lockfiles, dependencies, generated data, or project metadata unless necessary.
* Explain unavoidable changes outside the original scope.

## Git Safety

* Do not run `git reset --hard`, `git clean`, destructive checkout commands, or equivalent destructive operations without explicit approval.
* Do not stage, commit, push, rebase, amend, or force-push unless explicitly authorized.
* Never remove secrets from local configuration without preserving the user's required local setup.
* Never commit API keys, passwords, tokens, `.env` files, or other secrets.
* Follow `.gitignore` and inspect whether sensitive files are already tracked.

## Before Finishing

* Inspect `git diff` and `git status`.
* Identify modified, created, deleted, staged, and untracked files accurately.
* Explain the purpose of each meaningful change.
* Report tests and checks actually performed.
* Leave unrelated user work untouched.

## Approval

If a task requires destructive actions, broad restructuring, changes to public interfaces, dependency upgrades, or significant changes to Git history, explain the consequences and request approval first.
