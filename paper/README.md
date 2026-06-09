# Soliloquy Paper Workspace

This directory contains the LaTeX paper source, figure copies, and Overleaf CLI
configuration for the Soliloquy project.

The canonical paper entry point is `main.tex`. It is self-contained and loads
figures from `figures/`.

## Setup

```bash
cd paper
npm install
```

Authenticate with Overleaf using an `overleaf_session2` cookie:

```bash
OVERLEAF_SESSION="..." make auth
make check
```

After authentication, `olcli` writes `.olcli.json` locally. That file is ignored
and should not be committed.

## Common Commands

```bash
make figures          # refresh figures/ from ../results/plots/paper_figures
make list             # list Overleaf projects
make pull             # pull the linked Overleaf project into this directory
make push             # push local paper files to the linked Overleaf project
make sync             # bidirectional sync, including local deletions
make sync-no-delete   # bidirectional sync without propagating local deletions
make compile          # trigger an Overleaf compile
make pdf              # compile on Overleaf and download soliloquy.pdf
make bbl              # download main.bbl for arXiv-style submissions
```

## Layout

```text
paper/
├── main.tex
├── references.bib
├── figures/
├── scripts/
├── package.json
├── package-lock.json
├── .gitignore
├── .olignore
└── Makefile
```

`figures/` contains the tracked paper figure copies. Refresh them with
`make figures` after regenerating `../results/plots/paper_figures/*.png`.
