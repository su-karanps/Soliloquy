# Soliloquy Paper Workspace

This directory contains the LaTeX paper source, paper figures, and Overleaf CLI
integration for the Soliloquy project.

It is linked to the existing Overleaf project:

```text
Circuits of Uncertainty
Project ID: 69f58e66fcfdf1cf3536aef1
```

The Overleaf CLI is pinned to the upstream GitHub repo:

```text
https://github.com/aloth/olcli.git#95007ca4707094aae2d2615ccde89964747e3e03
```

## Setup

```bash
cd paper
npm install
```

Authenticate with Overleaf using your session cookie:

```bash
OVERLEAF_SESSION="..." make auth
make check
```

You can get the cookie by logging into Overleaf, opening browser developer tools,
and copying the `overleaf_session2` cookie value.

## Common Commands

```bash
make figures                  # refresh paper/figures from ../results/plots/paper_figures
make list                     # list Overleaf projects
make pull                     # pull Circuits of Uncertainty into this directory
make push                     # push local paper/ files to the linked Overleaf project
make sync                     # bidirectional sync, including local deletions
make sync-no-delete           # bidirectional sync without propagating local deletions
make compile                  # trigger an Overleaf compile
make pdf                      # compile on Overleaf and download soliloquy.pdf
make bbl                      # download main.bbl for arXiv-style submissions
```

After `olcli pull`, this folder should contain `.olcli.json`, which lets `olcli`
auto-detect the linked Overleaf project for future `push`, `sync`, and `pdf`
commands.

## Layout

```text
paper/
├── main.tex
├── sections/
├── figures/
├── scripts/
├── package.json
├── .olignore
└── Makefile
```

`figures/` contains copies of the tracked paper figures from `../results/plots`.
Run `make figures` whenever the result plots are regenerated.
