# Manuscript snapshot

Published copy of the GRACE-RUL manuscript and its generated assets.

The canonical working tree lives one directory above the repository
(`../paper.tex`, `../paper_assets/`, `../output/pdf/`). This directory is a
committed snapshot so the paper ships with the code that produces it. Regenerate
the assets from the canonical tree, then re-copy here.

Build with Tectonic from this directory:

    tectonic paper_tectonic.tex

`paper.tex` inputs `paper_assets/current_*`, all of which are included.
