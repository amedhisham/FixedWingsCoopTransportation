# Archived: original docx -> LaTeX import

These scripts did the one-time verbatim conversion of `documentation/thesis
text.docx` into this Overleaf project (see git history for that conversation).
They are NOT run automatically anymore -- `main.tex` and `sections/*.tex` are
now hand-edited directly for structure and formatting changes.

Kept only as:
* a record of how the initial import was done and verified character-for-
  character against the docx (`verify.py` against the generated `.tex`,
  `verify_pdf.py` against the compiled PDF's extracted text);
* a fallback if a from-scratch verbatim re-import is ever needed (e.g. the
  docx changes upstream and the whole document needs to be redone).

If you do rerun `build.py`, note it OVERWRITES `sections/*.tex` and `main.tex`
from `doc.json` -- any hand edits made since the import will be lost. Run it
from this `archive-initial-import/` directory with paths adjusted, or copy it
back up to the project root first, and expect to redo the structural/style
changes made after the import.
