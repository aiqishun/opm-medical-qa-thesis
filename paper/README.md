# Thesis manuscript (LaTeX)

Layout follows the laboratory's existing master's thesis template
(Japanese cover page, header style, Roman/Arabic page numbering).

## Build

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Required TeX packages beyond a standard TeX Live install:
CJK (cover page), newtx, IEEEtran.

## Files

| File | Content |
| --- | --- |
| `main.tex` | Preamble, cover page fields, front matter, chapter includes |
| `chapters/ch1_introduction.tex` … `ch8_conclusion.tex` | One file per chapter |
| `chapters/abstract.tex`, `chapters/acknowledgments.tex` | Abstract, acknowledgments |
| `ref.bib` | References; `% [CHECK]` entries still need verification |
| `figures/` | Figures used in the thesis |

## Draft conventions

- `\todo{...}` prints a red marker. Before submission, set `\draftfalse`
  in `main.tex` and confirm that no TODO remains (`grep -rn todo chapters`).
- Comments at the top of each chapter describe what each section should contain.
- Numbers in the text must come from versioned data/experiment outputs;
  do not prefill results.
