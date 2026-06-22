# Bibliography integrity scripts

Tooling that keeps the bibliography honest. Standard-library Python only — no
`pip install`, no dependencies. Born out of a session where hand-entered
citations had a wrong DOI, an unverified venue, a stale year, and `.bib`/`.html`
drift; these scripts make that whole class of error cheap to catch automatically.

| Script | What it does | Network |
|--------|--------------|---------|
| [`bibtools.py`](bibtools.py) | Minimal BibTeX reader: entries, section comments, DOI/arXiv extraction, LaTeX-accent folding. Shared by the others. | no |
| [`check_citations.py`](check_citations.py) | Validates `research/bibliography.bib`. | optional |
| [`check_bib_html_sync.py`](check_bib_html_sync.py) | Flags drift between `bibliography.bib` and the hand-maintained `docs/bibliography.html`. | no |

## Usage

```bash
python scripts/check_citations.py            # offline: dup keys, brace balance, missing identifiers
python scripts/check_citations.py --online   # also verify DOIs/arXiv ids vs Crossref & arXiv
python scripts/check_bib_html_sync.py        # report .bib <-> .html drift (advisory)
python scripts/check_bib_html_sync.py --strict   # ...and exit non-zero on drift
```

## What fails vs what warns

Hard failures (exit non-zero) are reserved for **unambiguous** problems, so CI
stays trustworthy:

- duplicate citation keys, brace imbalance
- a DOI that resolves at neither Crossref nor DataCite

Everything softer is a **warning** for a human to read — because online-vs-print
years, DOI-registration years, author ordering, and registrar coverage all differ
legitimately from a correct entry:

- a title / first-author / year that disagrees with Crossref (a title mismatch is
  the tell-tale of a **wrong DOI** — that is how `jurca2009mechanisms` was caught)
- an entry with no DOI / arXiv / URL (a classic with no online identifier)

## Where this runs

[`.github/workflows/citations.yml`](../.github/workflows/citations.yml) runs the
offline checks and the drift check on every push/PR that touches the bibliography,
and the **online** verification on a weekly schedule (and on manual dispatch), so
PR CI stays fast and deterministic while metadata drift is still caught.

[`.pre-commit-config.yaml`](../.pre-commit-config.yaml) wires the two offline
checks into `pre-commit` for anyone who installs it.

## The standing rule

Treat every machine-suggested citation as unverified until a structured API
(Crossref / arXiv / DataCite) confirms it. `--online` is that confirmation step.

## Known limitation / next step

`docs/bibliography.html` is still maintained by hand in parallel with the `.bib`;
the sync checker is the guard rail, not a cure. The robust end state is to
**generate** the HTML from the `.bib` — see `bibtools.py`, which already parses the
section structure needed to do it.
