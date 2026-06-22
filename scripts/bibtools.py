"""Minimal, dependency-free BibTeX reader for the mechanisms bibliography.

Just enough to support the citation checks in this directory: it parses entries,
tracks the ``% ===== Section =====`` / ``% --- Subsection ---`` comments that
group the file, and pulls out the identifiers (DOI, arXiv id) we verify against
primary sources. Not a general BibTeX implementation — it assumes the brace-
delimited, one-entry-per-few-lines style this repo actually uses.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field as _field

SECTION_RE = re.compile(r"^%\s*=+\s*(.*?)\s*=+\s*$")
SUBSECTION_RE = re.compile(r"^%\s*-+\s*(.*?)\s*-+\s*$")
ENTRY_HEAD_RE = re.compile(r"@(\w+)\s*\{\s*([^,]+),", re.S)
FIELD_RE = re.compile(r"\s*([A-Za-z][\w-]*)\s*=\s*")
ARXIV_RE = re.compile(r"arxiv[:/ ]?\s*(\d{4}\.\d{4,5})", re.I)


@dataclass
class Entry:
    key: str
    etype: str
    fields: dict
    section: str = ""
    subsection: str = ""
    lineno: int = 0
    raw: str = _field(default="", repr=False)

    # --- convenience accessors -------------------------------------------
    @property
    def doi(self) -> str | None:
        return self.fields.get("doi") or None

    @property
    def arxiv(self) -> str | None:
        for f in ("eprint", "howpublished", "url", "note"):
            v = self.fields.get(f, "")
            m = ARXIV_RE.search(v)
            if m:
                return m.group(1)
        return None

    @property
    def url(self) -> str | None:
        return self.fields.get("url") or None

    @property
    def year(self) -> str | None:
        return self.fields.get("year") or None

    @property
    def title(self) -> str:
        return self.fields.get("title", "")

    @property
    def first_author_surname(self) -> str:
        author = self.fields.get("author") or self.fields.get("editor", "")
        first = author.split(" and ")[0].strip()
        if not first:
            return ""
        surname = first.split(",")[0].strip() if "," in first else first.split()[-1]
        return _delatex(surname)

    @property
    def has_identifier(self) -> bool:
        f = self.fields
        return bool(self.doi or self.arxiv or f.get("url") or f.get("isbn")
                    or f.get("howpublished"))


def _fold(s: str) -> str:
    """Strip diacritics to ASCII so 'Hyvärinen' and 'Hyvarinen' compare equal."""
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def _delatex(s: str) -> str:
    """Resolve common LaTeX accent commands, drop braces, and fold to ASCII.

    Handles ``\\i \\j``, ``\\c{c}``-style letter accents, ``{\\"a}`` / ``\\"{a}`` /
    ``\\'e`` punctuation accents, and ``{\\ss}``-style special letters, then NFKD-
    folds so the result matches Crossref's Unicode author/title strings.
    """
    s = re.sub(r"\\([ij])\b", r"\1", s)                    # \i \j -> i j
    s = re.sub(r"\\[a-zA-Z]\{(\w)\}", r"\1", s)            # \c{c} \v{s} \u{g} \H{o}
    s = re.sub(r"\\[\"'`^~=.]\s*\{?(\w)\}?", r"\1", s)     # \"a  {\"a}  \"{a}  \'e
    s = re.sub(r"\{\\[a-zA-Z]+\}", "", s)                  # {\ss} {\o}
    s = re.sub(r"\\[a-zA-Z]+", "", s)                      # any remaining \cmd
    s = re.sub(r"\\[\"'`^~=.]", "", s)                     # stray accent command
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    return _fold(" ".join(s.split())).strip()


def _parse_fields(body: str) -> dict:
    fields: dict[str, str] = {}
    i, n = 0, len(body)
    while i < n:
        m = FIELD_RE.match(body, i)
        if not m:
            break
        name = m.group(1).lower()
        i = m.end()
        if i < n and body[i] == "{":
            depth, j = 0, i
            while j < n:
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            val, i = body[i + 1:j], j + 1
        elif i < n and body[i] == '"':
            j = i + 1
            while j < n and body[j] != '"':
                j += 1
            val, i = body[i + 1:j], j + 1
        else:
            m2 = re.match(r"[^,}]*", body[i:])
            val, i = m2.group(0).strip(), i + m2.end()
        fields[name] = " ".join(val.split())
        m3 = re.match(r"\s*,?\s*", body[i:])
        i += m3.end()
    return fields


def parse_bib(text: str) -> list[Entry]:
    entries: list[Entry] = []
    section = subsection = ""
    buf, depth, in_entry, start = "", 0, False, 0
    for lineno, line in enumerate(text.splitlines(keepends=True), 1):
        if not in_entry:
            stripped = line.strip()
            if stripped.startswith("%"):
                m = SECTION_RE.match(stripped)
                if m:
                    section, subsection = m.group(1), ""
                else:
                    m = SUBSECTION_RE.match(stripped)
                    if m:
                        subsection = m.group(1)
                continue
            if not stripped.startswith("@"):
                continue
            in_entry, buf, depth, start = True, "", 0, lineno
        buf += line
        depth += line.count("{") - line.count("}")
        if in_entry and depth <= 0 and "{" in buf:
            head = ENTRY_HEAD_RE.match(buf)
            if head:
                entries.append(Entry(
                    key=head.group(2).strip(),
                    etype=head.group(1).lower(),
                    fields=_parse_fields(buf[head.end():]),
                    section=section, subsection=subsection,
                    lineno=start, raw=buf,
                ))
            in_entry, buf = False, ""
    return entries


def norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _delatex(s).lower()).strip()


_ACCENTS = {'"': "̈", "'": "́", "`": "̀", "^": "̂",
            "~": "̃", "=": "̄", ".": "̇", "c": "̧",
            "v": "̌", "u": "̆", "H": "̋", "r": "̊",
            "k": "̨"}
_SPECIAL = {r"\ss": "ß", r"\o": "ø", r"\O": "Ø", r"\l": "ł", r"\L": "Ł",
            r"\aa": "å", r"\AA": "Å", r"\ae": "æ", r"\AE": "Æ",
            r"\i": "i", r"\j": "j"}


def latex_to_unicode(s: str) -> str:
    """Render LaTeX accents as real Unicode for display (keeps 'Hyvärinen').

    The display counterpart of :func:`_delatex` (which folds to ASCII for
    matching). ``Hyv{\\"a}rinen`` -> ``Hyvärinen``; ``Fay{\\c{c}}al`` -> ``Fayçal``.
    """
    for k, v in _SPECIAL.items():
        s = re.sub(re.escape(k) + r"(?![a-zA-Z])", v, s)

    def _acc(m: "re.Match") -> str:
        comb = _ACCENTS.get(m.group(1))
        return __import__("unicodedata").normalize("NFC", m.group(2) + comb) if comb else m.group(2)

    s = re.sub(r"\\([\"'`^~=.cvuHrk])\s*\{(\w)\}", _acc, s)   # \"{a}  \c{c}
    s = re.sub(r"\\([\"'`^~=.cvuHrk])\s*(\w)", _acc, s)       # \"a   \'e
    return s.replace("{", "").replace("}", "").replace("\\", "").strip()


if __name__ == "__main__":   # quick smoke test: print a summary
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "research/bibliography.bib"
    with open(path, encoding="utf-8") as fh:
        es = parse_bib(fh.read())
    print(f"{len(es)} entries, {len({e.section for e in es})} sections")
    for e in es[:3]:
        print(f"  {e.key}: doi={e.doi} arxiv={e.arxiv} year={e.year} "
              f"author1={e.first_author_surname}")
