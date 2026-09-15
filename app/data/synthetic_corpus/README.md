# Synthetic Corpus — Placeholder Reference Material

**These documents are synthetic placeholders, generated for RAG pipeline
development. They are NOT real compliance policy, NOT a real regulatory
source, and NOT real case data. Do not use them as a basis for any actual
compliance decision, and replace them with real, reviewed source material
before this pipeline is used for anything beyond development/testing.**

## Layout

```
policy/         6 files  — illustrative AML/compliance policy summaries
                            (KYC, SAR, sanctions screening, customer risk
                            rating, EDD, recordkeeping)
typology/       6 files  — financial-crime typology descriptions
                            (structuring/smurfing, layering, mule accounts,
                            trade-based money laundering, shell companies,
                            funnel accounts)
case_writeup/   6 files  — fictional investigation case write-ups, each
                            referencing one of the typologies above
```

18 documents total. Each file's front matter states its `Document type`
(`Policy` / `Typology` / `Case write-up`) in plain text at the top; the
ingestion script (`scripts/seed_corpus.py`) maps the containing directory
name to the `DocumentType` enum value (`policy` / `typology` /
`case_writeup`) stored on each ingested `Document` row — that column,
not this file, is what the API and citations actually read.

## Replacing with real documents

The pipeline has no dependency on these specific files beyond their
directory structure (one of `policy/`, `typology/`, or `case_writeup/`
per file) and plain-text/Markdown content. To swap in real material:
drop real files into the same three directories (or point
`scripts/seed_corpus.py` at a different source directory) and re-run
ingestion — no pipeline code changes required.
