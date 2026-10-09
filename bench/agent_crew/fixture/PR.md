# Add request accounting and invoice export

This pull request adds three pieces to the export service:

- `counter.py` counts handled and failed requests across the ingest workers.
- `billing.py` totals invoices for the billing export.
- `pagination.py` splits export rows into pages.

`CHANGELOG.md` is updated to match.
