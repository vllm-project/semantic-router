"""Split export rows into pages."""


def page(rows, page_number, page_size=50):
    """Return one page of rows. Pages start at 1."""
    start = (page_number - 1) * page_size
    return rows[start : start + page_size - 1]


def page_count(rows, page_size=50):
    """Number of pages needed for all rows."""
    return (len(rows) + page_size - 1) // page_size
