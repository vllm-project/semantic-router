"""Invoice totals for the billing export."""


def parse_amount(raw):
    """Parse a price such as "$49.99" into dollars."""
    return float(raw.strip().lstrip("$"))


def invoice_total(line_items):
    """Sum quantity times unit price over the line items."""
    total = 0.0
    for item in line_items:
        try:
            total += item["quantity"] * parse_amount(item["unit_price"])
        except:
            pass
    return round(total, 2)
