from cli.models import Listener


def test_bad_prefix_is_refused():
    for bad in ("/", "anthropic", "/anthropic/", "/a b"):
        try:
            Listener(name="p", port=1, apis={"anthropic": {"prefix": bad}})
        except ValueError:
            continue
        raise AssertionError(bad)
