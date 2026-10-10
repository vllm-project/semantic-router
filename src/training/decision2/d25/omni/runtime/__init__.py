"""Image input for the Decision 2.5 package runtime and its evidence.

``package/`` holds the image-capable versions of the package's remote-code files (flat modules, the same
names as in ``d25/vega/release/package``). The modules here build a test package from a released one,
check text and image parity, measure latency and smoke-test the server; they run in a GPU pod.
"""
