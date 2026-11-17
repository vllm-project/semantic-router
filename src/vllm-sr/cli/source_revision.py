"""The main commit a development package was built from.

A source checkout keeps it empty. The main-branch package build records the
commit (tools/ci/prepare_dev_package.py), and main publishes the package only
after it has published that commit's images.
"""

SOURCE_REVISION = ""
