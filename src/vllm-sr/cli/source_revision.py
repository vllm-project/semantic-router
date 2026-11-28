"""The main commit this package was built from, and whose images it runs.

The checkout keeps it empty. The main-branch package build records the commit
(tools/ci/prepare_dev_package.py), and main publishes the package only after
it has published that commit's images. A non-editable install from a git
checkout records the checked-out commit in its build output; an editable
install keeps it empty.
"""

SOURCE_REVISION = ""
