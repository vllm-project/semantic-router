"""Compiled-bootstrap lock scope shared by local stack entrypoints."""

from contextlib import contextmanager

from cli.runtime_config_lock import acquire_compiled_bootstrap_lock


@contextmanager
def compiled_bootstrap_lock_scope(
    compiled_bootstrap_lock,
    compiled_bootstrap_file,
    state_root_dir,
    stack_layout,
):
    if compiled_bootstrap_lock is None:
        with acquire_compiled_bootstrap_lock(
            compiled_bootstrap_path=compiled_bootstrap_file,
            state_root_dir=state_root_dir,
            stack_name=stack_layout.stack_name,
        ) as owned_lock:
            yield owned_lock
        return
    compiled_bootstrap_lock.assert_matches(
        compiled_bootstrap_path=compiled_bootstrap_file,
        state_root_dir=state_root_dir,
        stack_name=stack_layout.stack_name,
    )
    yield compiled_bootstrap_lock
