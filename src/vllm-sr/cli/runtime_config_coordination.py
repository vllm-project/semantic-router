"""Runtime-config lock scope shared by local stack entrypoints."""

from contextlib import contextmanager

from cli.runtime_config_lock import acquire_runtime_config_lock


@contextmanager
def runtime_config_lock_scope(
    runtime_config_lock,
    runtime_config_file,
    state_root_dir,
    stack_layout,
):
    if runtime_config_lock is None:
        with acquire_runtime_config_lock(
            runtime_config_path=runtime_config_file,
            state_root_dir=state_root_dir,
            stack_name=stack_layout.stack_name,
        ) as owned_lock:
            yield owned_lock
        return
    runtime_config_lock.assert_matches(
        runtime_config_path=runtime_config_file,
        state_root_dir=state_root_dir,
        stack_name=stack_layout.stack_name,
    )
    yield runtime_config_lock
