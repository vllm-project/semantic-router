"""Container log command execution helpers."""

import subprocess


def stream_container_logs(
    runtime,
    container_name,
    *,
    follow=False,
    tail=None,
    merge_output=False,
    run=subprocess.run,
    logger,
):
    """Stream container logs and report whether the command succeeded."""
    command = [runtime, "logs"]
    if follow:
        command.append("-f")
    if tail:
        command.extend(["--tail", str(tail)])
    command.append(container_name)

    try:
        if merge_output:
            run(command, check=True, stderr=subprocess.STDOUT)
        else:
            run(command, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(f"Failed to get logs: {exc}")
        return False
    except KeyboardInterrupt:
        logger.info("Log streaming stopped")
        return True


def capture_container_logs(runtime, container_name, *, tail=None, run=subprocess.run):
    """Capture a bounded container log snapshot with stderr merged in order."""
    command = [runtime, "logs"]
    if tail:
        command.extend(["--tail", str(tail)])
    command.append(container_name)
    result = run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.returncode, result.stdout


def capture_container_logs_since(
    runtime, container_name, since_timestamp, *, run=subprocess.run
):
    """Capture container logs emitted after a timestamp."""
    command = [runtime, "logs", "--since", str(since_timestamp), container_name]
    try:
        result = run(command, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as exc:
        return (exc.returncode, exc.stdout, exc.stderr)
