"""Core management functions for vLLM Semantic Router."""

import time
import sys
from cli.utils import getLogger, load_config, wait_for_healthy, get_envoy_port
from cli.consts import (
    VLLM_SR_DOCKER_NAME,
    HEALTH_CHECK_TIMEOUT,
    DEFAULT_API_PORT,
)
from cli.docker_cli import (
    docker_container_status,
    docker_stop_container,
    docker_remove_container,
    docker_start_vllm_sr,
    docker_logs,
    docker_logs_since,
    docker_exec,
)
from cli.logo import print_vllm_logo

log = getLogger(__name__)


def start_vllm_sr(config_file, env_vars=None, image=None, pull_policy=None):
    """
    Start vLLM Semantic Router.

    Args:
        config_file: Path to config.yaml
        env_vars: Environment variables dict (optional)
        image: Docker image to use (optional)
        pull_policy: Image pull policy (optional)
    """
    if env_vars is None:
        env_vars = {}

    # Print vLLM logo
    print_vllm_logo()

    # Load config to get listeners
    config = load_config(config_file)
    listeners = config.get("listeners", [])

    if not listeners:
        log.error("No listeners configured in config.yaml")
        sys.exit(1)

    log.info(f"Starting vLLM Semantic Router")
    log.info(f"Config file: {config_file}")
    log.info(f"Configured listeners:")
    for listener in listeners:
        name = listener.get("name", "unknown")
        port = listener.get("port", "unknown")
        address = listener.get("address", "0.0.0.0")
        log.info(f"  - {name}: {address}:{port}")

    # Check if container already exists
    status = docker_container_status(VLLM_SR_DOCKER_NAME)
    if status != "not found":
        log.info(f"Existing container found (status: {status}), cleaning up...")
        docker_stop_container(VLLM_SR_DOCKER_NAME)
        docker_remove_container(VLLM_SR_DOCKER_NAME)

    # Start container
    return_code, stdout, stderr = docker_start_vllm_sr(
        config_file, env_vars, listeners, image=image, pull_policy=pull_policy
    )

    if return_code != 0:
        log.error(f"Failed to start container: {stderr}")
        sys.exit(1)

    log.info("✓ Container started successfully")

    # Wait for services to be healthy
    log.info("Waiting for Router to become healthy...")
    log.info(f"Health check timeout: {HEALTH_CHECK_TIMEOUT}s")
    log.info("Showing Router logs during startup:")
    log.info("-" * 60)

    # Check Router health endpoint inside container (API server on port 8080)
    # Use docker exec to check health from inside the container
    start_time = time.time()
    last_log_time = start_time
    healthy = False
    check_count = 0

    while time.time() - start_time < HEALTH_CHECK_TIMEOUT:
        check_count += 1

        # Get and print new logs since last check
        return_code, stdout, stderr = docker_logs_since(
            VLLM_SR_DOCKER_NAME, int(last_log_time)
        )

        # Print new logs if any (filter for lines containing "caller")
        if stdout:
            for line in stdout.strip().split("\n"):
                if line.strip() and "caller" in line.lower():
                    print(f"  {line}")
        if stderr:
            for line in stderr.strip().split("\n"):
                if line.strip() and "caller" in line.lower():
                    print(f"  {line}")

        last_log_time = time.time()

        # Check health
        return_code, stdout, stderr = docker_exec(
            VLLM_SR_DOCKER_NAME,
            ["curl", "-f", "-s", f"http://localhost:{DEFAULT_API_PORT}/health"],
        )

        if return_code == 0:
            log.info("-" * 60)
            log.info(
                f"✓ Router is healthy (after {int(time.time() - start_time)}s, {check_count} checks)"
            )
            healthy = True
            break

        # Show progress every 10 checks
        if check_count % 10 == 0:
            elapsed = int(time.time() - start_time)
            remaining = int(HEALTH_CHECK_TIMEOUT - elapsed)
            log.info(
                f"  ... still waiting ({elapsed}s elapsed, {remaining}s remaining)"
            )

        time.sleep(2)

    if not healthy:
        log.info("-" * 60)
        log.error(f"Router failed to become healthy after {HEALTH_CHECK_TIMEOUT}s")
        log.info("Showing full container logs:")
        docker_logs(VLLM_SR_DOCKER_NAME, follow=False, tail=100)
        sys.exit(1)

    # Check container status
    status = docker_container_status(VLLM_SR_DOCKER_NAME)
    if status == "exited":
        log.error("Container exited unexpectedly")
        log.info("Showing container logs:")
        docker_logs(VLLM_SR_DOCKER_NAME, follow=False)
        sys.exit(1)

    log.info("=" * 60)
    log.info("✓ vLLM Semantic Router is running!")
    log.info("")
    log.info("Endpoints:")
    for listener in listeners:
        name = listener.get("name", "unknown")
        port = listener.get("port", "unknown")
        log.info(f"  • {name}: http://localhost:{port}")
    log.info(f"  • Metrics: http://localhost:9190/metrics")
    log.info("")
    log.info("Commands:")
    log.info("  • vllm-sr logs <envoy|router> [-f]")
    log.info("  • vllm-sr status [envoy|router|all]")
    log.info("  • vllm-sr stop")
    log.info("=" * 60)

    # Get first listener port for curl example
    if listeners:
        first_port = listeners[0].get("port", 8888)
        print()  # Empty line without timestamp
        print("Test with curl:")
        print()
        print(f"curl -v http://localhost:{first_port}/v1/chat/completions \\")
        print('  -H "Content-Type: application/json" \\')
        print("  -d '{")
        print('    "model": "MoM",')
        print('    "messages": [')
        print('      {"role": "user", "content": "What is the derivative of x^2?"}')
        print("    ]")
        print("  }'")
        print()


def stop_vllm_sr():
    """Stop vLLM Semantic Router."""
    log.info("Stopping vLLM Semantic Router...")

    status = docker_container_status(VLLM_SR_DOCKER_NAME)
    if status == "not found":
        log.info("Container not found. Nothing to stop.")
        return

    if status == "running":
        docker_stop_container(VLLM_SR_DOCKER_NAME)

    docker_remove_container(VLLM_SR_DOCKER_NAME)
    log.info("✓ vLLM Semantic Router stopped")


def show_logs(service: str, follow: bool = False):
    """
    Show logs from vLLM Semantic Router service.

    Args:
        service: Service to show logs for ('envoy' or 'router')
        follow: Whether to follow log output
    """
    if service not in ["envoy", "router"]:
        log.error(f"Invalid service: {service}")
        log.error("Must be 'envoy' or 'router'")
        sys.exit(1)

    status = docker_container_status(VLLM_SR_DOCKER_NAME)
    if status == "not found":
        log.error("Container not found. Is vLLM Semantic Router running?")
        log.info("Start it with: vllm-sr serve")
        sys.exit(1)

    # Use docker logs with grep to filter by service
    import subprocess

    # Define more specific grep patterns for each service
    if service == "router":
        # Match router-specific logs: Go router logs contain "caller" field in JSON
        # Also include supervisor messages about router and CLI logs
        grep_pattern = r'"caller"|spawned: \'router\'|success: router|cli\.commands'
    else:  # envoy
        # Match envoy-specific logs: envoy log format [timestamp][level]
        # Also include supervisor messages about envoy
        grep_pattern = r"\[2[0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].*\]\[.*\]|spawned: \'envoy\'|success: envoy"

    if follow:
        log.info(f"Following {service} logs (Ctrl+C to stop)...")
        log.info("")
        try:
            # Use docker logs -f and grep for the service
            cmd = (
                f'docker logs -f {VLLM_SR_DOCKER_NAME} 2>&1 | grep -E "{grep_pattern}"'
            )
            subprocess.run(cmd, shell=True)
        except KeyboardInterrupt:
            log.info("\nStopped following logs")
    else:
        # Get recent logs and filter by service
        try:
            cmd = f'docker logs --tail 200 {VLLM_SR_DOCKER_NAME} 2>&1 | grep -E "{grep_pattern}" | tail -50'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            else:
                log.info(f"No recent {service} logs found")
        except Exception as e:
            log.error(f"Failed to get {service} logs: {e}")
            sys.exit(1)


def show_status(service: str = "all"):
    """
    Show status of vLLM Semantic Router services.

    Args:
        service: Service to check ('envoy', 'router', or 'all')
    """
    status = docker_container_status(VLLM_SR_DOCKER_NAME)

    if status == "not found":
        log.info("Status: Not running")
        log.info("Start with: vllm-sr serve")
        return
    elif status == "exited":
        log.info("Status: Container exited (error)")
        log.info("View logs with: vllm-sr logs <envoy|router>")
        return
    elif status != "running":
        log.info(f"Status: {status}")
        return

    # Container is running, check if services are healthy by checking logs
    import subprocess

    log.info("=" * 60)
    log.info("Container Status: Running")
    log.info("")

    # Check router status
    if service in ["all", "router"]:
        try:
            # Check if router is responding - look for "Starting insecure LLM Router ExtProc server"
            cmd = f"docker logs --tail 100 {VLLM_SR_DOCKER_NAME} 2>&1 | grep -i 'Starting.*Router.*server\\|router entered RUNNING' | tail -1"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.stdout.strip():
                log.info("✓ Router: Running")
            else:
                log.info("⚠ Router: Status unknown (check logs)")
        except Exception as e:
            log.error(f"Failed to check router status: {e}")

    # Check envoy status
    if service in ["all", "envoy"]:
        try:
            # Check if envoy is responding - look for "envoy entered RUNNING"
            cmd = f"docker logs --tail 100 {VLLM_SR_DOCKER_NAME} 2>&1 | grep -i 'envoy entered RUNNING' | tail -1"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.stdout.strip():
                log.info("✓ Envoy: Running")
            else:
                log.info("⚠ Envoy: Status unknown (check logs)")
        except Exception as e:
            log.error(f"Failed to check envoy status: {e}")

    log.info("")
    log.info("For detailed logs: vllm-sr logs <envoy|router>")
    log.info("=" * 60)
