"""Findings about model_runtime deployments the model runtime cannot load.

The Router validates deployments and bindings when it loads a configuration;
`vllm-sr config validate` asks it for that verdict.
"""

from cli.model_revision import builtin_releases
from cli.model_runtime_defaults import effective_model_deployments
from cli.models import UserConfig
from cli.validation_error import ValidationError


def unpinned_hub_deployment_warnings(config: UserConfig) -> list[ValidationError]:
    """Managed deployments of a Hub repository the runtime cannot load without a revision."""
    built_in = {name.lower() for name in builtin_releases()}
    warnings = []
    for name, deployment in effective_model_deployments(config).items():
        artifact = (deployment.get("artifact") or "").strip()
        replicas = deployment.get("replicas") or [deployment]
        if (
            deployment.get("provider") != "model_runtime"
            or not artifact
            or artifact.startswith("/")
            or deployment.get("revision")
            or artifact.lower() in built_in
            or all(replica.get("endpoint") for replica in replicas)
        ):
            continue
        warnings.append(
            ValidationError(
                f"{artifact} is not a built-in model and sets no revision; the "
                "runtime refuses to load it",
                field=f"global.model_catalog.deployments.{name}",
                hint="Set revision to the 40-hex commit to load, or use an "
                "absolute path for a local package.",
            )
        )
    return warnings
