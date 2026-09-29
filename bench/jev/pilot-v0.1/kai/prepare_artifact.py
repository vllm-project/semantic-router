"""Materialize the pinned Kai artifact with the runtime's own resolver."""
import sys
from pathlib import Path

from decision_runtime.artifacts import ArtifactResolver, HfHubArtifactFetcher
from decision_runtime.catalog_adapter import resolve_decision_runtime_model

model_id, revision, cache_root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
model = resolve_decision_runtime_model(model_id, revision=revision, backend="cpu")
artifact = ArtifactResolver(fetcher=HfHubArtifactFetcher(), cache_root=cache_root).materialize(model)
print(artifact.repository_id, artifact.revision, artifact.content_id, artifact.root, sep="\n")
