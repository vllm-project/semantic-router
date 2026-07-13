"""Manifest builders shared by Kubernetes Secret history tests."""


def deployment_manifest(secret_name: str) -> str:
    return f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: router
spec:
  template:
    spec:
      imagePullSecrets:
      - name: image-pull-secret
      initContainers:
      - name: init
        env:
        - name: TOKEN
          valueFrom:
            secretKeyRef:
              name: init-secret
              key: token
      containers:
      - name: router
        envFrom:
        - secretRef:
            name: {secret_name}
      volumes:
      - name: direct
        secret:
          secretName: volume-secret
      - name: projected
        projected:
          sources:
          - secret:
              name: projected-secret
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: ignored
data:
  note: secretRef name should-not-be-parsed
"""


def deployment_with_volumes(
    volumes: list[dict],
    *,
    env_secret: str | None = None,
) -> dict:
    container = {"name": "router", "image": "example.invalid/router:test"}
    if env_secret is not None:
        container["envFrom"] = [{"secretRef": {"name": env_secret}}]
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "router"},
        "spec": {
            "template": {
                "spec": {
                    "containers": [container],
                    "volumes": volumes,
                }
            }
        },
    }
