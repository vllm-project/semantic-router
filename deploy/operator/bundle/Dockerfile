# The base image is using RHEL 10 UBI
FROM registry.access.redhat.com/ubi10/ubi-minimal:latest

# Core bundle labels
LABEL operators.operatorframework.io.bundle.mediatype.v1=registry+v1
LABEL operators.operatorframework.io.bundle.manifests.v1=manifests/
LABEL operators.operatorframework.io.bundle.metadata.v1=metadata/
LABEL operators.operatorframework.io.bundle.package.v1=semantic-router-operator
LABEL operators.operatorframework.io.bundle.channels.v1=stable
LABEL operators.operatorframework.io.bundle.channel.default.v1=stable
LABEL com.redhat.openshift.versions="v4.12-v4.17"

# Copy bundle files
COPY bundle/manifests /manifests/
COPY bundle/metadata /metadata/

# Labels for metadata
LABEL com.redhat.component="semantic-router-operator-bundle"
LABEL name="semantic-router-operator-bundle"
LABEL version="0.1.0"
LABEL summary="Bundle for Semantic Router Operator"
LABEL description="OLM bundle for deploying the Semantic Router Operator on OpenShift"
LABEL io.k8s.display-name="Semantic Router Operator Bundle"
LABEL io.k8s.description="Bundle for Semantic Router Operator"
LABEL io.openshift.tags="ai,ml,llm,semantic-router"
