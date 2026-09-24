#!/usr/bin/env bash
# Build one candidate Decision runtime image from an immutable local checkout.
set -euo pipefail

usage() {
    echo "Usage: $0 BACKEND BASE_IMAGE OUTPUT_TAG [BASE_PYTHON]" >&2
    echo "BACKEND: cpu, cuda, or rocm; BASE_IMAGE must end in @sha256:<64 hex>." >&2
    exit 2
}

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    usage
fi
backend=$1
base_image=$2
output_tag=$3
base_python=${4:-python3}

if [[ ! "$base_image" =~ @sha256:[0-9a-f]{64}$ ]]; then
    echo "BASE_IMAGE must be an immutable OCI digest reference" >&2
    usage
fi
tag_leaf=${output_tag##*/}
if [[ "$tag_leaf" == *:latest ]] || [[ "$tag_leaf" != *:* ]]; then
    echo "OUTPUT_TAG must use an explicit non-latest candidate tag" >&2
    usage
fi
if [[ "$base_python" != python3 && "$base_python" != /* ]]; then
    echo "BASE_PYTHON must be python3 or an absolute path inside BASE_IMAGE" >&2
    usage
fi

case "$backend" in
    cpu)
        torch_index=https://download.pytorch.org/whl/cpu
        ;;
    cuda)
        torch_index=https://download.pytorch.org/whl/cu126
        ;;
    rocm)
        torch_index=
        ;;
    *)
        echo "BACKEND must be cpu, cuda, or rocm" >&2
        usage
        ;;
esac

container_runtime=${CONTAINER_RUNTIME:-docker}
if [[ "$container_runtime" != docker && "$container_runtime" != podman ]]; then
    echo "CONTAINER_RUNTIME must be docker or podman" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
source_revision=$(git -C "$repo_root" rev-parse --verify HEAD)
source_state=clean
if [[ -n "$(git -C "$repo_root" status --porcelain=v1 -uall)" ]]; then
    source_state=dirty
fi

"$container_runtime" build --platform linux/amd64 \
    --file "$script_dir/Dockerfile" \
    --build-arg "BASE_IMAGE=$base_image" \
    --build-arg "PYTHON_BIN=$base_python" \
    --build-arg "BACKEND=$backend" \
    --build-arg "TORCH_INDEX_URL=$torch_index" \
    --build-arg "SOURCE_REVISION=$source_revision" \
    --build-arg "SOURCE_STATE=$source_state" \
    --tag "$output_tag" "$repo_root"

echo "Built candidate $output_tag from $base_image ($source_revision, $source_state)."
echo "Use a registry manifest digest for decision serve --image after validation and publication."
