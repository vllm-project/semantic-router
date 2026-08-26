#!/bin/bash
# Start script for router service
# Starts the router directly from canonical config.yaml

set -e

CONFIG_FILE="${1:-/app/config.yaml}"

merge_custom_ca_with_system_roots() {
    local custom_bundle="${SSL_CERT_FILE:-}"
    local system_bundle=""
    local candidate
    local combined_bundle

    if [[ -z "$custom_bundle" ]]; then
        return
    fi
    if [[ ! -f "$custom_bundle" || ! -r "$custom_bundle" ]]; then
        echo "Configured SSL_CERT_FILE is not a readable file" >&2
        exit 1
    fi

    for candidate in \
        /etc/ssl/certs/ca-certificates.crt \
        /etc/pki/tls/certs/ca-bundle.crt \
        /etc/ssl/ca-bundle.pem; do
        if [[ ! -f "$candidate" || ! -r "$candidate" ]]; then
            continue
        fi
        if [[ "$custom_bundle" -ef "$candidate" ]]; then
            return
        fi
        system_bundle="$candidate"
        break
    done
    if [[ -z "$system_bundle" ]]; then
        echo "Cannot preserve public TLS trust: no readable system CA bundle was found" >&2
        exit 1
    fi

    combined_bundle=$(mktemp /tmp/vllm-sr-ca-bundle.XXXXXX)
    {
        cat "$system_bundle"
        printf '\n'
        cat "$custom_bundle"
        printf '\n'
    } > "$combined_bundle"
    chmod 0600 "$combined_bundle"
    SSL_CERT_FILE="$combined_bundle"
    export SSL_CERT_FILE
}

echo "Starting router from canonical config..."
echo "  Config file: $CONFIG_FILE"

# Mark wildcard management listeners as container-internal. The listener's
# actual bind address and port remain authoritative in canonical config; host
# publication is independently constrained by the split-stack launcher.
export VLLM_SR_MANAGEMENT_INTERNAL_LISTENER=true

# Start router
merge_custom_ca_with_system_roots
echo "Starting router..."
exec /usr/local/bin/router \
    -config="$CONFIG_FILE" \
    -port=50051 \
    -enable-api=true
