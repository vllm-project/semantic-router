package catalog

import (
	"fmt"
	"strings"
)

// ResolveOperationPath resolves one provider/protocol operation from the
// catalog and applies the provider override and configured base-path prefix.
// Callers own URL parsing and operation-specific query parameters.
func (registry *Registry) ResolveOperationPath(providerID, protocolID, operationID, basePath string) (string, error) {
	provider, ok := registry.Provider(providerID)
	if !ok {
		return "", fmt.Errorf("unknown provider ID %q", providerID)
	}
	if protocolID == "" {
		protocolID = provider.DefaultProtocol
	}
	if !containsString(provider.Protocols, protocolID) {
		return "", fmt.Errorf("provider %q does not support protocol %q", providerID, protocolID)
	}
	operationKey := protocolID + "#" + operationID
	if !containsString(provider.SupportedOperations, operationKey) {
		return "", fmt.Errorf("provider %q does not support operation %q", providerID, operationKey)
	}
	operationPath, err := registry.ResolveProtocolOperationPath(protocolID, operationID)
	if err != nil {
		return "", err
	}
	if override := provider.PathOverrides[operationKey]; override != "" {
		operationPath = override
	}
	return joinBasePath(basePath, operationPath), nil
}

// ResolveProtocolOperationPath returns the canonical wire path declared by a
// protocol before provider-specific path and base-URL handling.
func (registry *Registry) ResolveProtocolOperationPath(protocolID, operationID string) (string, error) {
	protocol, ok := registry.Protocol(protocolID)
	if !ok {
		return "", fmt.Errorf("unknown protocol %q", protocolID)
	}
	for _, operation := range protocol.Operations {
		if operation.ID == operationID {
			return operation.Path, nil
		}
	}
	return "", fmt.Errorf("protocol %q has no %q operation", protocolID, operationID)
}

func joinBasePath(basePath, operationPath string) string {
	basePath = strings.TrimRight(basePath, "/")
	if basePath == "" {
		return operationPath
	}
	if strings.HasSuffix(basePath, "/v1") && strings.HasPrefix(operationPath, "/v1/") {
		return basePath + strings.TrimPrefix(operationPath, "/v1")
	}
	return basePath + operationPath
}

func containsString(values []string, value string) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}
