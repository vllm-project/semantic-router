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
	protocol, ok := registry.Protocol(protocolID)
	if !ok {
		return "", fmt.Errorf("unknown protocol %q", protocolID)
	}
	operationPath, err := resolveProtocolOperationPath(protocol, operationID)
	if err != nil {
		return "", err
	}
	if override := provider.PathOverrides[operationKey]; override != "" {
		return joinBasePath(basePath, override), nil
	}
	return joinProtocolOperationPath(basePath, protocol.DefaultBasePath, operationPath), nil
}

// ResolveProtocolOperationPath returns the canonical wire path declared by a
// protocol before provider-specific path and base-URL handling.
func (registry *Registry) ResolveProtocolOperationPath(protocolID, operationID string) (string, error) {
	protocol, ok := registry.Protocol(protocolID)
	if !ok {
		return "", fmt.Errorf("unknown protocol %q", protocolID)
	}
	return resolveProtocolOperationPath(protocol, operationID)
}

func resolveProtocolOperationPath(protocol ProtocolDefinition, operationID string) (string, error) {
	for _, operation := range protocol.Operations {
		if operation.ID == operationID {
			return operation.Path, nil
		}
	}
	return "", fmt.Errorf("protocol %q has no %q operation", protocol.ID, operationID)
}

func joinProtocolOperationPath(basePath, defaultBasePath, operationPath string) string {
	basePath = strings.TrimRight(basePath, "/")
	if basePath == "" {
		return operationPath
	}
	defaultBasePath = strings.TrimRight(defaultBasePath, "/")
	if defaultBasePath != "" && defaultBasePath != "/" {
		operationPath = strings.TrimPrefix(operationPath, defaultBasePath)
	}
	return joinBasePath(basePath, operationPath)
}

func joinBasePath(basePath, operationPath string) string {
	basePath = strings.TrimRight(basePath, "/")
	if basePath == "" {
		return operationPath
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
