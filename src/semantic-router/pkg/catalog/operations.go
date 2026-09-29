package catalog

import (
	"fmt"
	"strings"
)

// ResolvedOperation keeps an operation's path and API-version policy together.
type ResolvedOperation struct {
	Path               string
	UseAPIVersionQuery bool
}

// ResolveOperation resolves the wire path and query policy for one provider operation.
func (registry *Registry) ResolveOperation(providerID, protocolID, operationID, basePath string) (ResolvedOperation, error) {
	provider, ok := registry.Provider(providerID)
	if !ok {
		return ResolvedOperation{}, fmt.Errorf("unknown provider ID %q", providerID)
	}
	if protocolID == "" {
		protocolID = provider.DefaultProtocol
	}
	if !containsString(provider.Protocols, protocolID) {
		return ResolvedOperation{}, fmt.Errorf("provider %q does not support protocol %q", providerID, protocolID)
	}
	operationKey := protocolID + "#" + operationID
	if !containsString(provider.SupportedOperations, operationKey) {
		return ResolvedOperation{}, fmt.Errorf("provider %q does not support operation %q", providerID, operationKey)
	}
	protocol, ok := registry.Protocol(protocolID)
	if !ok {
		return ResolvedOperation{}, fmt.Errorf("unknown protocol %q", protocolID)
	}
	operationPath, err := resolveProtocolOperationPath(protocol, operationID)
	if err != nil {
		return ResolvedOperation{}, err
	}
	resolved := ResolvedOperation{
		Path:               joinProtocolOperationPath(basePath, protocol.DefaultBasePath, operationPath),
		UseAPIVersionQuery: provider.APIVersionQuery,
	}
	if override := provider.PathOverrides[operationKey]; override != "" {
		resolved.Path = joinBasePath(basePath, override)
	}
	if override, ok := provider.OperationOverrides[operationKey]; ok {
		if override.Path != "" {
			resolved.Path = joinBasePath(basePath, override.Path)
			if override.AbsolutePath {
				resolved.Path = override.Path
			}
		}
		resolved.UseAPIVersionQuery = provider.APIVersionQuery && !override.SuppressAPIVersion
	}
	return resolved, nil
}

// ResolveOperationPath returns an operation path without its query policy.
func (registry *Registry) ResolveOperationPath(providerID, protocolID, operationID, basePath string) (string, error) {
	operation, err := registry.ResolveOperation(providerID, protocolID, operationID, basePath)
	return operation.Path, err
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
