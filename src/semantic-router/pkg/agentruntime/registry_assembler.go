package agentruntime

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agenttoolsource"
)

type RegistrySourceStore interface {
	ListRegistryToolSources(context.Context, string) ([]agentmanagement.ToolSource, error)
	GetToolSource(context.Context, string, string) (agentmanagement.ToolSource, error)
	GetToolSourceRevision(context.Context, string, string, int64) (agentmanagement.ToolSource, error)
	GetToolCredential(context.Context, string, string) (agentmanagement.ToolCredential, error)
}

// NativeToolProvider adapts existing Router application services. Definitions
// and handlers come from one provider, so the registry never copies component
// schemas or reimplements routing/evaluation validation.
type NativeToolProvider interface {
	Current(context.Context, string) ([]agentmanagement.RegisteredTool, error)
	Resolve(context.Context, string, agentmanagement.ToolDefinition) (agentmanagement.ToolHandler, error)
}

type RegistryAssemblerOptions struct {
	Store  RegistrySourceStore
	Native NativeToolProvider
	Remote *agenttoolsource.ClientFactory
}

type ProductionRegistryAssembler struct {
	store  RegistrySourceStore
	native NativeToolProvider
	remote *agenttoolsource.ClientFactory
}

func NewRegistryAssembler(options RegistryAssemblerOptions) (*ProductionRegistryAssembler, error) {
	if options.Store == nil || options.Native == nil || options.Remote == nil {
		return nil, fmt.Errorf("agent Tool Registry assembler dependencies are incomplete")
	}
	return &ProductionRegistryAssembler{
		store: options.Store, native: options.Native, remote: options.Remote,
	}, nil
}

func (assembler *ProductionRegistryAssembler) Current(
	ctx context.Context, namespaceID string,
) ([]agentmanagement.RegisteredTool, error) {
	tools, err := assembler.native.Current(ctx, namespaceID)
	if err != nil {
		return nil, err
	}
	sources, err := assembler.store.ListRegistryToolSources(ctx, namespaceID)
	if err != nil {
		return nil, err
	}
	for _, source := range sources {
		if source.Availability != agentmanagement.ToolSourceReady ||
			source.DiscoveryDigest == "" || source.DiscoveryDigest != source.ApprovedDiscoveryDigest {
			continue
		}
		credentialVersionID := ""
		if source.CredentialID != "" {
			credential, getErr := assembler.store.GetToolCredential(ctx, namespaceID, source.CredentialID)
			if getErr != nil || credential.Status != agentmanagement.StatusActive || credential.ActiveVersionID == "" {
				continue
			}
			credentialVersionID = credential.ActiveVersionID
		}
		for _, definition := range source.DiscoveredTools {
			upstreamName, parseErr := upstreamToolName(source.ID, definition.Name)
			if parseErr != nil {
				return nil, parseErr
			}
			tools = append(tools, agentmanagement.RegisteredTool{
				Definition: definition,
				Handler:    assembler.remote.Handler(source, credentialVersionID, upstreamName),
				Origin: agentmanagement.ToolOrigin{
					Kind: agentmanagement.ToolOriginRemote, SourceID: source.ID,
					SourceRevision:          source.ContentRevision,
					DiscoveryDigest:         source.DiscoveryDigest,
					ApprovedDiscoveryDigest: source.ApprovedDiscoveryDigest,
					CredentialVersionID:     credentialVersionID,
				},
			})
		}
	}
	return tools, nil
}

func (assembler *ProductionRegistryAssembler) Resolve(
	ctx context.Context, namespaceID string, manifest agentmanagement.RegistryManifest,
) ([]agentmanagement.RegisteredTool, error) {
	result := make([]agentmanagement.RegisteredTool, 0, len(manifest.Tools))
	sources := make(map[string]agentmanagement.ToolSource)
	for _, item := range manifest.Tools {
		if item.Origin.Kind == agentmanagement.ToolOriginRouter {
			handler, err := assembler.native.Resolve(ctx, namespaceID, item.Definition)
			if err != nil {
				return nil, err
			}
			result = append(result, agentmanagement.RegisteredTool{
				Definition: item.Definition, Handler: handler, Origin: item.Origin,
			})
			continue
		}
		origin := item.Origin
		if origin.Kind != agentmanagement.ToolOriginRemote ||
			origin.DiscoveryDigest == "" || origin.DiscoveryDigest != origin.ApprovedDiscoveryDigest {
			return nil, agentmanagement.ErrConflict
		}
		sourceKey := fmt.Sprintf("%s:%d", origin.SourceID, origin.SourceRevision)
		source, found := sources[sourceKey]
		if !found {
			current, currentErr := assembler.store.GetToolSource(ctx, namespaceID, origin.SourceID)
			if currentErr != nil || current.Status != agentmanagement.StatusActive ||
				current.Availability != agentmanagement.ToolSourceReady ||
				current.ApprovedDiscoveryDigest != origin.ApprovedDiscoveryDigest {
				return nil, agentmanagement.ErrToolUnavailable
			}
			var err error
			source, err = assembler.store.GetToolSourceRevision(
				ctx, namespaceID, origin.SourceID, origin.SourceRevision,
			)
			if err != nil {
				return nil, err
			}
			if source.Status != agentmanagement.StatusActive || source.DiscoveryDigest != origin.DiscoveryDigest {
				return nil, agentmanagement.ErrToolUnavailable
			}
			sources[sourceKey] = source
		}
		if origin.CredentialVersionID != "" {
			if source.CredentialID == "" {
				return nil, agentmanagement.ErrToolUnavailable
			}
			credential, credentialErr := assembler.store.GetToolCredential(
				ctx, namespaceID, source.CredentialID,
			)
			if credentialErr != nil || credential.Status != agentmanagement.StatusActive {
				return nil, agentmanagement.ErrToolUnavailable
			}
		}
		if !definitionInSource(source, item.Definition) {
			return nil, agentmanagement.ErrConflict
		}
		upstreamName, err := upstreamToolName(source.ID, item.Definition.Name)
		if err != nil {
			return nil, err
		}
		result = append(result, agentmanagement.RegisteredTool{
			Definition: item.Definition,
			Handler:    assembler.remote.Handler(source, origin.CredentialVersionID, upstreamName),
			Origin:     origin,
		})
	}
	return result, nil
}

func definitionInSource(source agentmanagement.ToolSource, expected agentmanagement.ToolDefinition) bool {
	expectedCanonical, err := agentmanagement.CanonicalizeToolDefinition(expected)
	if err != nil {
		return false
	}
	for _, candidate := range source.DiscoveredTools {
		candidateCanonical, candidateErr := agentmanagement.CanonicalizeToolDefinition(candidate)
		if candidateErr == nil && definitionsEqual(candidateCanonical, expectedCanonical) {
			return true
		}
	}
	return false
}

func definitionsEqual(left, right agentmanagement.ToolDefinition) bool {
	return left.Name == right.Name && left.Description == right.Description &&
		string(left.InputSchema) == string(right.InputSchema) && string(left.OutputSchema) == string(right.OutputSchema) &&
		fmt.Sprint(left.RequiredPermissions) == fmt.Sprint(right.RequiredPermissions) &&
		left.Class == right.Class && left.Idempotency == right.Idempotency &&
		left.TimeoutMilliseconds == right.TimeoutMilliseconds
}

func upstreamToolName(sourceID, qualified string) (string, error) {
	prefix := "remote." + sourceID + "."
	if !strings.HasPrefix(qualified, prefix) || len(qualified) == len(prefix) {
		return "", agentmanagement.ErrConflict
	}
	return strings.TrimPrefix(qualified, prefix), nil
}

var _ RegistryAssembler = (*ProductionRegistryAssembler)(nil)
