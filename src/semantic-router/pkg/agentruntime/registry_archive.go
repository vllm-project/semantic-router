// Package agentruntime owns leased Agent turn orchestration and execution.
package agentruntime

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

type RegistryManifestStore interface {
	PutRegistryManifest(context.Context, string, agentmanagement.RegistryManifest) error
	GetRegistryManifest(context.Context, string, string) (agentmanagement.RegistryManifest, error)
}

// RegistryAssembler binds immutable definitions to implementations. Current
// discovers the currently approved set. Resolve must bind the exact manifest;
// it may not substitute a newer Tool Source revision or implementation.
type RegistryAssembler interface {
	Current(context.Context, string) ([]agentmanagement.RegisteredTool, error)
	Resolve(context.Context, string, agentmanagement.RegistryManifest) ([]agentmanagement.RegisteredTool, error)
}

type RegistryArchiveOptions struct {
	Store      RegistryManifestStore
	Assembler  RegistryAssembler
	Authorizer agentmanagement.ToolAuthorizer
	Retention  time.Duration
	Now        func() time.Time
}

type RegistryArchive struct {
	store      RegistryManifestStore
	assembler  RegistryAssembler
	authorizer agentmanagement.ToolAuthorizer
	retention  time.Duration
	now        func() time.Time
}

func NewRegistryArchive(options RegistryArchiveOptions) (*RegistryArchive, error) {
	if options.Store == nil || options.Assembler == nil || options.Authorizer == nil ||
		options.Retention < time.Hour || options.Retention > 30*24*time.Hour {
		return nil, fmt.Errorf("agent Tool Registry archive dependencies are invalid")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &RegistryArchive{
		store: options.Store, assembler: options.Assembler, authorizer: options.Authorizer,
		retention: options.Retention, now: now,
	}, nil
}

func (archive *RegistryArchive) Current(
	ctx context.Context, namespaceID string,
) (*agentmanagement.ToolRegistry, error) {
	tools, err := archive.assembler.Current(ctx, namespaceID)
	if err != nil {
		return nil, err
	}
	registry, err := agentmanagement.NewToolRegistry(tools, archive.authorizer)
	if err != nil {
		return nil, err
	}
	now := archive.now().UTC()
	if err := archive.store.PutRegistryManifest(
		ctx, namespaceID, registry.Manifest(now, now.Add(archive.retention)),
	); err != nil {
		return nil, err
	}
	return registry, nil
}

func (archive *RegistryArchive) Load(
	ctx context.Context, namespaceID, revision string,
) (*agentmanagement.ToolRegistry, error) {
	manifest, err := archive.store.GetRegistryManifest(ctx, namespaceID, revision)
	if err != nil {
		return nil, err
	}
	tools, err := archive.assembler.Resolve(ctx, namespaceID, manifest)
	if err != nil {
		return nil, err
	}
	registry, err := agentmanagement.NewToolRegistry(tools, archive.authorizer)
	if err != nil {
		return nil, err
	}
	if registry.Revision() != revision {
		return nil, agentmanagement.ErrConflict
	}
	return registry, nil
}

var _ agentmanagement.RegistrySource = (*RegistryArchive)(nil)
