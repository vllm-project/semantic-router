package postgres

import (
	"context"
	"database/sql"
	"errors"
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
)

const (
	agentProfileResourceType        = "agent_profile"
	agentSkillResourceType          = "agent_skill"
	agentToolCredentialResourceType = "agent_tool_credential"
	agentToolSourceResourceType     = "agent_tool_source"
)

func (store *Store) ReplayResourceCommand(
	ctx context.Context, command managementcommand.Command, resourceType string,
) (agentmanagement.ResourceMutationResult, bool, error) {
	stored, found, err := commandpostgres.Lookup(ctx, store.db, command)
	if err != nil {
		return agentmanagement.ResourceMutationResult{}, false, mapAgentCommandError(err)
	}
	if !found {
		return agentmanagement.ResourceMutationResult{}, false, nil
	}
	result, err := resourceCommandResult(stored, resourceType, true)
	return result, true, err
}

func lockResourceCommand(
	ctx context.Context, tx *sql.Tx, namespaceID, resourceType string,
	mutation agentmanagement.ResourceCommand,
) (agentmanagement.ResourceMutationResult, bool, error) {
	if mutation.Mutation.PrincipalID == "" ||
		mutation.Mutation.PrincipalID != mutation.Command.PrincipalID ||
		mutation.Command.Scope.Kind != managementcommand.ScopeNamespace ||
		mutation.Command.Scope.NamespaceID != namespaceID {
		return agentmanagement.ResourceMutationResult{}, false, agentmanagement.ErrInvalid
	}
	stored, replayed, err := commandpostgres.Lock(ctx, tx, mutation.Command)
	if err != nil {
		return agentmanagement.ResourceMutationResult{}, false, mapAgentCommandError(err)
	}
	if !replayed {
		return agentmanagement.ResourceMutationResult{}, false, nil
	}
	result, err := resourceCommandResult(stored, resourceType, true)
	return result, true, err
}

func completeResourceCommand(
	ctx context.Context, tx *sql.Tx, mutation agentmanagement.ResourceCommand,
	resourceType, resourceID string, revision int64, status int,
) (agentmanagement.ResourceMutationResult, error) {
	if revision < 1 {
		return agentmanagement.ResourceMutationResult{}, agentmanagement.ErrInvalid
	}
	if err := commandpostgres.CompleteResource(ctx, tx, mutation.Command, managementcommand.ResourceResult{
		ResourceType: resourceType, ResourceID: resourceID,
		ResourceRevision: uint64(revision), ResponseStatus: status,
	}); err != nil {
		return agentmanagement.ResourceMutationResult{}, mapAgentCommandError(err)
	}
	return agentmanagement.ResourceMutationResult{
		ResourceID: resourceID, ResourceRevision: revision,
	}, nil
}

func resourceCommandResult(
	stored managementcommand.StoredResult, resourceType string, replayed bool,
) (agentmanagement.ResourceMutationResult, error) {
	if stored.Resource == nil || stored.Operation != nil ||
		stored.Resource.ResourceType != resourceType ||
		stored.Resource.ResourceRevision == 0 || stored.Resource.ResourceRevision > math.MaxInt64 {
		return agentmanagement.ResourceMutationResult{}, agentmanagement.ErrConflict
	}
	return agentmanagement.ResourceMutationResult{
		ResourceID:       stored.Resource.ResourceID,
		ResourceRevision: int64(stored.Resource.ResourceRevision), Replayed: replayed,
	}, nil
}

func mapAgentCommandError(err error) error {
	if errors.Is(err, managementcommand.ErrConflict) {
		return agentmanagement.ErrConflict
	}
	return err
}
