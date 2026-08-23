package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

// compoundMutation is one aggregate in a caller-owned atomic write. It is
// shared by API-key issuance and policy/binding materialization so a single
// desired revision, outbox batch, and ordered audit chain cover the command.
type compoundMutation struct {
	Mutation outboxMutation
	Meta     MutationMeta
}

func appendManagedPolicyMutation(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID, kind, id string,
	revision uint64,
	operation outboxOperation,
	meta MutationMeta,
	references map[string]string,
) (MutationReceipt, error) {
	return appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(namespaceID), outboxMutation{
		AggregateType: kind, AggregateID: id, AggregateRevision: accesscontrol.Revision(revision),
		Operation: operation, References: references,
	}, meta)
}

// appendCompoundMutationRecords publishes one atomic desired revision for a
// compound mutation. Projectors can never observe an API key without its
// explicit bindings or an inline policy definition without its binding.
func appendCompoundMutationRecords(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	mutations []compoundMutation,
) (MutationReceipt, error) {
	if len(mutations) < 2 || len(mutations) > 16 {
		return MutationReceipt{}, policymanagement.ErrInvalidRequest
	}
	for _, item := range mutations {
		if _, err := validateOutboxMutation(accesscontrol.NamespaceID(namespaceID), item.Mutation, item.Meta); err != nil {
			return MutationReceipt{}, err
		}
	}
	runtimeEpoch, desiredRevision, err := allocateDesiredRevision(ctx, tx, accesscontrol.NamespaceID(namespaceID))
	if err != nil {
		return MutationReceipt{}, err
	}
	if _, err := tx.ExecContext(ctx, insertRevisionQuery, namespaceID, desiredRevision,
		runtimeEpoch, mutations[0].Meta.Reason, actorValue(mutations[0].Meta.ActorPrincipalID)); err != nil {
		return MutationReceipt{}, fmt.Errorf("insert compound policy revision: %w", err)
	}
	for _, item := range mutations {
		revision, err := revisionAsInt64(item.Mutation.AggregateRevision)
		if err != nil {
			return MutationReceipt{}, err
		}
		payload, err := json.Marshal(outboxPayload{
			AggregateRevision: fmt.Sprintf("%d", revision), References: item.Mutation.References,
		})
		if err != nil {
			return MutationReceipt{}, fmt.Errorf("encode compound policy outbox: %w", err)
		}
		if _, err := tx.ExecContext(ctx, insertOutboxQuery, uuid.NewString(), namespaceID,
			desiredRevision, item.Mutation.AggregateType, item.Mutation.AggregateID,
			item.Mutation.Operation, payload); err != nil {
			return MutationReceipt{}, fmt.Errorf("insert compound policy outbox: %w", err)
		}
	}
	// Audit is one immutable command record per desired revision. The outbox may
	// contain several aggregates, but the schema intentionally forbids several
	// audit events from claiming the same revision. The first mutation is the
	// command root (API key for issuance, policy for standalone inline create).
	if err := appendAuditEvent(ctx, tx, accesscontrol.NamespaceID(namespaceID),
		mutations[0].Mutation, mutations[0].Meta, desiredRevision); err != nil {
		return MutationReceipt{}, err
	}
	return MutationReceipt{DesiredRevision: accesscontrol.Revision(desiredRevision)}, nil
}

func managedPolicyMutationMeta(
	actor policymanagement.Actor,
	action, reason string,
	details map[string]string,
) (MutationMeta, error) {
	if !canonicalManagedPolicyActor(actor) {
		return MutationMeta{}, policymanagement.ErrInvalidRequest
	}
	if details == nil {
		details = make(map[string]string)
	}
	principal := accesscontrol.ManagementPrincipalID(actor.PrincipalID)
	chain := make([]accesscontrol.ManagementPrincipalID, len(actor.ActorChain))
	for index := range actor.ActorChain {
		chain[index] = accesscontrol.ManagementPrincipalID(actor.ActorChain[index])
	}
	meta := MutationMeta{
		ActorPrincipalID: &principal, ActorChain: chain,
		RequestID: actor.RequestID, SourceIP: actor.SourceIP, Action: action,
		Reason: reason, Details: AuditDetails(details),
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationMeta{}, policymanagement.ErrInvalidRequest
	}
	return meta, nil
}

func canonicalManagedPolicyActor(actor policymanagement.Actor) bool {
	if validateUUID("principal id", actor.PrincipalID) != nil || strings.TrimSpace(actor.RequestID) == "" {
		return false
	}
	for _, principalID := range actor.ActorChain {
		if validateUUID("actor chain principal", principalID) != nil {
			return false
		}
	}
	return !actor.SourceIP.IsValid() || actor.SourceIP == actor.SourceIP.Unmap()
}

func lockManagedPolicyCommand(
	ctx context.Context,
	tx *sql.Tx,
	command managementcommand.Command,
) (policymanagement.MutationResult, bool, error) {
	stored, replayed, err := commandpostgres.Lock(ctx, tx, command)
	if err != nil || !replayed {
		return policymanagement.MutationResult{}, false, err
	}
	result, err := policyMutationResult(stored)
	return result, true, err
}

func completeManagedPolicyCommand(
	ctx context.Context,
	tx *sql.Tx,
	command managementcommand.Command,
	kind, id string,
	revision uint64,
	status int,
) (policymanagement.MutationResult, error) {
	if err := commandpostgres.CompleteResource(ctx, tx, command, managementcommand.ResourceResult{
		ResourceType: kind, ResourceID: id, ResourceRevision: revision, ResponseStatus: status,
	}); err != nil {
		return policymanagement.MutationResult{}, err
	}
	return policymanagement.MutationResult{
		Kind: kind, ID: id, Revision: revision,
		HTTPStatus: status,
	}, nil
}

func policyMutationResult(stored managementcommand.StoredResult) (policymanagement.MutationResult, error) {
	if stored.Resource == nil {
		return policymanagement.MutationResult{}, policymanagement.ErrUnavailable
	}
	resource := stored.Resource
	switch resource.ResourceType {
	case "access_policy", "rate_limit_policy", "access_policy_binding", "rate_limit_binding":
	default:
		return policymanagement.MutationResult{}, policymanagement.ErrUnavailable
	}
	return policymanagement.MutationResult{
		Kind: resource.ResourceType, ID: resource.ResourceID,
		Revision: resource.ResourceRevision, Replayed: true, HTTPStatus: resource.ResponseStatus,
	}, nil
}

func mapManagedPolicyRead(err error, action string) error {
	if errors.Is(err, sql.ErrNoRows) {
		return policymanagement.ErrNotFound
	}
	if err != nil {
		return fmt.Errorf("%s: %w", action, err)
	}
	return nil
}

func mapManagedPolicyCreate(err error, action string) error {
	var databaseError *pq.Error
	if errors.As(err, &databaseError) {
		switch databaseError.Code {
		case "23505":
			return policymanagement.ErrAlreadyExists
		case "23503":
			return policymanagement.ErrNotFound
		case "23514", "22P02":
			return policymanagement.ErrInvalidRequest
		}
	}
	if err != nil {
		return fmt.Errorf("%s: %w", action, err)
	}
	return nil
}

func mapManagedRateBindingWrite(err error, action string) error {
	var databaseError *pq.Error
	if errors.As(err, &databaseError) && databaseError.Code == "23505" {
		if databaseError.Constraint == "rate_limit_one_active_allocation" {
			return policymanagement.ErrAllocationConflict
		}
		return policymanagement.ErrAlreadyExists
	}
	return mapManagedPolicyCreate(err, action)
}

func mapManagedPolicyCAS(err error, action string) error {
	if errors.Is(err, sql.ErrNoRows) {
		return policymanagement.ErrRevisionConflict
	}
	return mapManagedPolicyCreate(err, action)
}

func mapManagedPolicyDelete(err error, action string) error {
	var databaseError *pq.Error
	if errors.As(err, &databaseError) && databaseError.Code == "23503" {
		return policymanagement.ErrResourceInUse
	}
	if err != nil {
		return fmt.Errorf("%s: %w", action, err)
	}
	return nil
}

func stringsTrimmed(value string) string { return strings.TrimSpace(value) }

func (adapter *policyManagementRepositoryAdapter) CreateAccessPolicy(ctx context.Context, mutation policymanagement.CreateAccessPolicyMutation) (policymanagement.MutationResult, error) {
	return adapter.store.CreateManagedAccessPolicy(ctx, mutation)
}

func (adapter *policyManagementRepositoryAdapter) UpdateAccessPolicy(ctx context.Context, policy policymanagement.AccessPolicy, expected uint64, actor policymanagement.Actor) (policymanagement.MutationResult, error) {
	return adapter.store.UpdateManagedAccessPolicy(ctx, policy, expected, actor)
}

func (adapter *policyManagementRepositoryAdapter) DeleteAccessPolicy(ctx context.Context, namespaceID, policyID string, expected uint64, actor policymanagement.Actor) (policymanagement.MutationResult, error) {
	return adapter.store.DeleteManagedAccessPolicy(ctx, namespaceID, policyID, expected, actor)
}

func (adapter *policyManagementRepositoryAdapter) CreateRateLimitPolicy(ctx context.Context, mutation policymanagement.CreateRateLimitPolicyMutation) (policymanagement.MutationResult, error) {
	return adapter.store.CreateManagedRateLimitPolicy(ctx, mutation)
}

func (adapter *policyManagementRepositoryAdapter) UpdateRateLimitPolicy(ctx context.Context, policy policymanagement.RateLimitPolicy, expected uint64, actor policymanagement.Actor) (policymanagement.MutationResult, error) {
	return adapter.store.UpdateManagedRateLimitPolicy(ctx, policy, expected, actor)
}

func (adapter *policyManagementRepositoryAdapter) DeleteRateLimitPolicy(ctx context.Context, namespaceID, policyID string, expected uint64, actor policymanagement.Actor) (policymanagement.MutationResult, error) {
	return adapter.store.DeleteManagedRateLimitPolicy(ctx, namespaceID, policyID, expected, actor)
}

func (adapter *policyManagementRepositoryAdapter) GetAccessBinding(ctx context.Context, namespaceID, bindingID string) (policymanagement.AccessPolicyBinding, error) {
	return adapter.store.GetManagedAccessBinding(ctx, namespaceID, bindingID)
}

func (adapter *policyManagementRepositoryAdapter) ListAccessBindings(ctx context.Context, query policymanagement.BindingQuery) (policymanagement.RepositoryPage[policymanagement.AccessPolicyBinding], error) {
	return adapter.store.ListManagedAccessBindings(ctx, query)
}

func (adapter *policyManagementRepositoryAdapter) CreateAccessBinding(ctx context.Context, mutation policymanagement.CreateAccessBindingMutation) (policymanagement.MutationResult, error) {
	return adapter.store.CreateManagedAccessBinding(ctx, mutation)
}

func (adapter *policyManagementRepositoryAdapter) UpdateAccessBinding(ctx context.Context, namespaceID, bindingID string, expected uint64, status accesscontrol.BindingStatus, actor policymanagement.Actor) (policymanagement.MutationResult, error) {
	return adapter.store.UpdateManagedAccessBinding(ctx, namespaceID, bindingID, expected, status, actor)
}

func (adapter *policyManagementRepositoryAdapter) DeleteAccessBinding(ctx context.Context, namespaceID, bindingID string, expected uint64, actor policymanagement.Actor) (policymanagement.MutationResult, error) {
	return adapter.store.DeleteManagedAccessBinding(ctx, namespaceID, bindingID, expected, actor)
}

func (adapter *policyManagementRepositoryAdapter) GetRateBinding(ctx context.Context, namespaceID, bindingID string) (policymanagement.RateLimitBinding, error) {
	return adapter.store.GetManagedRateBinding(ctx, namespaceID, bindingID)
}

func (adapter *policyManagementRepositoryAdapter) ListRateBindings(ctx context.Context, query policymanagement.BindingQuery) (policymanagement.RepositoryPage[policymanagement.RateLimitBinding], error) {
	return adapter.store.ListManagedRateBindings(ctx, query)
}

func (adapter *policyManagementRepositoryAdapter) CreateRateBinding(ctx context.Context, mutation policymanagement.CreateRateBindingMutation) (policymanagement.MutationResult, error) {
	return adapter.store.CreateManagedRateBinding(ctx, mutation)
}

func (adapter *policyManagementRepositoryAdapter) CreateInlineRateBinding(ctx context.Context, mutation policymanagement.CreateInlineRateBindingMutation) (policymanagement.InlineRateBindingResult, error) {
	return adapter.store.CreateManagedInlineRateBinding(ctx, mutation)
}

func (adapter *policyManagementRepositoryAdapter) UpdateRateBinding(ctx context.Context, namespaceID, bindingID string, expected uint64, status accesscontrol.BindingStatus, actor policymanagement.Actor) (policymanagement.MutationResult, error) {
	return adapter.store.UpdateManagedRateBinding(ctx, namespaceID, bindingID, expected, status, actor)
}

func (adapter *policyManagementRepositoryAdapter) DeleteRateBinding(ctx context.Context, namespaceID, bindingID string, expected uint64, actor policymanagement.Actor) (policymanagement.MutationResult, error) {
	return adapter.store.DeleteManagedRateBinding(ctx, namespaceID, bindingID, expected, actor)
}
