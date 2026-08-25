package postgres

import (
	"context"
	"database/sql"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apikeymanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

func createAPIKeyInTransaction(
	ctx context.Context,
	tx *sql.Tx,
	mutation apikeymanagement.CreateMutation,
	meta MutationMeta,
) (apikeymanagement.MutationResult, error) {
	if replay, found, err := replayAPIKeyCreation(ctx, tx, mutation); err != nil || found {
		return replay, err
	}
	created, err := insertManagedAPIKey(ctx, tx, mutation)
	if err != nil {
		return apikeymanagement.MutationResult{}, err
	}
	mutations := []compoundMutation{{Mutation: outboxMutation{
		AggregateType: "api_key", AggregateID: string(mutation.Key.ID), AggregateRevision: created.Revision,
		Operation: outboxCreated, References: map[string]string{"credentialId": string(mutation.Credential.ID)},
	}, Meta: meta}}
	policyActor := apiKeyPolicyActor(mutation.Actor)
	mutations, err = appendAccessBindingMutations(ctx, tx, mutation, policyActor, mutations)
	if err != nil {
		return apikeymanagement.MutationResult{}, err
	}
	mutations, err = appendRateLimitOverrideMutations(ctx, tx, mutation, policyActor, mutations)
	if err != nil {
		return apikeymanagement.MutationResult{}, err
	}
	if err := appendAPIKeyCreateMutationRecords(ctx, tx, mutation.Key.NamespaceID, mutations); err != nil {
		return apikeymanagement.MutationResult{}, err
	}
	if err := completeAPIKeyCreation(ctx, tx, mutation, created); err != nil {
		return apikeymanagement.MutationResult{}, err
	}
	return apikeymanagement.MutationResult{Key: created, HTTPStatus: 201}, nil
}

func replayAPIKeyCreation(
	ctx context.Context,
	tx *sql.Tx,
	mutation apikeymanagement.CreateMutation,
) (apikeymanagement.MutationResult, bool, error) {
	replay, found, err := lockAPIKeySecretCommand(ctx, tx, mutation.Command)
	if err != nil || !found {
		return apikeymanagement.MutationResult{}, found, err
	}
	key, err := scanAPIKey(tx.QueryRowContext(
		ctx, getAPIKeyQuery, mutation.Key.NamespaceID, replay.Result.ResourceID,
	))
	if err != nil {
		return apikeymanagement.MutationResult{}, true, mapAPIKeyReadError(err, "read replayed API key")
	}
	return apikeymanagement.MutationResult{
		Key: key, HTTPStatus: replay.Result.ResponseStatus, Replayed: true, Stored: &replay,
	}, true, nil
}

func insertManagedAPIKey(
	ctx context.Context,
	tx *sql.Tx,
	mutation apikeymanagement.CreateMutation,
) (accesscontrol.APIKey, error) {
	if err := validateAPIKeyRelationshipsTx(ctx, tx, mutation.Key); err != nil {
		return accesscontrol.APIKey{}, err
	}
	if _, err := tx.ExecContext(
		ctx, insertSubjectQuery, mutation.Key.NamespaceID, mutation.Key.ID,
		accesscontrol.SubjectKindAPIKey, mutation.Key.CreatedAt,
	); err != nil {
		return accesscontrol.APIKey{}, mapAPIKeyCreateError(err, "insert API-key subject")
	}
	ownerUser, ownerTeam := apiKeyOwnerValues(mutation.Key.Owner)
	created, err := scanAPIKey(tx.QueryRowContext(ctx, insertAPIKeyQuery,
		mutation.Key.ID, mutation.Key.NamespaceID, mutation.Key.Name, ownerUser, ownerTeam,
		nullableTeamID(mutation.Key.ContextTeamID), mutation.Key.Status, mutation.Key.ExpiresAt,
		mutation.Key.PolicyEpoch, mutation.Key.DelegationEpoch, mutation.Key.Revision,
		mutation.Key.CreatedAt, mutation.Key.UpdatedAt,
	))
	if err != nil {
		return accesscontrol.APIKey{}, mapAPIKeyCreateError(err, "insert API key")
	}
	if err := insertCredential(ctx, tx, mutation.Key.NamespaceID, mutation.Credential); err != nil {
		return accesscontrol.APIKey{}, mapAPIKeyCreateError(err, "insert API-key credential")
	}
	return created, nil
}

func appendAccessBindingMutations(
	ctx context.Context,
	tx *sql.Tx,
	mutation apikeymanagement.CreateMutation,
	actor policymanagement.Actor,
	mutations []compoundMutation,
) ([]compoundMutation, error) {
	for _, binding := range mutation.AccessBindings {
		materialized, err := materializeManagedAccessBinding(ctx, tx, binding)
		if err != nil {
			return nil, err
		}
		bindingMeta, err := managedPolicyMutationMeta(
			actor, "access_policy_binding.create", "Bind AccessPolicy to API key.",
			map[string]string{"apiKeyId": string(mutation.Key.ID)},
		)
		if err != nil {
			return nil, err
		}
		mutations = append(mutations, compoundMutation{Mutation: outboxMutation{
			AggregateType: "access_policy_binding", AggregateID: materialized.ID,
			AggregateRevision: accesscontrol.Revision(materialized.Revision), Operation: outboxCreated,
			References: managedBindingReferences(materialized.PolicyID, materialized.Subject),
		}, Meta: bindingMeta})
	}
	return mutations, nil
}

func appendRateLimitOverrideMutations(
	ctx context.Context,
	tx *sql.Tx,
	mutation apikeymanagement.CreateMutation,
	actor policymanagement.Actor,
	mutations []compoundMutation,
) ([]compoundMutation, error) {
	if mutation.RateLimitOverride == nil {
		return mutations, nil
	}
	materialized, err := materializeManagementAPIKeyRateLimitOverride(ctx, tx, managedRateLimitOverride{
		PolicyID: mutation.RateLimitOverride.PolicyID, InlinePolicy: mutation.RateLimitOverride.InlinePolicy,
		Binding: mutation.RateLimitOverride.Binding,
	})
	if err != nil {
		return nil, err
	}
	if materialized.Created {
		policyMeta, policyMetaErr := managedPolicyMutationMeta(
			actor, "rate_limit_policy.create", "Create inline RateLimitPolicy for API key.",
			map[string]string{"apiKeyId": string(mutation.Key.ID)},
		)
		if policyMetaErr != nil {
			return nil, policyMetaErr
		}
		mutations = append(mutations, compoundMutation{Mutation: outboxMutation{
			AggregateType: "rate_limit_policy", AggregateID: materialized.Policy.ID,
			AggregateRevision: accesscontrol.Revision(materialized.Policy.Revision), Operation: outboxCreated,
		}, Meta: policyMeta})
	}
	bindingMeta, err := managedPolicyMutationMeta(
		actor, "rate_limit_binding.create", "Bind RateLimitPolicy to API key.",
		map[string]string{"apiKeyId": string(mutation.Key.ID)},
	)
	if err != nil {
		return nil, err
	}
	return append(mutations, compoundMutation{Mutation: outboxMutation{
		AggregateType: "rate_limit_binding", AggregateID: materialized.Binding.ID,
		AggregateRevision: accesscontrol.Revision(materialized.Binding.Revision), Operation: outboxCreated,
		References: managedRateBindingReferences(materialized.Binding),
	}, Meta: bindingMeta}), nil
}

func completeAPIKeyCreation(
	ctx context.Context,
	tx *sql.Tx,
	mutation apikeymanagement.CreateMutation,
	created accesscontrol.APIKey,
) error {
	return commandpostgres.CompleteSecretResource(
		ctx, tx, mutation.Command,
		managementcommand.ResourceResult{
			ResourceType: "api_key", ResourceID: string(created.ID),
			ResourceRevision: uint64(created.Revision), ResponseStatus: 201,
		},
		managementcommand.SecretResponse{
			Ciphertext: mutation.Response.Ciphertext, Nonce: mutation.Response.Nonce,
			KEKVersion: mutation.Response.KeyVersion, ExpiresAt: mutation.ResponseExpiresAt,
		},
	)
}
