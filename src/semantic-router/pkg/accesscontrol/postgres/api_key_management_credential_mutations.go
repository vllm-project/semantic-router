package postgres

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apikeymanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
)

func rotateCredentialTx(
	ctx context.Context,
	tx *sql.Tx,
	mutation apikeymanagement.RotateMutation,
	meta MutationMeta,
) (apikeymanagement.MutationResult, error) {
	replay, found, err := lockAPIKeySecretCommand(ctx, tx, mutation.Command)
	if err != nil {
		return apikeymanagement.MutationResult{}, err
	}
	if found {
		return replayedCredentialRotation(ctx, tx, mutation.NamespaceID, replay)
	}
	updated, err := scanAPIKey(tx.QueryRowContext(ctx, managementAdvanceAPIKeyRevisionQuery,
		mutation.NamespaceID, mutation.KeyID, mutation.ExpectedRevision, mutation.Credential.CreatedAt))
	if err != nil {
		return apikeymanagement.MutationResult{}, mapAPIKeyCAS(err, "rotate API-key credential")
	}
	if err := transitionPreviousCredential(ctx, tx, mutation); err != nil {
		return apikeymanagement.MutationResult{}, err
	}
	if err := insertCredential(ctx, tx, accesscontrol.NamespaceID(mutation.NamespaceID), mutation.Credential); err != nil {
		return apikeymanagement.MutationResult{}, mapAPIKeyCreateError(err, "insert rotated credential")
	}
	if err := recordCredentialRotation(ctx, tx, mutation, updated, meta); err != nil {
		return apikeymanagement.MutationResult{}, err
	}
	return apikeymanagement.MutationResult{Key: updated, HTTPStatus: 200}, nil
}

func replayedCredentialRotation(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	replay apikeymanagement.StoredSecret,
) (apikeymanagement.MutationResult, error) {
	key, err := scanAPIKey(tx.QueryRowContext(ctx, getAPIKeyQuery, namespaceID, replay.Result.ResourceID))
	if err != nil {
		return apikeymanagement.MutationResult{}, mapAPIKeyReadError(err, "read replayed API key")
	}
	return apikeymanagement.MutationResult{
		Key: key, HTTPStatus: replay.Result.ResponseStatus, Replayed: true, Stored: &replay,
	}, nil
}

func transitionPreviousCredential(
	ctx context.Context,
	tx *sql.Tx,
	mutation apikeymanagement.RotateMutation,
) error {
	var result sql.Result
	var err error
	if mutation.RetireAt == nil {
		result, err = tx.ExecContext(ctx, managementRevokePreviousCredentialQuery,
			mutation.NamespaceID, mutation.KeyID, mutation.PreviousCredentialID, mutation.Credential.CreatedAt)
	} else {
		result, err = tx.ExecContext(ctx, retireCredentialQuery,
			mutation.NamespaceID, mutation.KeyID, mutation.PreviousCredentialID, *mutation.RetireAt)
	}
	if err != nil {
		return fmt.Errorf("retire prior credential: %w", err)
	}
	return requireOneRow(result, apikeymanagement.ErrCredentialUnavailable)
}

func recordCredentialRotation(
	ctx context.Context,
	tx *sql.Tx,
	mutation apikeymanagement.RotateMutation,
	updated accesscontrol.APIKey,
	meta MutationMeta,
) error {
	if _, err := appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(mutation.NamespaceID), outboxMutation{
		AggregateType: "api_key", AggregateID: mutation.KeyID, AggregateRevision: updated.Revision,
		Operation: outboxCredentialRotated, References: map[string]string{
			"credentialId":        string(mutation.Credential.ID),
			"retiredCredentialId": mutation.PreviousCredentialID,
		},
	}, meta); err != nil {
		return err
	}
	return commandpostgres.CompleteSecretResource(ctx, tx, mutation.Command,
		managementcommand.ResourceResult{
			ResourceType: "api_key", ResourceID: mutation.KeyID,
			ResourceRevision: uint64(updated.Revision), ResponseStatus: 200,
		},
		managementcommand.SecretResponse{
			Ciphertext: mutation.Response.Ciphertext, Nonce: mutation.Response.Nonce,
			KEKVersion: mutation.Response.KeyVersion, ExpiresAt: mutation.ResponseExpiresAt,
		})
}

type credentialRevocation struct {
	namespaceID      string
	keyID            string
	credentialID     string
	expected         uint64
	expectedRevision int64
	meta             MutationMeta
}

func revokeCredentialTx(
	ctx context.Context,
	tx *sql.Tx,
	revocation credentialRevocation,
) (apikeymanagement.MutationResult, error) {
	if err := validateCredentialRevocation(ctx, tx, revocation); err != nil {
		return apikeymanagement.MutationResult{}, err
	}
	updated, err := advanceAPIKeyRevision(
		ctx, tx, accesscontrol.NamespaceID(revocation.namespaceID), accesscontrol.APIKeyID(revocation.keyID),
		revocation.expectedRevision,
	)
	if err != nil {
		return apikeymanagement.MutationResult{}, mapAPIKeyCAS(err, "revoke API-key credential")
	}
	result, err := tx.ExecContext(
		ctx, revokeCredentialQuery, revocation.namespaceID, revocation.keyID, revocation.credentialID,
	)
	if err != nil {
		return apikeymanagement.MutationResult{}, err
	}
	if err := requireOneRow(result, apikeymanagement.ErrCredentialUnavailable); err != nil {
		return apikeymanagement.MutationResult{}, err
	}
	if err := recordCredentialRevocation(ctx, tx, revocation, updated); err != nil {
		return apikeymanagement.MutationResult{}, err
	}
	return apikeymanagement.MutationResult{Key: updated, HTTPStatus: 204}, nil
}

func validateCredentialRevocation(ctx context.Context, tx *sql.Tx, revocation credentialRevocation) error {
	key, err := scanAPIKey(tx.QueryRowContext(
		ctx, getAPIKeyQuery+" FOR UPDATE", revocation.namespaceID, revocation.keyID,
	))
	if err != nil {
		return mapAPIKeyReadError(err, "lock API key")
	}
	if uint64(key.Revision) != revocation.expected {
		return apikeymanagement.ErrRevisionConflict
	}
	if key.Status != accesscontrol.APIKeyStatusActive {
		return nil
	}
	var count int
	if err := tx.QueryRowContext(
		ctx, managementCountUsableCredentialsQuery, revocation.namespaceID, revocation.keyID,
	).Scan(&count); err != nil {
		return err
	}
	if count <= 1 {
		return apikeymanagement.ErrLastActiveCredential
	}
	return nil
}

func recordCredentialRevocation(
	ctx context.Context,
	tx *sql.Tx,
	revocation credentialRevocation,
	updated accesscontrol.APIKey,
) error {
	_, err := appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(revocation.namespaceID), outboxMutation{
		AggregateType: "api_key", AggregateID: revocation.keyID, AggregateRevision: updated.Revision,
		Operation: outboxCredentialRevoked,
		References: map[string]string{"credentialId": revocation.credentialID},
	}, revocation.meta)
	return err
}
