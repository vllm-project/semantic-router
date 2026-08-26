package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

// ReissueIssuerSessionInTransaction issues another short-lived token for the
// durable Management session that already represents the same upstream login.
// A fresh issuer assertion is still required by the caller. The durable token
// ID remains stable so independently cached tokens from Dashboard replicas can
// overlap until their own bounded expiry instead of invalidating one another.
func (s *Store) ReissueIssuerSessionInTransaction(
	ctx context.Context,
	tx *sql.Tx,
	draft managementauth.SessionDraft,
	issue managementauth.PreparedSessionIssuer,
) (managementauth.LiveSession, managementauth.IssuedToken, bool, error) {
	if s == nil || s.db == nil || tx == nil || issue == nil {
		return managementauth.LiveSession{}, managementauth.IssuedToken{}, false,
			managementauth.ErrAuthenticationUnavailable
	}
	if draft.IssuerSessionID == nil {
		return managementauth.LiveSession{}, managementauth.IssuedToken{}, false, nil
	}
	if !validIssuerReissueDraft(draft) {
		return managementauth.LiveSession{}, managementauth.IssuedToken{}, false,
			managementauth.ErrAuthenticationDenied
	}
	var principalStatus string
	if err := tx.QueryRowContext(ctx, lockPrincipalQuery, draft.PrincipalID).Scan(&principalStatus); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return managementauth.LiveSession{}, managementauth.IssuedToken{}, false,
				managementauth.ErrAuthenticationDenied
		}
		return managementauth.LiveSession{}, managementauth.IssuedToken{}, false,
			fmt.Errorf("lock issuer Management principal: %w", err)
	}
	if principalStatus != string(managementauth.ResourceActive) {
		return managementauth.LiveSession{}, managementauth.IssuedToken{}, false,
			managementauth.ErrAuthenticationDenied
	}

	var sessionID string
	var now time.Time
	assurance, _, err := encodeAssurance(managementauth.Session{
		EvidenceKind: draft.EvidenceKind,
		Human:        draft.Human,
	})
	if err != nil {
		return managementauth.LiveSession{}, managementauth.IssuedToken{}, false,
			managementauth.ErrAuthenticationDenied
	}
	err = tx.QueryRowContext(
		ctx,
		lockIssuerSessionQuery,
		draft.PrincipalID,
		draft.AuthSourceID,
		*draft.IssuerSessionID,
		draft.Audience,
		draft.AuthenticatedAt.UTC(),
		assurance,
	).Scan(&sessionID, &now)
	if errors.Is(err, sql.ErrNoRows) {
		return managementauth.LiveSession{}, managementauth.IssuedToken{}, false, nil
	}
	if err != nil {
		return managementauth.LiveSession{}, managementauth.IssuedToken{}, false,
			fmt.Errorf("lock issuer Management session: %w", err)
	}
	now = now.UTC()

	current, err := getWith(ctx, tx, sessionID)
	if err != nil {
		return managementauth.LiveSession{}, managementauth.IssuedToken{}, false, err
	}
	if err := current.ValidateAt(now); err != nil || !issuerReissueDraftMatches(draft, current) {
		return managementauth.LiveSession{}, managementauth.IssuedToken{}, false,
			managementauth.ErrAuthenticationDenied
	}

	issued, err := issue(ctx, current, now)
	if err != nil || !validReissuedToken(issued, current) {
		if err == nil {
			err = managementauth.ErrAuthenticationUnavailable
		}
		return managementauth.LiveSession{}, managementauth.IssuedToken{}, false, err
	}
	return current, issued, true, nil
}

func validIssuerReissueDraft(draft managementauth.SessionDraft) bool {
	return draft.PrincipalID != "" && draft.TokenID != "" &&
		draft.AuthSourceKind == managementauth.AuthSourceIssuer && draft.AuthSourceID != "" &&
		draft.EvidenceKind == managementauth.EvidenceHuman && draft.Human != nil && draft.Workload == nil &&
		!draft.AuthenticatedAt.IsZero() && !draft.EvidenceExpiresAt.IsZero() &&
		draft.IssuerSessionID != nil && *draft.IssuerSessionID != ""
}

func issuerReissueDraftMatches(draft managementauth.SessionDraft, live managementauth.LiveSession) bool {
	return live.PrincipalID == draft.PrincipalID &&
		live.Audience == draft.Audience && live.AuthSourceKind == draft.AuthSourceKind &&
		live.AuthSourceID == draft.AuthSourceID && live.EvidenceKind == draft.EvidenceKind &&
		live.IssuerSessionID != nil && draft.IssuerSessionID != nil &&
		*live.IssuerSessionID == *draft.IssuerSessionID &&
		live.AuthenticatedAt.Equal(draft.AuthenticatedAt.UTC()) &&
		!draft.EvidenceExpiresAt.UTC().Before(live.ExpiresAt) &&
		live.Human != nil && draft.Human != nil &&
		live.Human.AuthenticationTime == draft.Human.AuthenticationTime &&
		live.Human.AAL == draft.Human.AAL && slices.Equal(live.Human.AMR, draft.Human.AMR)
}

func validReissuedToken(issued managementauth.IssuedToken, session managementauth.LiveSession) bool {
	return issued.AccessToken != "" && issued.TokenType == "Bearer" && issued.ExpiresIn > 0 &&
		issued.ManagementSessionID == session.ID
}
