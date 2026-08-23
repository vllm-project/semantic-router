package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	managementcommandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotareconciliation"
)

const unknownUsageWaiveAction = "unknown_usage_fence.waive"

func (s *Store) ReadyQuotaReconciliation(ctx context.Context, codec *managementcommand.Codec) error {
	if s == nil || s.db == nil || codec == nil {
		return quotareconciliation.ErrUnavailable
	}
	if err := s.db.PingContext(ctx); err != nil {
		return err
	}
	if err := managementcommandpostgres.ValidateReferencedHMACVersions(ctx, s.db, codec); err != nil {
		return err
	}
	rows, err := s.db.QueryContext(ctx, `SELECT reconciliation_id,plan_digest,phase,runtime_stream_id
FROM unknown_usage_reconciliation_plans LIMIT 0`)
	if err != nil {
		return err
	}
	return rows.Close()
}

func (s *Store) AuthorizeWaive(
	ctx context.Context,
	namespaceID string,
	session managementauth.LiveSession,
	now time.Time,
) error {
	if s == nil || s.db == nil || now.IsZero() || session.ValidateAt(now) != nil {
		return quotareconciliation.ErrWaiveDenied
	}
	var payload []byte
	var seedVersion, revision int64
	var updatedAt time.Time
	err := s.db.QueryRowContext(ctx, `SELECT action_requirements,seed_version,revision,updated_at
FROM management_security_policies WHERE namespace_id=$1`, namespaceID).
		Scan(&payload, &seedVersion, &revision, &updatedAt)
	if err != nil {
		return quotareconciliation.ErrUnavailable
	}
	requirements := map[string]managementauth.ActionRequirement{}
	if json.Unmarshal(payload, &requirements) != nil || seedVersion <= 0 || revision <= 0 {
		return quotareconciliation.ErrUnavailable
	}
	policy := managementauth.SessionPolicy{
		AccessTokenTTL: time.Second, SessionTTL: time.Second, MaxActiveSessions: 1,
		ActionRequirements: requirements, SeedVersion: managementauth.SupportedSessionPolicySeedVersion,
		Revision: uint64(revision), UpdatedAt: updatedAt.UTC(),
	}
	if policy.Validate() != nil {
		return quotareconciliation.ErrUnavailable
	}
	requirement, found := policy.ActionRequirements[unknownUsageWaiveAction]
	if !found || len(requirement.AnyOf) == 0 || !requirement.Allows(session, now) {
		return quotareconciliation.ErrWaiveDenied
	}
	return nil
}

func appendQuotaReconciliationAudit(
	ctx context.Context,
	tx *sql.Tx,
	request quotareconciliation.ReconcileRequest,
	fenceID string,
	revision uint64,
	action, reason string,
	details map[string]string,
) error {
	if revision == 0 {
		return errors.New("unknown-usage audit revision is invalid")
	}
	principal := accesscontrol.ManagementPrincipalID(request.Actor.PrincipalID)
	actors := make([]accesscontrol.ManagementPrincipalID, len(request.Actor.ActorChain))
	for index, actor := range request.Actor.ActorChain {
		actors[index] = accesscontrol.ManagementPrincipalID(actor)
	}
	return appendObservedAuditEvent(ctx, tx, accesscontrol.NamespaceID(request.NamespaceID),
		"unknown_usage_fence", fenceID, accesscontrol.Revision(revision), MutationMeta{
			ActorPrincipalID: &principal, ActorChain: actors, RequestID: request.Actor.RequestID,
			SourceIP: request.Actor.SourceIP, Action: action, Reason: reason, Details: AuditDetails(details),
		})
}
