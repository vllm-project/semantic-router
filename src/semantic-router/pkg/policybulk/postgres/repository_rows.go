// Package postgres implements durable policy bulk queue row handling.
package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"net/netip"
	"time"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policybulk"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

type queryRower interface {
	QueryRowContext(context.Context, string, ...any) *sql.Row
}

func getOperation(ctx context.Context, queryer queryRower, namespaceID, operationID string) (policybulk.Operation, error) {
	operation, err := scanOperation(queryer.QueryRowContext(ctx, getOperationQuery, namespaceID, operationID))
	if err != nil {
		return policybulk.Operation{}, mapReadError(err)
	}
	return operation, nil
}

func finalizeOperation(ctx context.Context, tx *sql.Tx, operationID string, now time.Time) (policybulk.OperationState, error) {
	var state string
	err := tx.QueryRowContext(ctx, `WITH counts AS (
  SELECT count(*) FILTER (WHERE state IN ('succeeded','failed','cancelled')) AS completed,
         count(*) FILTER (WHERE state='succeeded') AS succeeded,
         count(*) FILTER (WHERE state='failed') AS failed,
         count(*) FILTER (WHERE state='cancelled') AS cancelled,
         count(*) FILTER (WHERE state='running') AS running,
         count(*) FILTER (WHERE state='pending') AS pending,
         count(*) AS total,
         coalesce(jsonb_agg(jsonb_build_object('itemId',item_id,'code',error_code,'reason',error_reason)
           ORDER BY ordinal) FILTER (WHERE state='failed'),'[]'::jsonb) AS item_errors
  FROM policy_bulk_operation_items WHERE operation_id=$1
), next AS (
  SELECT counts.completed,counts.total,counts.item_errors AS next_item_errors,
    CASE
      WHEN o.cancelled_at IS NOT NULL AND counts.running+counts.pending=0 THEN 'cancelled'
      WHEN counts.running>0 THEN 'running'
      WHEN counts.pending>0 THEN 'pending'
      WHEN counts.succeeded=counts.total THEN 'succeeded'
      WHEN counts.failed=counts.total THEN 'failed'
      ELSE 'partially_succeeded'
    END AS state
  FROM counts CROSS JOIN management_operations o WHERE o.id=$1
)
UPDATE management_operations o SET state=next.state,progress_completed=next.completed,
    progress_total=next.total,item_errors=next.next_item_errors,updated_at=$2,
    completed_at=CASE WHEN next.state IN ('succeeded','partially_succeeded','failed','cancelled')
      THEN coalesce(o.completed_at,$2) ELSE NULL END
FROM next WHERE o.id=$1 RETURNING o.state`, operationID, now).Scan(&state)
	if err != nil {
		return "", fmt.Errorf("finalize policy bulk operation: %w", err)
	}
	return policybulk.OperationState(state), nil
}

func expireExhaustedClaims(ctx context.Context, tx *sql.Tx, now time.Time, maximumAttempts int) error {
	operationIDs, err := updateExpiredClaims(ctx, tx, cancelExpiredClaimsQuery, now)
	if err != nil {
		return fmt.Errorf("cancel expired policy bulk claims: %w", err)
	}
	exhausted, err := updateExpiredClaims(ctx, tx, exhaustedClaimsQuery, now, maximumAttempts)
	if err != nil {
		return fmt.Errorf("expire exhausted policy bulk claims: %w", err)
	}
	operationIDs = append(operationIDs, exhausted...)
	expired, err := updateExpiredClaims(ctx, tx, expireOperationItemsQuery, now)
	if err != nil {
		return fmt.Errorf("expire policy bulk operations: %w", err)
	}
	operationIDs = append(operationIDs, expired...)
	seen := make(map[string]struct{}, len(operationIDs))
	for _, operationID := range operationIDs {
		if _, found := seen[operationID]; found {
			continue
		}
		seen[operationID] = struct{}{}
		if _, err := finalizeOperation(ctx, tx, operationID, now); err != nil {
			return err
		}
	}
	return nil
}

func updateExpiredClaims(
	ctx context.Context,
	tx *sql.Tx,
	query string,
	arguments ...any,
) (_ []string, returnErr error) {
	rows, err := tx.QueryContext(ctx, query, arguments...)
	if err != nil {
		return nil, err
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	var operationIDs []string
	for rows.Next() {
		var operationID string
		if err := rows.Scan(&operationID); err != nil {
			return nil, err
		}
		operationIDs = append(operationIDs, operationID)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return operationIDs, nil
}

func operationCancelled(ctx context.Context, tx *sql.Tx, operationID string) (bool, error) {
	var cancelled bool
	if err := tx.QueryRowContext(ctx, `SELECT cancelled_at IS NOT NULL FROM management_operations
WHERE id=$1 FOR UPDATE`, operationID).Scan(&cancelled); err != nil {
		return false, mapReadError(err)
	}
	return cancelled, nil
}

func terminalState(state policybulk.OperationState) bool {
	return state == policybulk.OperationSucceeded || state == policybulk.OperationPartiallySucceeded ||
		state == policybulk.OperationFailed || state == policybulk.OperationCancelled
}

func requireChanged(result sql.Result) error {
	count, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("read policy bulk write result: %w", err)
	}
	if count != 1 {
		return policybulk.ErrLeaseLost
	}
	return nil
}

func mapReadError(err error) error {
	if errors.Is(err, sql.ErrNoRows) {
		return policybulk.ErrNotFound
	}
	return err
}

func mapWriteError(err error) error {
	if errors.Is(err, managementcommand.ErrConflict) {
		return policybulk.ErrConflict
	}
	var databaseError *pq.Error
	if errors.As(err, &databaseError) && databaseError.Code == "23505" {
		return policybulk.ErrConflict
	}
	return err
}

func validOperation(operation policybulk.Operation) bool {
	return operation.ID != "" && operation.NamespaceID != "" && operation.OriginPrincipalID != "" &&
		(operation.Kind == policybulk.AccessBindingOperationKind || operation.Kind == policybulk.RateBindingOperationKind) &&
		operation.Version == 1 && operation.State == policybulk.OperationPending && operation.Total > 0 &&
		operation.Total == uint64(len(operation.TargetIDs)) && operation.Total == uint64(len(operation.Targets)) &&
		!operation.CreatedAt.IsZero()
}

func matchesAccessTargets(targets []policybulk.OperationTarget, items []policybulk.AccessBindingItem) bool {
	if len(targets) != len(items) {
		return false
	}
	for index, target := range targets {
		item := items[index]
		if target.ItemID != item.ItemID || target.Kind != policybulk.ItemKindAccessBinding ||
			target.PolicyID != item.PolicyID || target.InlinePolicy || target.Subject != item.Subject || target.Mode != "" {
			return false
		}
	}
	return true
}

func matchesRateTargets(targets []policybulk.OperationTarget, items []policybulk.RateBindingItem) bool {
	if len(targets) != len(items) {
		return false
	}
	for index, target := range targets {
		item := items[index]
		if target.ItemID != item.ItemID || target.Kind != policybulk.ItemKindRateBinding ||
			target.PolicyID != item.PolicyID || target.InlinePolicy != (item.InlinePolicy != nil) ||
			target.Subject != item.Subject || target.Mode != item.Mode {
			return false
		}
	}
	return true
}

func nullableUUID(value string) any {
	if value == "" {
		return nil
	}
	return value
}

func nullableString(value string) any {
	if value == "" {
		return nil
	}
	return value
}

func nullableDuration(value policymanagement.ISODuration) any {
	if value.Duration() == 0 {
		return nil
	}
	return int64(value.Duration())
}

func insertRule(ctx context.Context, tx *sql.Tx, operationID, itemID string,
	ordinal int, rule policymanagement.RateLimitRule,
) error {
	_, err := tx.ExecContext(ctx, insertRateRuleQuery, operationID, itemID, ordinal,
		nullableUUID(rule.ID), rule.Metric, rule.Algorithm, nullableString(string(rule.Limit)),
		nullableDuration(rule.Window), nullableString(string(rule.CalendarPeriod)),
		nullableString(rule.Timezone), nullableString(string(rule.BucketCapacity)),
		nullableString(string(rule.RefillAmount)), nullableDuration(rule.RefillPeriod),
		nullableDuration(rule.GCRAEmissionInterval), rule.GCRABurstTolerance,
		rule.Accounting, rule.Enforcement)
	if err != nil {
		return fmt.Errorf("insert policy bulk rate rule: %w", err)
	}
	return nil
}

func claimItemID(claim policybulk.ItemClaim) string {
	if claim.Access != nil {
		return claim.Access.ItemID
	}
	if claim.Rate != nil {
		return claim.Rate.ItemID
	}
	return ""
}

type rowScanner interface{ Scan(...any) error }

func scanOperation(scanner rowScanner) (policybulk.Operation, error) {
	var operation policybulk.Operation
	var actorChain, targets, targetIDs, itemErrors []byte
	var desiredRevision, publicationRevision, appliedRevision sql.NullInt64
	var completedAt sql.NullTime
	if err := scanner.Scan(&operation.ID, &operation.NamespaceID, &operation.Kind,
		&operation.OriginPrincipalID, &actorChain, &operation.Version, &operation.State, &operation.Completed,
		&operation.Total, &targets, &targetIDs, &desiredRevision, &publicationRevision, &appliedRevision,
		&itemErrors, &operation.CreatedAt, &operation.UpdatedAt, &completedAt); err != nil {
		return policybulk.Operation{}, err
	}
	if err := json.Unmarshal(actorChain, &operation.ActorChain); err != nil {
		return policybulk.Operation{}, fmt.Errorf("decode policy bulk actor chain: %w", err)
	}
	if err := json.Unmarshal(targetIDs, &operation.TargetIDs); err != nil {
		return policybulk.Operation{}, fmt.Errorf("decode policy bulk targets: %w", err)
	}
	if err := json.Unmarshal(targets, &operation.Targets); err != nil {
		return policybulk.Operation{}, fmt.Errorf("decode policy bulk target scope: %w", err)
	}
	if err := json.Unmarshal(itemErrors, &operation.ItemErrors); err != nil {
		return policybulk.Operation{}, fmt.Errorf("decode policy bulk item errors: %w", err)
	}
	if operation.Version == 0 || !operation.State.Valid() ||
		(operation.Kind != policybulk.AccessBindingOperationKind && operation.Kind != policybulk.RateBindingOperationKind) ||
		(desiredRevision.Valid && desiredRevision.Int64 < 0) ||
		(publicationRevision.Valid && publicationRevision.Int64 < 0) ||
		(appliedRevision.Valid && appliedRevision.Int64 < 0) ||
		terminalState(operation.State) != completedAt.Valid {
		return policybulk.Operation{}, errors.New("stored policy bulk operation is invalid")
	}
	operation.Failed = uint64(len(operation.ItemErrors))
	if desiredRevision.Valid {
		operation.DesiredRevision = uint64(desiredRevision.Int64)
	}
	if publicationRevision.Valid {
		operation.PublicationRevision = uint64(publicationRevision.Int64)
	}
	if appliedRevision.Valid {
		operation.AppliedRevision = uint64(appliedRevision.Int64)
	}
	operation.CreatedAt = operation.CreatedAt.UTC()
	operation.UpdatedAt = operation.UpdatedAt.UTC()
	if completedAt.Valid {
		value := completedAt.Time.UTC()
		operation.CompletedAt = &value
	}
	return operation, nil
}

func scanClaim(scanner rowScanner) (policybulk.ItemClaim, error) {
	var claim policybulk.ItemClaim
	var actorChain []byte
	var sourceIP, accessPolicyID, ratePolicyID, inlineName, inlineDescription, bindingMode sql.NullString
	var itemID, subjectID string
	var subjectKind accesscontrol.SubjectKind
	if err := scanner.Scan(&claim.OperationID, &claim.NamespaceID, &claim.OriginPrincipalID,
		&actorChain, &claim.Context.RequestID, &sourceIP, &itemID, &claim.ItemKind,
		&accessPolicyID, &ratePolicyID, &inlineName, &inlineDescription,
		&subjectID, &subjectKind, &bindingMode, &claim.Attempt); err != nil {
		return policybulk.ItemClaim{}, err
	}
	if err := json.Unmarshal(actorChain, &claim.ActorChain); err != nil {
		return policybulk.ItemClaim{}, fmt.Errorf("decode policy bulk claim actor chain: %w", err)
	}
	if sourceIP.Valid {
		parsed, err := netip.ParseAddr(sourceIP.String)
		if err != nil {
			return policybulk.ItemClaim{}, fmt.Errorf("decode policy bulk source IP: %w", err)
		}
		claim.Context.SourceIP = parsed.Unmap()
	}
	subject := policymanagement.Subject{Type: subjectKind, ID: subjectID}
	switch claim.ItemKind {
	case policybulk.ItemKindAccessBinding:
		if !accessPolicyID.Valid || ratePolicyID.Valid || inlineName.Valid || bindingMode.Valid {
			return policybulk.ItemClaim{}, errors.New("stored AccessPolicy bulk item is invalid")
		}
		claim.Access = &policybulk.AccessBindingItem{
			ItemID: itemID, PolicyID: accessPolicyID.String, Subject: subject,
		}
	case policybulk.ItemKindRateBinding:
		if !bindingMode.Valid || accessPolicyID.Valid || ratePolicyID.Valid == inlineName.Valid {
			return policybulk.ItemClaim{}, errors.New("stored RateLimit bulk item is invalid")
		}
		claim.Rate = &policybulk.RateBindingItem{
			ItemID: itemID, PolicyID: ratePolicyID.String, Subject: subject,
			Mode: accesscontrol.RateBindingMode(bindingMode.String),
		}
		if inlineName.Valid {
			if !inlineDescription.Valid {
				return policybulk.ItemClaim{}, errors.New("stored inline RateLimit policy is invalid")
			}
			claim.Rate.InlinePolicy = &policybulk.InlineRateLimitPolicy{
				Name: inlineName.String, Description: inlineDescription.String,
			}
		}
	default:
		return policybulk.ItemClaim{}, errors.New("stored policy bulk item kind is invalid")
	}
	return claim, nil
}

func loadRateRules(
	ctx context.Context,
	tx *sql.Tx,
	operationID string,
	itemID string,
) (_ []policymanagement.RateLimitRule, returnErr error) {
	rows, err := tx.QueryContext(ctx, loadRateRulesQuery, operationID, itemID)
	if err != nil {
		return nil, fmt.Errorf("load policy bulk rate rules: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	var rules []policymanagement.RateLimitRule
	for rows.Next() {
		var rule policymanagement.RateLimitRule
		var ruleID, limit, period, timezone, capacity, refill sql.NullString
		var window, refillPeriod, emission, burst sql.NullInt64
		if err := rows.Scan(&ruleID, &rule.Metric, &rule.Algorithm, &limit, &window,
			&period, &timezone, &capacity, &refill, &refillPeriod, &emission,
			&burst, &rule.Accounting, &rule.Enforcement, &rule.Ordinal); err != nil {
			return nil, fmt.Errorf("scan policy bulk rate rule: %w", err)
		}
		rule.ID, rule.Limit = ruleID.String, accesscontrol.QuotaValue(limit.String)
		rule.Window = policymanagement.ISODuration(window.Int64)
		rule.CalendarPeriod, rule.Timezone = accesscontrol.CalendarPeriod(period.String), timezone.String
		rule.BucketCapacity, rule.RefillAmount = accesscontrol.QuotaValue(capacity.String), accesscontrol.QuotaValue(refill.String)
		rule.RefillPeriod = policymanagement.ISODuration(refillPeriod.Int64)
		rule.GCRAEmissionInterval = policymanagement.ISODuration(emission.Int64)
		if burst.Valid {
			value := burst.Int64
			rule.GCRABurstTolerance = &value
		}
		rules = append(rules, rule)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate policy bulk rate rules: %w", err)
	}
	return rules, nil
}

var _ policybulk.Repository = (*Repository)(nil)
