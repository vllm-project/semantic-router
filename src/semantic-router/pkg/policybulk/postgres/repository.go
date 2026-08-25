// Package postgres implements the durable, typed policy bulk queue.
package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policybulk"
)

const (
	insertOperationQuery = `INSERT INTO management_operations
  (id,namespace_id,kind,origin_principal_id,actor_chain,request_digest,state,
   progress_completed,progress_total,target_scope,target_ids,item_errors,created_at,updated_at)
VALUES ($1,$2,$3,$4,$5,$6,'pending',0,$7,$8,$9,'[]'::jsonb,$10,$10)`
	insertOperationContextQuery = `INSERT INTO policy_bulk_operation_contexts
  (operation_id,request_id,source_ip,expires_at) VALUES ($1,$2,$3,$4)`
	insertAccessItemQuery = `INSERT INTO policy_bulk_operation_items
  (operation_id,namespace_id,item_id,ordinal,item_kind,access_policy_id,
   subject_id,subject_kind,state,available_at,created_at,updated_at)
VALUES ($1,$2,$3,$4,'access_policy_binding',$5,$6,$7,'pending',$8,$8,$8)`
	insertRateItemQuery = `INSERT INTO policy_bulk_operation_items
  (operation_id,namespace_id,item_id,ordinal,item_kind,rate_policy_id,
   inline_policy_name,inline_policy_description,subject_id,subject_kind,binding_mode,
   state,available_at,created_at,updated_at)
VALUES ($1,$2,$3,$4,'rate_limit_binding',$5,$6,$7,$8,$9,$10,'pending',$11,$11,$11)`
	insertRateRuleQuery = `INSERT INTO policy_bulk_rate_rules
  (operation_id,item_id,ordinal,rule_id,metric,algorithm,limit_value,window_nanoseconds,
   calendar_period,timezone,bucket_capacity,refill_amount,refill_period_nanoseconds,
   gcra_emission_interval_nanoseconds,gcra_burst_tolerance,accounting,enforcement)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)`

	operationColumns = `o.id,o.namespace_id,o.kind,o.origin_principal_id,o.actor_chain,o.version,o.state,
	       o.progress_completed,o.progress_total,o.target_scope,o.target_ids,
	       o.desired_revision,o.publication_revision,o.applied_revision,
	       o.item_errors,o.created_at,o.updated_at,o.completed_at`
	getOperationQuery = `SELECT ` + operationColumns + ` FROM management_operations o
WHERE o.namespace_id=$1 AND o.id=$2
  AND o.kind IN ('access_policy_bindings.bulk_apply','rate_limit_bindings.bulk_apply')`

	claimItemQuery = `SELECT i.operation_id,i.namespace_id,o.origin_principal_id,o.actor_chain,
	       c.request_id,host(c.source_ip),i.item_id,i.item_kind,i.access_policy_id,
       i.rate_policy_id,i.inline_policy_name,i.inline_policy_description,
       i.subject_id,i.subject_kind,i.binding_mode,i.attempt_count
FROM policy_bulk_operation_items i
JOIN management_operations o ON o.id=i.operation_id
JOIN policy_bulk_operation_contexts c ON c.operation_id=i.operation_id
WHERE o.cancelled_at IS NULL AND o.state IN ('pending','running')
  AND c.expires_at > $1
  AND i.attempt_count < $2
  AND ((i.state='pending' AND i.available_at <= $1)
    OR (i.state='running' AND i.lease_expires_at <= $1))
ORDER BY i.available_at,i.operation_id,i.ordinal
FOR UPDATE OF i SKIP LOCKED LIMIT 1`
	claimUpdateQuery = `UPDATE policy_bulk_operation_items
SET state='running',attempt_count=attempt_count+1,lease_owner=$3,lease_token=$4,
    lease_expires_at=$5,updated_at=$2
WHERE operation_id=$1 AND item_id=$6`
	markOperationRunningQuery = `UPDATE management_operations
SET state='running',updated_at=$2 WHERE id=$1 AND state='pending'`
	loadRateRulesQuery = `SELECT rule_id,metric,algorithm,limit_value::text,window_nanoseconds,
       calendar_period,timezone,bucket_capacity::text,refill_amount::text,
       refill_period_nanoseconds,gcra_emission_interval_nanoseconds,
       gcra_burst_tolerance,accounting,enforcement,ordinal
FROM policy_bulk_rate_rules WHERE operation_id=$1 AND item_id=$2 ORDER BY ordinal`

	completeItemQuery = `UPDATE policy_bulk_operation_items
SET state='succeeded',result_binding_id=$5,result_policy_id=$6,
    lease_owner=NULL,lease_token=NULL,lease_expires_at=NULL,
    error_code=NULL,error_reason=NULL,finished_at=$4,updated_at=$4
WHERE operation_id=$1 AND item_id=$2 AND state='running'
  AND lease_owner=$3 AND lease_token=$7`
	lockOperationCancelQuery = `SELECT state,cancelled_at IS NOT NULL,version FROM management_operations
WHERE namespace_id=$1 AND id=$2
  AND kind IN ('access_policy_bindings.bulk_apply','rate_limit_bindings.bulk_apply')
FOR UPDATE`
	cancelOperationQuery = `UPDATE management_operations SET cancelled_at=$3,updated_at=$3
WHERE namespace_id=$1 AND id=$2 AND version=$4`
	cancelPendingItemsQuery = `UPDATE policy_bulk_operation_items
SET state='cancelled',finished_at=$2,updated_at=$2
WHERE operation_id=$1 AND state='pending'`
	exhaustedClaimsQuery = `UPDATE policy_bulk_operation_items i
SET state='failed',lease_owner=NULL,lease_token=NULL,lease_expires_at=NULL,
    error_code='worker_retries_exhausted',error_reason='The worker lease expired too many times.',
    finished_at=$1,updated_at=$1
FROM management_operations o
WHERE o.id=i.operation_id AND o.cancelled_at IS NULL
  AND i.state='running' AND i.lease_expires_at <= $1 AND i.attempt_count >= $2
RETURNING i.operation_id`
	cancelExpiredClaimsQuery = `UPDATE policy_bulk_operation_items i
SET state='cancelled',lease_owner=NULL,lease_token=NULL,lease_expires_at=NULL,
    error_code=NULL,error_reason=NULL,finished_at=$1,updated_at=$1
FROM management_operations o
WHERE o.id=i.operation_id AND o.cancelled_at IS NOT NULL
  AND i.state='running' AND i.lease_expires_at <= $1
RETURNING i.operation_id`
	expireOperationItemsQuery = `UPDATE policy_bulk_operation_items i
SET state='failed',lease_owner=NULL,lease_token=NULL,lease_expires_at=NULL,
    error_code='operation_expired',error_reason='The operation lifetime expired.',
    finished_at=$1,updated_at=$1
FROM management_operations o, policy_bulk_operation_contexts c
WHERE o.id=i.operation_id AND c.operation_id=i.operation_id
  AND o.cancelled_at IS NULL AND c.expires_at <= $1
  AND (i.state='pending' OR (i.state='running' AND i.lease_expires_at <= $1))
RETURNING i.operation_id`
)

type Repository struct{ db *sql.DB }

func NewRepository(db *sql.DB) (*Repository, error) {
	if db == nil {
		return nil, policybulk.ErrUnavailable
	}
	return &Repository{db: db}, nil
}

func (repository *Repository) Ready(ctx context.Context, codec *managementcommand.Codec) error {
	if repository == nil || repository.db == nil || codec == nil {
		return policybulk.ErrUnavailable
	}
	var table sql.NullString
	if err := repository.db.QueryRowContext(ctx,
		`SELECT to_regclass('policy_bulk_operation_items')::text`).Scan(&table); err != nil || !table.Valid {
		if err == nil {
			err = errors.New("policy bulk schema is not installed")
		}
		return fmt.Errorf("policy bulk readiness: %w", err)
	}
	return commandpostgres.ValidateReferencedHMACVersions(ctx, repository.db, codec)
}

func (repository *Repository) EnqueueAccess(ctx context.Context, command managementcommand.Command,
	operation policybulk.Operation, operationContext policybulk.OperationContext,
	items []policybulk.AccessBindingItem,
) (policybulk.EnqueueResult, error) {
	if policybulk.ValidateAccessItems(items) != nil || !matchesAccessTargets(operation.Targets, items) {
		return policybulk.EnqueueResult{}, policybulk.ErrInvalidRequest
	}
	return repository.enqueue(ctx, command, operation, operationContext, func(tx *sql.Tx) error {
		for index, item := range items {
			if _, err := tx.ExecContext(ctx, insertAccessItemQuery, operation.ID, operation.NamespaceID,
				item.ItemID, index, item.PolicyID, item.Subject.ID, item.Subject.Type, operation.CreatedAt); err != nil {
				return fmt.Errorf("insert AccessPolicy bulk item: %w", err)
			}
		}
		return nil
	})
}

func (repository *Repository) EnqueueRate(ctx context.Context, command managementcommand.Command,
	operation policybulk.Operation, operationContext policybulk.OperationContext,
	items []policybulk.RateBindingItem,
) (policybulk.EnqueueResult, error) {
	if policybulk.ValidateRateItems(items) != nil || !matchesRateTargets(operation.Targets, items) {
		return policybulk.EnqueueResult{}, policybulk.ErrInvalidRequest
	}
	return repository.enqueue(ctx, command, operation, operationContext, func(tx *sql.Tx) error {
		for index, item := range items {
			var policyID, name, description any
			if item.InlinePolicy == nil {
				policyID = item.PolicyID
			} else {
				name, description = item.InlinePolicy.Name, item.InlinePolicy.Description
			}
			if _, err := tx.ExecContext(ctx, insertRateItemQuery, operation.ID, operation.NamespaceID,
				item.ItemID, index, policyID, name, description, item.Subject.ID, item.Subject.Type,
				item.Mode, operation.CreatedAt); err != nil {
				return fmt.Errorf("insert RateLimit bulk item: %w", err)
			}
			if item.InlinePolicy != nil {
				for ordinal, rule := range item.InlinePolicy.Rules {
					if err := insertRule(ctx, tx, operation.ID, item.ItemID, ordinal, rule); err != nil {
						return err
					}
				}
			}
		}
		return nil
	})
}

func (repository *Repository) enqueue(ctx context.Context, command managementcommand.Command,
	operation policybulk.Operation, operationContext policybulk.OperationContext,
	insertItems func(*sql.Tx) error,
) (policybulk.EnqueueResult, error) {
	if repository == nil || repository.db == nil || !validOperation(operation) ||
		operationContext.RequestID == "" || len(operationContext.RequestID) > 256 ||
		!operationContext.ExpiresAt.After(operation.CreatedAt) {
		return policybulk.EnqueueResult{}, policybulk.ErrInvalidRequest
	}
	tx, err := repository.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelReadCommitted})
	if err != nil {
		return policybulk.EnqueueResult{}, fmt.Errorf("begin policy bulk enqueue: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	stored, replayed, err := commandpostgres.Lock(ctx, tx, command)
	if err != nil {
		return policybulk.EnqueueResult{}, mapWriteError(err)
	}
	if replayed {
		if stored.Operation == nil {
			return policybulk.EnqueueResult{}, policybulk.ErrConflict
		}
		replayedOperation, err := getOperation(ctx, tx, operation.NamespaceID, stored.Operation.OperationID)
		if err != nil {
			return policybulk.EnqueueResult{}, err
		}
		if err := tx.Commit(); err != nil {
			return policybulk.EnqueueResult{}, fmt.Errorf("commit policy bulk replay: %w", err)
		}
		return policybulk.EnqueueResult{Operation: replayedOperation, Replayed: true}, nil
	}
	actorChain, _ := json.Marshal(operation.ActorChain)
	targetIDs, _ := json.Marshal(operation.TargetIDs)
	targetScope, _ := json.Marshal(operation.Targets)
	digest := command.ActiveDigest().RequestDigest
	if _, err := tx.ExecContext(ctx, insertOperationQuery, operation.ID, operation.NamespaceID,
		operation.Kind, operation.OriginPrincipalID, actorChain, digest[:], operation.Total,
		targetScope, targetIDs, operation.CreatedAt); err != nil {
		return policybulk.EnqueueResult{}, mapWriteError(fmt.Errorf("insert policy bulk operation: %w", err))
	}
	var sourceIP any
	if operationContext.SourceIP.IsValid() {
		sourceIP = operationContext.SourceIP.Unmap().String()
	}
	if _, err := tx.ExecContext(ctx, insertOperationContextQuery, operation.ID,
		operationContext.RequestID, sourceIP, operationContext.ExpiresAt); err != nil {
		return policybulk.EnqueueResult{}, fmt.Errorf("insert policy bulk context: %w", err)
	}
	if err := insertItems(tx); err != nil {
		return policybulk.EnqueueResult{}, mapWriteError(err)
	}
	if err := commandpostgres.CompleteOperation(ctx, tx, command, managementcommand.OperationResult{
		OperationID: operation.ID, ResponseStatus: 202,
	}); err != nil {
		return policybulk.EnqueueResult{}, mapWriteError(err)
	}
	if err := tx.Commit(); err != nil {
		return policybulk.EnqueueResult{}, fmt.Errorf("commit policy bulk enqueue: %w", err)
	}
	return policybulk.EnqueueResult{Operation: operation}, nil
}

func (repository *Repository) Get(ctx context.Context, namespaceID, operationID string) (policybulk.Operation, error) {
	if repository == nil || repository.db == nil {
		return policybulk.Operation{}, policybulk.ErrUnavailable
	}
	return getOperation(ctx, repository.db, namespaceID, operationID)
}

func (repository *Repository) List(
	ctx context.Context,
	query policybulk.OperationQuery,
) (_ policybulk.RepositoryPage, returnErr error) {
	if repository == nil || repository.db == nil {
		return policybulk.RepositoryPage{}, policybulk.ErrUnavailable
	}
	if query.NamespaceID == "" || query.Limit < 1 || query.Limit > 200 ||
		(query.Kind != "" && query.Kind != policybulk.AccessBindingOperationKind &&
			query.Kind != policybulk.RateBindingOperationKind) ||
		(query.State != "" && !query.State.Valid()) ||
		(query.After != nil && (query.After.CreatedAt.IsZero() || query.After.ID == "")) {
		return policybulk.RepositoryPage{}, policybulk.ErrInvalidRequest
	}
	if !validOperationVisibility(query.NamespaceID, query.Visibility) {
		return policybulk.RepositoryPage{}, policybulk.ErrInvalidRequest
	}
	accessEmpty := operationPolicyScopeEmpty(query.Visibility.Access, accesscontrol.ScopeResourceAccessPolicy)
	rateEmpty := operationPolicyScopeEmpty(query.Visibility.Rate, accesscontrol.ScopeResourceRateLimitPolicy)
	if (query.Kind == policybulk.AccessBindingOperationKind && accessEmpty) ||
		(query.Kind == policybulk.RateBindingOperationKind && rateEmpty) ||
		(query.Kind == "" && accessEmpty && rateEmpty) {
		return policybulk.RepositoryPage{Items: []policybulk.Operation{}}, nil
	}
	arguments := []any{query.NamespaceID}
	statement := strings.Builder{}
	statement.WriteString(`SELECT ` + operationColumns + ` FROM management_operations o
WHERE o.namespace_id=$1
  AND o.kind IN ('access_policy_bindings.bulk_apply','rate_limit_bindings.bulk_apply')`)
	appendOperationVisibility(&statement, &arguments, query.Visibility)
	appendFilter := func(column string, value any) {
		arguments = append(arguments, value)
		statement.WriteString(fmt.Sprintf(" AND %s=$%d", column, len(arguments)))
	}
	if query.OriginPrincipalID != "" {
		appendFilter("o.origin_principal_id", query.OriginPrincipalID)
	}
	if query.Kind != "" {
		appendFilter("o.kind", query.Kind)
	}
	if query.State != "" {
		appendFilter("o.state", query.State)
	}
	if query.After != nil {
		arguments = append(arguments, query.After.CreatedAt, query.After.ID)
		statement.WriteString(fmt.Sprintf(" AND (o.created_at<$%d OR (o.created_at=$%d AND o.id>$%d::uuid))",
			len(arguments)-1, len(arguments)-1, len(arguments)))
	}
	arguments = append(arguments, query.Limit+1)
	statement.WriteString(fmt.Sprintf(" ORDER BY o.created_at DESC,o.id ASC LIMIT $%d", len(arguments)))
	rows, err := repository.db.QueryContext(ctx, statement.String(), arguments...)
	if err != nil {
		return policybulk.RepositoryPage{}, fmt.Errorf("list policy bulk operations: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	items := make([]policybulk.Operation, 0, query.Limit+1)
	for rows.Next() {
		operation, err := scanOperation(rows)
		if err != nil {
			return policybulk.RepositoryPage{}, fmt.Errorf("scan policy bulk operation list: %w", err)
		}
		items = append(items, operation)
	}
	if err := rows.Err(); err != nil {
		return policybulk.RepositoryPage{}, fmt.Errorf("iterate policy bulk operations: %w", err)
	}
	hasMore := len(items) > query.Limit
	if hasMore {
		items = items[:query.Limit]
	}
	return policybulk.RepositoryPage{Items: items, HasMore: hasMore}, nil
}

func appendOperationVisibility(statement *strings.Builder, arguments *[]any, visibility policybulk.OperationVisibility) {
	bind := func(value any) string {
		*arguments = append(*arguments, value)
		return fmt.Sprintf("$%d", len(*arguments))
	}
	subjectCoverage := func(scope accesscontrol.ResultScope, alias string) string {
		all := bind(scope.All)
		users := bind(pq.Array(scope.UserIDs))
		teams := bind(pq.Array(scope.TeamIDs))
		keys := bind(pq.Array(scope.APIKeyIDs))
		return fmt.Sprintf(`(%s OR (%s.subject_kind='user' AND %s.subject_id=ANY(%s::uuid[]))
        OR (%s.subject_kind='team' AND %s.subject_id=ANY(%s::uuid[]))
        OR (%s.subject_kind='api_key' AND %s.subject_id=ANY(%s::uuid[])))`,
			all, alias, alias, users, alias, alias, teams, alias, alias, keys)
	}
	domainCoverage := func(scope accesscontrol.ResultScope, itemKind, policyColumn string,
		resourceType accesscontrol.ScopeResourceType,
	) string {
		all := bind(scope.All)
		policies := bind(pq.Array(scope.IDs(resourceType)))
		return fmt.Sprintf(`NOT EXISTS (
      SELECT 1 FROM policy_bulk_operation_items i
      WHERE i.operation_id=o.id AND i.item_kind='%s'
        AND NOT (%s OR (i.%s IS NOT NULL AND i.%s=ANY(%s::uuid[])))
    )`, itemKind, all, policyColumn, policyColumn, policies)
	}
	accessCoverage := domainCoverage(visibility.Access, "access_policy_binding", "access_policy_id",
		accesscontrol.ScopeResourceAccessPolicy)
	rateCoverage := domainCoverage(visibility.Rate, "rate_limit_binding", "rate_policy_id",
		accesscontrol.ScopeResourceRateLimitPolicy)

	opAll := bind(visibility.Operation.All)
	opAccessPolicies := bind(pq.Array(visibility.Operation.IDs(accesscontrol.ScopeResourceAccessPolicy)))
	opRatePolicies := bind(pq.Array(visibility.Operation.IDs(accesscontrol.ScopeResourceRateLimitPolicy)))
	opSubjects := subjectCoverage(visibility.Operation, "i")
	opTargetCoverage := fmt.Sprintf(`NOT EXISTS (
    SELECT 1 FROM policy_bulk_operation_items i
    WHERE i.operation_id=o.id AND NOT (
      %s OR (((i.item_kind='access_policy_binding' AND i.access_policy_id=ANY(%s::uuid[]))
        OR (i.item_kind='rate_limit_binding' AND i.rate_policy_id IS NOT NULL AND i.rate_policy_id=ANY(%s::uuid[])))
        AND %s)
    )
	  )`, opAll, opAccessPolicies, opRatePolicies, opSubjects)
	origin := bind(visibility.PrincipalID)
	fmt.Fprintf(statement, ` AND (
  ((o.kind='%s' AND %s) OR (o.kind='%s' AND %s))
  AND (o.origin_principal_id=%s::uuid OR %s)
)`, policybulk.AccessBindingOperationKind, accessCoverage,
		policybulk.RateBindingOperationKind, rateCoverage, origin, opTargetCoverage)
}

func validOperationVisibility(namespaceID string, visibility policybulk.OperationVisibility) bool {
	if _, err := uuid.Parse(visibility.PrincipalID); err != nil {
		return false
	}
	for _, scope := range []accesscontrol.ResultScope{visibility.Operation, visibility.Access, visibility.Rate} {
		if string(scope.NamespaceID) != namespaceID {
			return false
		}
		if _, err := scope.Digest(); err != nil {
			return false
		}
	}
	return true
}

func operationPolicyScopeEmpty(scope accesscontrol.ResultScope, resourceType accesscontrol.ScopeResourceType) bool {
	return !scope.All && len(scope.IDs(resourceType)) == 0
}

func (repository *Repository) Cancel(ctx context.Context, command managementcommand.Command,
	request policybulk.CancelRequest,
) (policybulk.CancelResult, error) {
	if repository == nil || repository.db == nil {
		return policybulk.CancelResult{}, policybulk.ErrUnavailable
	}
	tx, cancelErr := repository.db.BeginTx(ctx, nil)
	if cancelErr != nil {
		return policybulk.CancelResult{}, fmt.Errorf("begin policy bulk cancel: %w", cancelErr)
	}
	defer func() { _ = tx.Rollback() }()
	stored, replayed, cancelErr := commandpostgres.Lock(ctx, tx, command)
	if cancelErr != nil {
		return policybulk.CancelResult{}, mapWriteError(cancelErr)
	}
	if replayed {
		if stored.Operation == nil || stored.Operation.OperationID != request.OperationID {
			return policybulk.CancelResult{}, policybulk.ErrConflict
		}
		operation, err := getOperation(ctx, tx, request.NamespaceID, request.OperationID)
		if err != nil {
			return policybulk.CancelResult{}, err
		}
		if err := tx.Commit(); err != nil {
			return policybulk.CancelResult{}, fmt.Errorf("commit policy bulk cancel replay: %w", err)
		}
		return policybulk.CancelResult{Operation: operation, Replayed: true}, nil
	}
	var state string
	var cancelled bool
	var version uint64
	if err := tx.QueryRowContext(ctx, lockOperationCancelQuery, request.NamespaceID,
		request.OperationID).Scan(&state, &cancelled, &version); err != nil {
		return policybulk.CancelResult{}, mapReadError(err)
	}
	if version != request.ExpectedVersion {
		return policybulk.CancelResult{}, policybulk.ErrRevisionConflict
	}
	if terminalState(policybulk.OperationState(state)) || cancelled {
		return policybulk.CancelResult{}, policybulk.ErrConflict
	}
	now := time.Now().UTC().Truncate(time.Microsecond)
	changed, cancelErr := tx.ExecContext(ctx, cancelOperationQuery, request.NamespaceID,
		request.OperationID, now, request.ExpectedVersion)
	if cancelErr != nil {
		return policybulk.CancelResult{}, fmt.Errorf("cancel policy bulk operation: %w", cancelErr)
	}
	if err := requireChanged(changed); err != nil {
		return policybulk.CancelResult{}, policybulk.ErrRevisionConflict
	}
	if _, err := tx.ExecContext(ctx, cancelPendingItemsQuery, request.OperationID, now); err != nil {
		return policybulk.CancelResult{}, fmt.Errorf("cancel pending policy bulk items: %w", err)
	}
	if _, err := finalizeOperation(ctx, tx, request.OperationID, now); err != nil {
		return policybulk.CancelResult{}, err
	}
	operation, cancelErr := getOperation(ctx, tx, request.NamespaceID, request.OperationID)
	if cancelErr != nil {
		return policybulk.CancelResult{}, cancelErr
	}
	if err := commandpostgres.CompleteOperation(ctx, tx, command, managementcommand.OperationResult{
		OperationID: request.OperationID, ResponseStatus: 200,
	}); err != nil {
		return policybulk.CancelResult{}, mapWriteError(err)
	}
	if err := tx.Commit(); err != nil {
		return policybulk.CancelResult{}, fmt.Errorf("commit policy bulk cancel: %w", err)
	}
	return policybulk.CancelResult{Operation: operation}, nil
}

func (repository *Repository) Claim(ctx context.Context, workerID string, now time.Time,
	lease time.Duration, maximumAttempts int,
) (policybulk.ItemClaim, bool, error) {
	if repository == nil || repository.db == nil || workerID == "" || lease <= 0 || maximumAttempts < 1 {
		return policybulk.ItemClaim{}, false, policybulk.ErrInvalidRequest
	}
	tx, claimErr := repository.db.BeginTx(ctx, nil)
	if claimErr != nil {
		return policybulk.ItemClaim{}, false, fmt.Errorf("begin policy bulk claim: %w", claimErr)
	}
	defer func() { _ = tx.Rollback() }()
	if err := expireExhaustedClaims(ctx, tx, now, maximumAttempts); err != nil {
		return policybulk.ItemClaim{}, false, err
	}
	claim, claimErr := scanClaim(tx.QueryRowContext(ctx, claimItemQuery, now, maximumAttempts))
	if errors.Is(claimErr, sql.ErrNoRows) {
		if err := tx.Commit(); err != nil {
			return policybulk.ItemClaim{}, false, fmt.Errorf("commit empty policy bulk claim: %w", err)
		}
		return policybulk.ItemClaim{}, false, nil
	}
	if claimErr != nil {
		return policybulk.ItemClaim{}, false, fmt.Errorf("scan policy bulk claim: %w", claimErr)
	}
	claim.Attempt++
	claim.LeaseOwner = workerID
	claim.LeaseToken = uuid.NewString()
	claim.LeaseExpiresAt = now.Add(lease)
	if _, err := tx.ExecContext(ctx, claimUpdateQuery, claim.OperationID, now, workerID,
		claim.LeaseToken, claim.LeaseExpiresAt, claimItemID(claim)); err != nil {
		return policybulk.ItemClaim{}, false, fmt.Errorf("claim policy bulk item: %w", err)
	}
	if _, err := tx.ExecContext(ctx, markOperationRunningQuery, claim.OperationID, now); err != nil {
		return policybulk.ItemClaim{}, false, fmt.Errorf("mark policy bulk operation running: %w", err)
	}
	if claim.Rate != nil && claim.Rate.InlinePolicy != nil {
		rules, err := loadRateRules(ctx, tx, claim.OperationID, claim.Rate.ItemID)
		if err != nil {
			return policybulk.ItemClaim{}, false, err
		}
		claim.Rate.InlinePolicy.Rules = rules
	}
	if err := tx.Commit(); err != nil {
		return policybulk.ItemClaim{}, false, fmt.Errorf("commit policy bulk claim: %w", err)
	}
	return claim, true, nil
}

func (repository *Repository) Complete(ctx context.Context, claim policybulk.ItemClaim,
	result policybulk.ItemResult, now time.Time,
) (policybulk.Operation, error) {
	return repository.finish(ctx, claim, now, func(tx *sql.Tx) error {
		updated, err := tx.ExecContext(ctx, completeItemQuery, claim.OperationID, claimItemID(claim),
			claim.LeaseOwner, now, nullableUUID(result.BindingID), nullableUUID(result.PolicyID), claim.LeaseToken)
		if err != nil {
			return fmt.Errorf("complete policy bulk item: %w", err)
		}
		return requireChanged(updated)
	})
}

func (repository *Repository) Fail(ctx context.Context, claim policybulk.ItemClaim,
	failure policybulk.ItemFailure, retry bool, retryAt, now time.Time, maximumAttempts int,
) (policybulk.Operation, error) {
	return repository.finish(ctx, claim, now, func(tx *sql.Tx) error {
		cancelled, err := operationCancelled(ctx, tx, claim.OperationID)
		if err != nil {
			return err
		}
		state := "failed"
		var availableAt any = now
		var finishedAt any = now
		var code, reason any = failure.Code, failure.Reason
		if cancelled {
			state, code, reason = "cancelled", nil, nil
		} else if retry && claim.Attempt < maximumAttempts {
			state, availableAt, finishedAt, code, reason = "pending", retryAt, nil, nil, nil
		}
		updated, err := tx.ExecContext(ctx, `UPDATE policy_bulk_operation_items
SET state=$5,available_at=$6,lease_owner=NULL,lease_token=NULL,lease_expires_at=NULL,
    error_code=$7,error_reason=$8,finished_at=$9,updated_at=$4
WHERE operation_id=$1 AND item_id=$2 AND state='running'
  AND lease_owner=$3 AND lease_token=$10`, claim.OperationID, claimItemID(claim), claim.LeaseOwner,
			now, state, availableAt, code, reason, finishedAt, claim.LeaseToken)
		if err != nil {
			return fmt.Errorf("fail policy bulk item: %w", err)
		}
		return requireChanged(updated)
	})
}

func (repository *Repository) finish(ctx context.Context, claim policybulk.ItemClaim, now time.Time,
	mutate func(*sql.Tx) error,
) (policybulk.Operation, error) {
	if repository == nil || repository.db == nil {
		return policybulk.Operation{}, policybulk.ErrUnavailable
	}
	tx, finishErr := repository.db.BeginTx(ctx, nil)
	if finishErr != nil {
		return policybulk.Operation{}, fmt.Errorf("begin policy bulk completion: %w", finishErr)
	}
	defer func() { _ = tx.Rollback() }()
	if err := mutate(tx); err != nil {
		return policybulk.Operation{}, err
	}
	if _, err := finalizeOperation(ctx, tx, claim.OperationID, now); err != nil {
		return policybulk.Operation{}, err
	}
	operation, finishErr := getOperation(ctx, tx, claim.NamespaceID, claim.OperationID)
	if finishErr != nil {
		return policybulk.Operation{}, finishErr
	}
	if err := tx.Commit(); err != nil {
		return policybulk.Operation{}, fmt.Errorf("commit policy bulk completion: %w", err)
	}
	return operation, nil
}
