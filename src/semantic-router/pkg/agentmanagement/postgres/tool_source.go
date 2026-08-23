package postgres

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
)

func (store *Store) ListToolSources(
	ctx context.Context, namespaceID string, query agentmanagement.ListQuery,
) (_ agentmanagement.ListResult[agentmanagement.ToolSource], returnErr error) {
	ids := scopedIDs(query.Scope, accesscontrol.ScopeResourceAgentToolSource)
	statement := sourceSelect + `
 WHERE s.namespace_id=$1 AND s.status<>'deleted' AND ($2 OR s.id=ANY($3::uuid[]))
	AND ($4='' OR lower(s.name) LIKE $4 ESCAPE '\' OR lower(s.description) LIKE $4 ESCAPE '\')
	AND ($5::timestamptz IS NULL OR (s.created_at,s.id)<($5,$6::uuid))
 ORDER BY s.created_at DESC,s.id DESC LIMIT $7`
	var afterTime any
	afterID := "00000000-0000-0000-0000-000000000000"
	if query.After != nil {
		afterTime, afterID = query.After.Timestamp, query.After.ID
	}
	rows, err := store.db.QueryContext(
		ctx, statement, namespaceID, query.Scope.All, pq.Array(ids),
		managementsearch.PrefixPattern(query.Search), afterTime, afterID, query.Limit,
	)
	if err != nil {
		return agentmanagement.ListResult[agentmanagement.ToolSource]{}, fmt.Errorf("list Agent Tool Sources: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	items := make([]agentmanagement.ToolSource, 0, query.Limit)
	for rows.Next() {
		value, scanErr := scanToolSource(rows)
		if scanErr != nil {
			return agentmanagement.ListResult[agentmanagement.ToolSource]{}, scanErr
		}
		items = append(items, value)
	}
	if err := rows.Err(); err != nil {
		return agentmanagement.ListResult[agentmanagement.ToolSource]{}, fmt.Errorf("iterate Agent Tool Sources: %w", err)
	}
	return agentmanagement.ListResult[agentmanagement.ToolSource]{Items: items, HasMore: len(items) == query.Limit}, nil
}

func (store *Store) GetToolSource(
	ctx context.Context, namespaceID, id string,
) (agentmanagement.ToolSource, error) {
	return scanToolSource(store.db.QueryRowContext(ctx, sourceSelect+`
 WHERE s.namespace_id=$1 AND s.id=$2 AND s.status<>'deleted'`, namespaceID, id))
}

func (store *Store) GetToolSourceRevision(
	ctx context.Context, namespaceID, id string, revision int64,
) (agentmanagement.ToolSource, error) {
	return scanToolSource(store.db.QueryRowContext(ctx, sourceRevisionSelect+`
 WHERE s.namespace_id=$1 AND s.id=$2 AND r.revision=$3`, namespaceID, id, revision))
}

// ListRegistryToolSources is an internal execution query. It returns only
// currently active sources; Registry assembly separately requires an exact
// approved discovery digest before any remote definition is registered.
func (store *Store) ListRegistryToolSources(
	ctx context.Context, namespaceID string,
) (_ []agentmanagement.ToolSource, returnErr error) {
	rows, err := store.db.QueryContext(ctx, sourceSelect+`
 WHERE s.namespace_id=$1 AND s.status='active'
 ORDER BY s.id`, namespaceID)
	if err != nil {
		return nil, fmt.Errorf("list Agent registry Tool Sources: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	result := make([]agentmanagement.ToolSource, 0)
	for rows.Next() {
		value, scanErr := scanToolSource(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		result = append(result, value)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Agent registry Tool Sources: %w", err)
	}
	return result, nil
}

func (store *Store) CreateToolSource(
	ctx context.Context, namespaceID, id string, input agentmanagement.ToolSourceInput,
	mutation agentmanagement.ResourceCommand,
) (agentmanagement.ResourceMutationResult, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.ResourceMutationResult, error) {
		if replay, found, err := lockResourceCommand(
			ctx, tx, namespaceID, agentToolSourceResourceType, mutation,
		); err != nil || found {
			return replay, err
		}
		encoded, digest, err := encodeSourceRevision(input, nil)
		if err != nil {
			return agentmanagement.ResourceMutationResult{}, err
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_tool_sources
  (id,namespace_id,name,description,source_kind,status,current_revision,revision)
VALUES ($1,$2,$3,$4,$5,'active',1,1)`, id, namespaceID, input.Name, input.Description, input.Kind); err != nil {
			return agentmanagement.ResourceMutationResult{}, classifyWriteError(err)
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_tool_source_revisions
  (source_id,namespace_id,revision,transport,endpoint,credential_id,egress_policy,
   discovered_tools,content_digest,created_by)
VALUES ($1,$2,1,$3,$4,$5,$6,'[]'::jsonb,$7,$8)`, id, namespaceID, input.Transport,
			input.Endpoint, nullableString(input.CredentialID), encoded.egress, digest[:],
			nullableString(mutation.Mutation.PrincipalID)); err != nil {
			return agentmanagement.ResourceMutationResult{}, classifyWriteError(err)
		}
		return completeResourceCommand(ctx, tx, mutation, agentToolSourceResourceType, id, 1, 201)
	})
}

func (store *Store) PatchToolSource(
	ctx context.Context, namespaceID, id string, expected int64, patch agentmanagement.ToolSourcePatch,
	mutation agentmanagement.MutationContext,
) (agentmanagement.ToolSource, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.ToolSource, error) {
		current, patchToolSourceErr := lockToolSource(ctx, tx, namespaceID, id, expected)
		if patchToolSourceErr != nil {
			return agentmanagement.ToolSource{}, patchToolSourceErr
		}
		input, status := applyToolSourcePatch(current, patch)
		tools, discoveryDigest, invalidateApproval, patchToolSourceErr := sourceRevisionDiscovery(current, input)
		if patchToolSourceErr != nil {
			return agentmanagement.ToolSource{}, patchToolSourceErr
		}
		if !invalidateApproval {
			result, err := tx.ExecContext(ctx, `UPDATE agent_tool_sources
SET name=$4,description=$5,status=$6,revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND status<>'deleted'`,
				namespaceID, id, expected, input.Name, input.Description, status)
			if err != nil {
				return agentmanagement.ToolSource{}, classifyWriteError(err)
			}
			if err := requireOneRow(result); err != nil {
				return agentmanagement.ToolSource{}, err
			}
			return scanToolSource(tx.QueryRowContext(ctx, sourceSelect+`
 WHERE s.namespace_id=$1 AND s.id=$2`, namespaceID, id))
		}
		contentRevision, patchToolSourceErr := ensureToolSourceRevision(
			ctx, tx, namespaceID, id, input, tools, discoveryDigest, mutation,
		)
		if patchToolSourceErr != nil {
			return agentmanagement.ToolSource{}, patchToolSourceErr
		}
		result, patchToolSourceErr := tx.ExecContext(ctx, `UPDATE agent_tool_sources
SET name=$4,description=$5,status=$6,current_revision=$7,
    approved_discovery_digest=CASE WHEN $8 THEN NULL ELSE approved_discovery_digest END,
    revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND status<>'deleted'`,
			namespaceID, id, expected, input.Name, input.Description, status, contentRevision,
			invalidateApproval)
		if patchToolSourceErr != nil {
			return agentmanagement.ToolSource{}, classifyWriteError(patchToolSourceErr)
		}
		if err := requireOneRow(result); err != nil {
			return agentmanagement.ToolSource{}, err
		}
		return scanToolSource(tx.QueryRowContext(ctx, sourceSelect+` WHERE s.namespace_id=$1 AND s.id=$2`, namespaceID, id))
	})
}

// sourceRevisionDiscovery binds discovery approval to the complete connection
// boundary that was tested. Metadata and enable/disable changes preserve a
// tested definition set; changing where or how the Router connects requires a
// fresh discovery and explicit approval before the source can re-enter a Tool
// Registry.
func sourceRevisionDiscovery(
	current agentmanagement.ToolSource, input agentmanagement.ToolSourceInput,
) ([]agentmanagement.ToolDefinition, []byte, bool, error) {
	if toolSourceConnectionChanged(current, input) {
		return []agentmanagement.ToolDefinition{}, nil, true, nil
	}
	digest, err := parseDigest(current.DiscoveryDigest)
	if err != nil {
		return nil, nil, false, err
	}
	return append([]agentmanagement.ToolDefinition(nil), current.DiscoveredTools...), digest, false, nil
}

func toolSourceConnectionChanged(
	current agentmanagement.ToolSource, input agentmanagement.ToolSourceInput,
) bool {
	if current.Transport != input.Transport || current.Endpoint != input.Endpoint ||
		current.CredentialID != input.CredentialID {
		return true
	}
	return !slices.Equal(current.EgressPolicy.AllowedHosts, input.EgressPolicy.AllowedHosts) ||
		!slices.Equal(current.EgressPolicy.AllowedPorts, input.EgressPolicy.AllowedPorts) ||
		!slices.Equal(current.EgressPolicy.AllowedPrivateCIDRs, input.EgressPolicy.AllowedPrivateCIDRs)
}

func (store *Store) DeleteToolSource(
	ctx context.Context, namespaceID, id string, expected int64, _ agentmanagement.MutationContext,
) (int64, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (int64, error) {
		result, err := tx.ExecContext(ctx, `UPDATE agent_tool_sources
SET status='deleted',revision=revision+1,updated_at=clock_timestamp(),deleted_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND status<>'deleted'`, namespaceID, id, expected)
		if err != nil {
			return 0, classifyWriteError(err)
		}
		if err := requireOneRow(result); err != nil {
			return 0, err
		}
		return expected + 1, nil
	})
}

func (store *Store) UpdateToolSourceDiscovery(
	ctx context.Context, namespaceID, id string, expected int64, tools []agentmanagement.ToolDefinition,
	mutation agentmanagement.ResourceCommand,
) (agentmanagement.ResourceMutationResult, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.ResourceMutationResult, error) {
		if replay, found, err := lockResourceCommand(
			ctx, tx, namespaceID, agentToolSourceResourceType, mutation,
		); err != nil || found {
			return replay, err
		}
		current, err := lockToolSource(ctx, tx, namespaceID, id, expected)
		if err != nil {
			return agentmanagement.ResourceMutationResult{}, err
		}
		input := agentmanagement.ToolSourceInput{
			Name: current.Name, Description: current.Description, Kind: current.Kind,
			Transport: current.Transport, Endpoint: current.Endpoint, CredentialID: current.CredentialID,
			EgressPolicy: current.EgressPolicy,
		}
		toolBytes, err := json.Marshal(tools)
		if err != nil {
			return agentmanagement.ResourceMutationResult{}, err
		}
		discoveryDigest := sha256.Sum256(toolBytes)
		contentRevision, err := ensureToolSourceRevision(
			ctx, tx, namespaceID, id, input, tools, discoveryDigest[:], mutation.Mutation,
		)
		if err != nil {
			return agentmanagement.ResourceMutationResult{}, err
		}
		result, err := tx.ExecContext(ctx, `UPDATE agent_tool_sources
SET current_revision=$4,revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND status='active'`, namespaceID, id, expected, contentRevision)
		if err != nil {
			return agentmanagement.ResourceMutationResult{}, classifyWriteError(err)
		}
		if err := requireOneRow(result); err != nil {
			return agentmanagement.ResourceMutationResult{}, err
		}
		return completeResourceCommand(
			ctx, tx, mutation, agentToolSourceResourceType, id, expected+1, 200,
		)
	})
}

func (store *Store) ApproveToolSourceDiscovery(
	ctx context.Context,
	namespaceID string,
	id string,
	expected int64,
	discoveryDigest string,
	mutation agentmanagement.ResourceCommand,
) (agentmanagement.ResourceMutationResult, error) {
	digest, err := parseDigest(discoveryDigest)
	if err != nil || len(digest) != sha256.Size {
		return agentmanagement.ResourceMutationResult{}, agentmanagement.ErrInvalid
	}
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.ResourceMutationResult, error) {
		if replay, found, err := lockResourceCommand(
			ctx, tx, namespaceID, agentToolSourceResourceType, mutation,
		); err != nil || found {
			return replay, err
		}
		current, err := lockToolSource(ctx, tx, namespaceID, id, expected)
		if err != nil {
			return agentmanagement.ResourceMutationResult{}, err
		}
		if current.Status != agentmanagement.StatusActive || current.DiscoveryDigest != discoveryDigest {
			return agentmanagement.ResourceMutationResult{}, agentmanagement.ErrConflict
		}
		result, err := tx.ExecContext(ctx, `UPDATE agent_tool_sources
SET approved_discovery_digest=$4,revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND status='active'`,
			namespaceID, id, expected, digest)
		if err != nil {
			return agentmanagement.ResourceMutationResult{}, classifyWriteError(err)
		}
		if err := requireOneRow(result); err != nil {
			return agentmanagement.ResourceMutationResult{}, err
		}
		return completeResourceCommand(
			ctx, tx, mutation, agentToolSourceResourceType, id, expected+1, 200,
		)
	})
}

type encodedSourceRevision struct{ egress []byte }

// ensureToolSourceRevision installs immutable executable content or reuses an
// exact historical revision. The caller must hold the Tool Source root row
// lock: that lock is the per-source allocator for monotonically increasing
// revision numbers, even after the root is moved back to older content.
func ensureToolSourceRevision(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	id string,
	input agentmanagement.ToolSourceInput,
	tools []agentmanagement.ToolDefinition,
	discoveryDigest []byte,
	mutation agentmanagement.MutationContext,
) (int64, error) {
	encoded, digest, ensureToolSourceRevisionErr := encodeSourceRevision(input, tools)
	if ensureToolSourceRevisionErr != nil {
		return 0, ensureToolSourceRevisionErr
	}
	var contentRevision int64
	ensureToolSourceRevisionErr = tx.QueryRowContext(ctx, `SELECT revision FROM agent_tool_source_revisions
WHERE namespace_id=$1 AND source_id=$2 AND content_digest=$3`,
		namespaceID, id, digest[:]).Scan(&contentRevision)
	if ensureToolSourceRevisionErr == nil {
		return contentRevision, nil
	}
	if !errors.Is(ensureToolSourceRevisionErr, sql.ErrNoRows) {
		return 0, fmt.Errorf("find existing Agent Tool Source revision: %w", ensureToolSourceRevisionErr)
	}
	if err := tx.QueryRowContext(ctx, `SELECT COALESCE(max(revision),0)+1
FROM agent_tool_source_revisions WHERE namespace_id=$1 AND source_id=$2`,
		namespaceID, id).Scan(&contentRevision); err != nil {
		return 0, fmt.Errorf("allocate Agent Tool Source revision: %w", err)
	}
	toolBytes, ensureToolSourceRevisionErr := json.Marshal(tools)
	if ensureToolSourceRevisionErr != nil {
		return 0, ensureToolSourceRevisionErr
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO agent_tool_source_revisions
  (source_id,namespace_id,revision,transport,endpoint,credential_id,egress_policy,
   discovered_tools,discovery_digest,content_digest,created_by)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)`, id, namespaceID, contentRevision,
		input.Transport, input.Endpoint, nullableString(input.CredentialID), encoded.egress,
		toolBytes, nullableBytes(discoveryDigest), digest[:], nullableString(mutation.PrincipalID)); err != nil {
		return 0, classifyWriteError(err)
	}
	return contentRevision, nil
}

func encodeSourceRevision(
	input agentmanagement.ToolSourceInput, tools []agentmanagement.ToolDefinition,
) (encodedSourceRevision, [sha256.Size]byte, error) {
	if tools == nil {
		tools = []agentmanagement.ToolDefinition{}
	}
	egress, err := json.Marshal(input.EgressPolicy)
	if err != nil {
		return encodedSourceRevision{}, [sha256.Size]byte{}, err
	}
	canonical, err := json.Marshal(struct {
		Kind         string                           `json:"kind"`
		Transport    string                           `json:"transport"`
		Endpoint     string                           `json:"endpoint"`
		CredentialID string                           `json:"credentialId,omitempty"`
		EgressPolicy agentmanagement.EgressPolicy     `json:"egressPolicy"`
		Tools        []agentmanagement.ToolDefinition `json:"tools"`
	}{
		Kind: input.Kind, Transport: input.Transport, Endpoint: input.Endpoint,
		CredentialID: input.CredentialID, EgressPolicy: input.EgressPolicy, Tools: tools,
	})
	if err != nil {
		return encodedSourceRevision{}, [sha256.Size]byte{}, err
	}
	return encodedSourceRevision{egress: egress}, sha256.Sum256(canonical), nil
}

func lockToolSource(
	ctx context.Context, tx *sql.Tx, namespaceID, id string, expected int64,
) (agentmanagement.ToolSource, error) {
	value, err := scanToolSource(tx.QueryRowContext(ctx, sourceSelect+`
 WHERE s.namespace_id=$1 AND s.id=$2 AND s.status<>'deleted' FOR UPDATE`, namespaceID, id))
	if err != nil {
		return agentmanagement.ToolSource{}, err
	}
	if value.Revision != expected {
		return agentmanagement.ToolSource{}, agentmanagement.ErrConflict
	}
	return value, nil
}

func applyToolSourcePatch(
	current agentmanagement.ToolSource, patch agentmanagement.ToolSourcePatch,
) (agentmanagement.ToolSourceInput, agentmanagement.Status) {
	input := agentmanagement.ToolSourceInput{
		Name: current.Name, Description: current.Description, Kind: current.Kind,
		Transport: current.Transport, Endpoint: current.Endpoint, CredentialID: current.CredentialID,
		EgressPolicy: current.EgressPolicy,
	}
	status := current.Status
	if patch.Name != nil {
		input.Name = *patch.Name
	}
	if patch.Description != nil {
		input.Description = *patch.Description
	}
	if patch.Transport != nil {
		input.Transport = *patch.Transport
	}
	if patch.Endpoint != nil {
		input.Endpoint = *patch.Endpoint
	}
	if patch.CredentialID.Present {
		input.CredentialID = ""
		if patch.CredentialID.Value != nil {
			input.CredentialID = *patch.CredentialID.Value
		}
	}
	if patch.EgressPolicy != nil {
		input.EgressPolicy = *patch.EgressPolicy
	}
	if patch.Status != nil {
		status = *patch.Status
	}
	return input, status
}
