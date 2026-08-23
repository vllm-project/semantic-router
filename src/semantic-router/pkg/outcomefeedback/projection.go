package outcomefeedback

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"time"
)

const ProjectionSchema = "outcome-learning.v1"

type ProjectionEntry struct {
	RecipeID             string `json:"recipe_id,omitempty"`
	RecipeName           string `json:"recipe_name,omitempty"`
	RecipeRevision       int64  `json:"recipe_revision,omitempty"`
	DecisionID           string `json:"decision_id,omitempty"`
	DecisionName         string `json:"decision_name,omitempty"`
	DecisionTier         int    `json:"decision_tier,omitempty"`
	ModelID              string `json:"model_id"`
	ModelName            string `json:"model_name"`
	ModelRevision        int64  `json:"model_revision"`
	GoodFitCount         int64  `json:"good_fit_count"`
	UnderpoweredCount    int64  `json:"underpowered_count"`
	OverprovisionedCount int64  `json:"overprovisioned_count"`
	FailedCount          int64  `json:"failed_count"`
}

type Projection struct {
	Schema      string            `json:"schema"`
	NamespaceID string            `json:"namespace_id"`
	Revision    int64             `json:"revision"`
	Entries     []ProjectionEntry `json:"entries"`
	Digest      string            `json:"-"`
}

func (projection Projection) Validate() error {
	if projection.Schema != ProjectionSchema || projection.Revision <= 0 ||
		!canonicalIdentifier(projection.NamespaceID, MaximumReplayIDSize) {
		return fmt.Errorf("invalid outcome learning projection identity")
	}
	previous := ""
	for _, entry := range projection.Entries {
		if entry.ModelID == "" || entry.ModelName == "" || entry.ModelRevision <= 0 ||
			entry.GoodFitCount < 0 || entry.UnderpoweredCount < 0 ||
			entry.OverprovisionedCount < 0 || entry.FailedCount < 0 {
			return fmt.Errorf("invalid outcome learning projection entry")
		}
		key := projectionEntryKey(entry)
		if previous != "" && key <= previous {
			return fmt.Errorf("outcome learning projection entries are not strictly ordered")
		}
		previous = key
	}
	return nil
}

func (projection Projection) Canonical() ([]byte, [sha256.Size]byte, error) {
	projection.Digest = ""
	if err := projection.Validate(); err != nil {
		return nil, [sha256.Size]byte{}, err
	}
	payload, err := json.Marshal(projection)
	if err != nil {
		return nil, [sha256.Size]byte{}, fmt.Errorf("encode outcome learning projection: %w", err)
	}
	return payload, sha256.Sum256(payload), nil
}

type ProjectionRepository interface {
	PendingNamespaces(context.Context, int) ([]string, error)
	Build(context.Context, string) (Projection, error)
	Stage(context.Context, Projection, []byte, [sha256.Size]byte) error
	MarkApplied(context.Context, string, int64, [sha256.Size]byte) error
}

type ProjectionPublisher interface {
	Publish(context.Context, Projection, []byte, [sha256.Size]byte) error
}

type ProjectorOptions struct {
	Repository ProjectionRepository
	Publisher  ProjectionPublisher
	BatchSize  int
	Interval   time.Duration
}

type Projector struct {
	repository ProjectionRepository
	publisher  ProjectionPublisher
	batchSize  int
	interval   time.Duration
}

func NewProjector(options ProjectorOptions) (*Projector, error) {
	if options.Repository == nil || options.Publisher == nil {
		return nil, errors.New("outcome projection repository and publisher are required")
	}
	batchSize := options.BatchSize
	if batchSize == 0 {
		batchSize = 32
	}
	if batchSize < 1 || batchSize > 1000 {
		return nil, errors.New("outcome projection batch size must be between 1 and 1000")
	}
	interval := options.Interval
	if interval == 0 {
		interval = 250 * time.Millisecond
	}
	if interval < 25*time.Millisecond || interval > time.Minute {
		return nil, errors.New("outcome projection interval is outside supported bounds")
	}
	return &Projector{
		repository: options.Repository, publisher: options.Publisher,
		batchSize: batchSize, interval: interval,
	}, nil
}

func (projector *Projector) ProcessOnce(ctx context.Context) (int, error) {
	if projector == nil || projector.repository == nil || projector.publisher == nil {
		return 0, ErrUnavailable
	}
	namespaces, err := projector.repository.PendingNamespaces(ctx, projector.batchSize)
	if err != nil {
		return 0, err
	}
	applied := 0
	for _, namespaceID := range namespaces {
		if err := projector.Rebuild(ctx, namespaceID); err != nil {
			return applied, err
		}
		applied++
	}
	return applied, nil
}

// Rebuild reconstructs the complete projection from immutable outcomes. It is
// safe after process loss and intentionally does not depend on in-memory
// experience or an outbox delivery count.
func (projector *Projector) Rebuild(ctx context.Context, namespaceID string) error {
	projection, err := projector.repository.Build(ctx, namespaceID)
	if err != nil {
		return err
	}
	payload, digest, err := projection.Canonical()
	if err != nil {
		return err
	}
	if err := projector.repository.Stage(ctx, projection, payload, digest); err != nil {
		return err
	}
	if err := projector.publisher.Publish(ctx, projection, payload, digest); err != nil {
		return err
	}
	return projector.repository.MarkApplied(ctx, projection.NamespaceID, projection.Revision, digest)
}

func (projector *Projector) Run(ctx context.Context) error {
	if _, err := projector.ProcessOnce(ctx); err != nil {
		return err
	}
	ticker := time.NewTicker(projector.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if _, err := projector.ProcessOnce(ctx); err != nil {
				return err
			}
		}
	}
}

func (repository *PostgresRepository) PendingNamespaces(
	ctx context.Context,
	limit int,
) (_ []string, returnErr error) {
	if repository == nil || repository.database == nil {
		return nil, ErrUnavailable
	}
	rows, err := repository.database.QueryContext(ctx, `SELECT namespace_id::text
FROM inference_outcome_projection_heads
WHERE desired_revision > applied_revision
ORDER BY updated_at, namespace_id
LIMIT $1`, limit)
	if err != nil {
		return nil, fmt.Errorf("%w: list pending outcome projections", ErrUnavailable)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	namespaces := make([]string, 0, limit)
	for rows.Next() {
		var namespaceID string
		if err := rows.Scan(&namespaceID); err != nil {
			return nil, fmt.Errorf("%w: scan pending outcome projection", ErrUnavailable)
		}
		namespaces = append(namespaces, namespaceID)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("%w: iterate pending outcome projections", ErrUnavailable)
	}
	return namespaces, nil
}

func (repository *PostgresRepository) Build(
	ctx context.Context,
	namespaceID string,
) (_ Projection, returnErr error) {
	if repository == nil || repository.database == nil {
		return Projection{}, ErrUnavailable
	}
	var revision int64
	if err := repository.database.QueryRowContext(ctx, `SELECT desired_revision
FROM inference_outcome_projection_heads WHERE namespace_id=$1`, namespaceID).Scan(&revision); err != nil {
		return Projection{}, fmt.Errorf("%w: read outcome projection head", ErrUnavailable)
	}
	rows, err := repository.database.QueryContext(ctx, `SELECT
  o.target_model_id, o.target_model_name, o.target_revision, o.verdict,
  r.routing_context
FROM inference_outcomes o
JOIN inference_replays r
  ON r.namespace_id=o.namespace_id AND r.replay_id=o.replay_id
WHERE o.namespace_id=$1 AND o.target='model' AND o.projection_revision <= $2
ORDER BY o.projection_revision`, namespaceID, revision)
	if err != nil {
		return Projection{}, fmt.Errorf("%w: read durable model outcomes", ErrUnavailable)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	entries := make(map[string]*ProjectionEntry)
	for rows.Next() {
		var (
			modelID, modelName string
			modelRevision      int64
			verdict            Verdict
			routingJSON        []byte
		)
		if err := rows.Scan(&modelID, &modelName, &modelRevision, &verdict, &routingJSON); err != nil {
			return Projection{}, fmt.Errorf("%w: scan durable model outcome", ErrUnavailable)
		}
		var routing ReplayRoutingContext
		if err := decodeStrictJSON(routingJSON, &routing); err != nil {
			return Projection{}, fmt.Errorf("%w: decode durable replay context", ErrUnavailable)
		}
		entry := ProjectionEntry{
			RecipeID: routing.RecipeID, RecipeName: routing.RecipeName,
			RecipeRevision: routing.RecipeRevision, DecisionID: routing.DecisionID,
			DecisionName: routing.DecisionName, DecisionTier: routing.DecisionTier,
			ModelID: modelID, ModelName: modelName, ModelRevision: modelRevision,
		}
		key := projectionEntryKey(entry)
		current := entries[key]
		if current == nil {
			current = &entry
			entries[key] = current
		}
		switch verdict {
		case VerdictGoodFit:
			current.GoodFitCount++
		case VerdictUnderpowered:
			current.UnderpoweredCount++
		case VerdictOverprovisioned:
			current.OverprovisionedCount++
		case VerdictFailed:
			current.FailedCount++
		default:
			return Projection{}, fmt.Errorf("%w: durable outcome verdict is invalid", ErrUnavailable)
		}
	}
	if err := rows.Err(); err != nil {
		return Projection{}, fmt.Errorf("%w: iterate durable model outcomes", ErrUnavailable)
	}
	keys := make([]string, 0, len(entries))
	for key := range entries {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	projection := Projection{
		Schema: ProjectionSchema, NamespaceID: namespaceID, Revision: revision,
		Entries: make([]ProjectionEntry, 0, len(keys)),
	}
	for _, key := range keys {
		projection.Entries = append(projection.Entries, *entries[key])
	}
	return projection, nil
}

func (repository *PostgresRepository) Stage(
	ctx context.Context,
	projection Projection,
	payload []byte,
	digest [sha256.Size]byte,
) error {
	result, err := repository.database.ExecContext(ctx, `INSERT INTO inference_outcome_projection_snapshots (
  namespace_id, revision, snapshot, snapshot_digest
) VALUES ($1,$2,$3,$4)
ON CONFLICT (namespace_id, revision) DO NOTHING`,
		projection.NamespaceID, projection.Revision, payload, digest[:],
	)
	if err != nil {
		return fmt.Errorf("%w: stage outcome learning projection", ErrUnavailable)
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("%w: inspect staged outcome learning projection", ErrUnavailable)
	}
	if rows == 1 {
		return nil
	}
	var existing []byte
	if err := repository.database.QueryRowContext(ctx, `SELECT snapshot_digest
FROM inference_outcome_projection_snapshots
WHERE namespace_id=$1 AND revision=$2`, projection.NamespaceID, projection.Revision).Scan(&existing); err != nil {
		return fmt.Errorf("%w: read staged outcome learning projection", ErrUnavailable)
	}
	if !bytes.Equal(existing, digest[:]) {
		return fmt.Errorf("%w: outcome projection revision has another digest", ErrUnavailable)
	}
	return nil
}

func (repository *PostgresRepository) MarkApplied(
	ctx context.Context,
	namespaceID string,
	revision int64,
	digest [sha256.Size]byte,
) error {
	transaction, err := repository.database.BeginTx(ctx, &sql.TxOptions{})
	if err != nil {
		return fmt.Errorf("%w: begin outcome projection acknowledgement", ErrUnavailable)
	}
	defer func() { _ = transaction.Rollback() }()
	result, err := transaction.ExecContext(ctx, `UPDATE inference_outcome_projection_heads
SET applied_revision=$2, applied_digest=$3, updated_at=clock_timestamp()
WHERE namespace_id=$1 AND desired_revision >= $2 AND applied_revision < $2`, namespaceID, revision, digest[:])
	if err != nil {
		return fmt.Errorf("%w: acknowledge outcome projection", ErrUnavailable)
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("%w: inspect outcome projection acknowledgement", ErrUnavailable)
	}
	if rows == 0 {
		var (
			appliedRevision int64
			appliedDigest   []byte
		)
		if err := transaction.QueryRowContext(ctx, `SELECT applied_revision, applied_digest
FROM inference_outcome_projection_heads WHERE namespace_id=$1 FOR SHARE`, namespaceID).Scan(
			&appliedRevision, &appliedDigest,
		); err != nil {
			return fmt.Errorf("%w: read outcome projection acknowledgement", ErrUnavailable)
		}
		if appliedRevision < revision || (appliedRevision == revision && !bytes.Equal(appliedDigest, digest[:])) {
			return fmt.Errorf("%w: outcome projection acknowledgement is inconsistent", ErrUnavailable)
		}
	}
	if _, err := transaction.ExecContext(ctx, `UPDATE inference_outcome_projection_outbox
SET state='applied', applied_at=clock_timestamp(), last_error=NULL
WHERE namespace_id=$1 AND desired_revision <= $2 AND state <> 'applied'`, namespaceID, revision); err != nil {
		return fmt.Errorf("%w: acknowledge outcome projection outbox", ErrUnavailable)
	}
	if err := transaction.Commit(); err != nil {
		return fmt.Errorf("%w: commit outcome projection acknowledgement", ErrUnavailable)
	}
	return nil
}

func projectionEntryKey(entry ProjectionEntry) string {
	return fmt.Sprintf("%s\x00%s\x00%020d\x00%s\x00%s\x00%010d\x00%s\x00%s\x00%020d",
		entry.RecipeID, entry.RecipeName, entry.RecipeRevision,
		entry.DecisionID, entry.DecisionName, entry.DecisionTier,
		entry.ModelID, entry.ModelName, entry.ModelRevision)
}

func decodeProjection(payload []byte, digest string) (Projection, error) {
	var projection Projection
	if err := decodeStrictJSON(payload, &projection); err != nil {
		return Projection{}, err
	}
	calculated := sha256.Sum256(payload)
	if hex.EncodeToString(calculated[:]) != digest {
		return Projection{}, errors.New("outcome projection digest mismatch")
	}
	projection.Digest = digest
	return projection, projection.Validate()
}
