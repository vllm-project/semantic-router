// Package training owns durable management state. Executors consume frozen requests
// through the internal attempt boundary; they do not own public resource identity.
package training

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
	c "github.com/vllm-project/semantic-router/src/semantic-router/pkg/trainingcontract"
)

var (
	ErrInvalid  = errors.New("invalid training request")
	ErrConflict = errors.New("training conflict")
	ErrNotFound = errors.New("training resource not found")
)

type Service struct {
	store     *workflowstore.Store
	directory string
}

func New(store *workflowstore.Store, directory string) (*Service, error) {
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return nil, err
	}
	return &Service{store: store, directory: directory}, nil
}

func (s *Service) update(ctx context.Context, fn func(*workflowstore.TrainingTx) error) error {
	return s.store.UpdateTraining(ctx, func(tx *workflowstore.TrainingTx) error {
		if err := auth.RevalidateContextIfPresent(ctx); err != nil {
			return err
		}
		return fn(tx)
	})
}

func metadata(prefix string) c.Metadata {
	return c.Metadata{SchemaVersion: c.Version, ID: prefix + "_" + uuid.NewString(), CreatedAt: time.Now().UTC()}
}
func invalid(err error) error { return fmt.Errorf("%w: %w", ErrInvalid, err) }
func record(kind, owner, id string, value any) (workflowstore.TrainingRecord, error) {
	body, err := json.Marshal(value)
	return workflowstore.TrainingRecord{Kind: kind, Owner: owner, ID: id, Body: body}, err
}

func insert(tx *workflowstore.TrainingTx, kind, owner, id string, value any) error {
	r, err := record(kind, owner, id, value)
	if err != nil {
		return err
	}
	return tx.Insert(r)
}

type reader interface {
	Get(string, string, string) (workflowstore.TrainingRecord, error)
}

func read[T any](tx reader, owner, kind, id string) (T, error) {
	var value T
	if err := c.ValidateHandle(id); err != nil {
		return value, invalid(err)
	}
	r, err := tx.Get(kind, id, owner)
	if errors.Is(err, sql.ErrNoRows) {
		return value, ErrNotFound
	}
	if err != nil {
		return value, err
	}
	err = json.Unmarshal(r.Body, &value)
	return value, err
}

func (s *Service) Get(ctx context.Context, owner, kind, id string) (json.RawMessage, error) {
	if err := c.ValidateHandle(id); err != nil {
		return nil, invalid(err)
	}
	r, err := s.store.GetTraining(ctx, kind, id, owner)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	return r.Body, err
}

func (s *Service) CreateAsset(ctx context.Context, owner string, spec c.DataAssetSpec) (c.DataAsset, error) {
	v := c.DataAsset{Metadata: metadata("asset"), DataAssetSpec: spec}
	if strings.TrimSpace(spec.Name) == "" {
		return v, invalid(errors.New("name is required"))
	}
	if err := c.ValidateTarget(spec.TargetContract); err != nil {
		return v, invalid(err)
	}
	err := s.update(ctx, func(tx *workflowstore.TrainingTx) error { return insert(tx, "data-assets", owner, v.ID, v) })
	return v, err
}

func (s *Service) CreateExperiment(ctx context.Context, owner string, spec c.ExperimentSpec) (c.Experiment, error) {
	v := c.Experiment{Metadata: metadata("experiment"), ExperimentSpec: spec}
	if strings.TrimSpace(spec.Name) == "" {
		return v, invalid(errors.New("name is required"))
	}
	if err := c.ValidateTarget(spec.TargetContract); err != nil {
		return v, invalid(err)
	}
	err := s.update(ctx, func(tx *workflowstore.TrainingTx) error { return insert(tx, "experiments", owner, v.ID, v) })
	return v, err
}

func (s *Service) CreateSnapshot(ctx context.Context, owner string, spec c.SnapshotSpec) (c.DataSnapshot, error) {
	v := c.DataSnapshot{Metadata: metadata("snapshot"), SnapshotSpec: spec}
	if err := c.ValidateProfile(spec.Profile); err != nil {
		return v, invalid(err)
	}
	if spec.Source != nil {
		if err := c.ValidateModel(*spec.Source); err != nil {
			return v, invalid(err)
		}
	}
	for _, step := range spec.Preprocessing {
		if err := c.ValidateComponent(step); err != nil {
			return v, invalid(err)
		}
	}
	err := s.update(ctx, func(tx *workflowstore.TrainingTx) error {
		asset, err := read[c.DataAsset](tx, owner, "data-assets", spec.AssetID)
		if err != nil {
			return err
		}
		if asset.TargetContract != spec.Profile.TargetContract {
			return invalid(errors.New("asset/profile target mismatch"))
		}
		upload, err := read[c.Upload](tx, owner, "uploads", spec.UploadHandle)
		if err != nil {
			return err
		}
		v.Content = upload.File
		return insert(tx, "data-snapshots", owner, v.ID, v)
	})
	return v, err
}

func validateRunResources(tx *workflowstore.TrainingTx, owner string, spec c.RunSpec) error {
	experiment, err := read[c.Experiment](tx, owner, "experiments", spec.ExperimentID)
	if err != nil {
		return err
	}
	snapshot, err := read[c.DataSnapshot](tx, owner, "data-snapshots", spec.SnapshotID)
	if err != nil {
		return err
	}
	if spec.TargetContract != experiment.TargetContract || spec.TargetContract != snapshot.Profile.TargetContract {
		return invalid(errors.New("run/experiment/snapshot target mismatch"))
	}
	return nil
}

func (s *Service) Validate(ctx context.Context, owner string, req c.ValidationRequest) (c.ValidationResponse, error) {
	if req.SchemaVersion != c.Version {
		return c.ValidationResponse{}, invalid(errors.New("unsupported schema_version"))
	}
	if err := c.ValidateRunSpec(req.Spec); err != nil {
		return c.ValidationResponse{}, invalid(err)
	}
	err := s.update(ctx, func(tx *workflowstore.TrainingTx) error { return validateRunResources(tx, owner, req.Spec) })
	return c.ValidationResponse{Valid: err == nil}, err
}

func (s *Service) Submit(ctx context.Context, owner string, req c.SubmitRunRequest) (c.RunGraph, error) {
	var g c.RunGraph
	if err := c.ValidateRun(req); err != nil {
		return g, invalid(err)
	}
	body, err := json.Marshal(req.Spec)
	if err != nil {
		return g, invalid(err)
	}
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(body))
	err = s.update(ctx, func(tx *workflowstore.TrainingTx) error {
		existing, lookupErr := tx.FindSubmission(owner, req.IdempotencyKey)
		if lookupErr != nil {
			return lookupErr
		}
		if existing != nil {
			if existing.Digest != digest {
				return fmt.Errorf("%w: idempotency key already used with a different spec", ErrConflict)
			}
			g, lookupErr = read[c.RunGraph](tx, owner, "runs", existing.RunID)
			return lookupErr
		}
		if validationErr := validateRunResources(tx, owner, req.Spec); validationErr != nil {
			return validationErr
		}
		m := metadata("run")
		g = c.RunGraph{Run: c.TrainingRun{Metadata: m, Spec: req.Spec, Status: c.Pending, UpdatedAt: m.CreatedAt}, Tasks: []c.RunTask{}, Outputs: c.RunOutputs{ArtifactIDs: []string{}, EvaluationIDs: []string{}, QualificationIDs: []string{}}}
		for _, spec := range req.Spec.Tasks {
			g.Tasks = append(g.Tasks, c.RunTask{Metadata: metadata("task"), RunID: m.ID, Spec: spec, Status: c.Pending, Attempts: []c.Attempt{}})
		}
		if insertErr := insert(tx, "runs", owner, m.ID, g); insertErr != nil {
			return insertErr
		}
		if submissionErr := tx.Submit(workflowstore.TrainingSubmission{Owner: owner, Key: req.IdempotencyKey, Digest: digest, RunID: m.ID}); submissionErr != nil {
			return submissionErr
		}
		return tx.Event(c.Event{RunID: m.ID, Status: c.Pending, RecordedAt: m.CreatedAt})
	})
	return g, err
}

func (s *Service) ListRuns(ctx context.Context, owner, experimentID string) ([]c.TrainingRun, error) {
	if _, err := s.Get(ctx, owner, "experiments", experimentID); err != nil {
		return nil, err
	}
	records, err := s.store.ListTraining(ctx, "runs", owner)
	if err != nil {
		return nil, err
	}
	out := []c.TrainingRun{}
	for _, r := range records {
		var g c.RunGraph
		if err := json.Unmarshal(r.Body, &g); err != nil {
			return nil, err
		}
		if g.Run.Spec.ExperimentID == experimentID {
			out = append(out, g.Run)
		}
	}
	return out, nil
}

func (s *Service) Events(ctx context.Context, owner, runID string, after int64) (c.EventPage, error) {
	if after < 0 {
		return c.EventPage{}, invalid(errors.New("after must be nonnegative"))
	}
	if _, err := s.Get(ctx, owner, "runs", runID); err != nil {
		return c.EventPage{}, err
	}
	return s.store.TrainingEvents(ctx, runID, after)
}
