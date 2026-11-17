package training

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
	c "github.com/vllm-project/semantic-router/src/semantic-router/pkg/trainingcontract"
)

// Complete publishes outputs with the task outcome in one transaction. Invalid
// worker reports fail the attempt; repeating a final report preserves output IDs.
func (s *Service) Complete(ctx context.Context, owner, runID, attemptID string, result c.WorkerResult) (c.RunGraph, error) {
	body, err := json.Marshal(result)
	if err != nil {
		return c.RunGraph{}, invalid(err)
	}
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(body))
	return s.mutate(ctx, owner, runID, func(tx *workflowstore.TrainingTx, g *c.RunGraph) error {
		belongs := false
		for _, t := range g.Tasks {
			for _, a := range t.Attempts {
				belongs = belongs || a.ID == attemptID
			}
		}
		if !belongs {
			return ErrNotFound
		}
		previous, err := read[string](tx, owner, "attempt-results", attemptID)
		if err == nil {
			if previous == digest {
				return nil
			}
			return fmt.Errorf("%w: attempt already has a different result", ErrConflict)
		}
		if !errors.Is(err, ErrNotFound) {
			return err
		}
		task, err := activeTask(g, attemptID)
		if err != nil {
			return err
		}
		records, out, err := outputs(tx, owner, g, task, result)
		if err != nil {
			if !errors.Is(err, ErrInvalid) && !errors.Is(err, ErrNotFound) {
				return err
			}
			result = c.WorkerResult{SchemaVersion: c.Version, Status: c.Failed, Diagnostic: "worker result rejected: " + err.Error()}
			records = nil
		}
		if result.Status == c.Running {
			return nil
		}
		for _, r := range records {
			if err := tx.Insert(r); err != nil {
				return err
			}
		}
		if result.Status == c.Succeeded {
			g.Outputs = out
		}
		a := &task.Attempts[len(task.Attempts)-1]
		now := time.Now().UTC()
		a.Status = result.Status
		a.FinishedAt = &now
		a.Diagnostic = result.Diagnostic
		task.Status = result.Status
		if err := insert(tx, "attempt-results", owner, attemptID, digest); err != nil {
			return err
		}
		if err := save(tx, owner, g, task.ID, result.Diagnostic); err != nil {
			return err
		}
		return settle(tx, owner, g)
	})
}

func verifyFile(tx *workflowstore.TrainingTx, owner, attempt string, file c.File) error {
	saved, err := read[ownedFile](tx, owner, "files", file.Handle)
	if err != nil {
		return err
	}
	if saved.File != file || saved.AttemptID != attempt {
		return invalid(errors.New("file metadata or producing attempt mismatch"))
	}
	return nil
}

func outputs(tx *workflowstore.TrainingTx, owner string, g *c.RunGraph, task *c.RunTask, result c.WorkerResult) ([]workflowstore.TrainingRecord, c.RunOutputs, error) {
	out := c.RunOutputs{ArtifactIDs: append([]string{}, g.Outputs.ArtifactIDs...), EvaluationIDs: append([]string{}, g.Outputs.EvaluationIDs...), QualificationIDs: append([]string{}, g.Outputs.QualificationIDs...)}
	reject := func(message string) ([]workflowstore.TrainingRecord, c.RunOutputs, error) {
		return nil, out, invalid(errors.New(message))
	}
	if result.SchemaVersion != c.Version {
		return reject("unsupported worker schema_version")
	}
	if result.Status != c.Running && result.Status != c.Succeeded && result.Status != c.Failed && result.Status != c.Cancelled {
		return reject("invalid worker status")
	}
	if result.Status != c.Succeeded && (len(result.Artifacts) > 0 || len(result.Evaluations) > 0 || len(result.Qualifications) > 0) {
		return reject("only successful attempts may publish outputs")
	}
	// Classifier qualification must retain existing Python provenance checks.
	// Qualification adapters and binding proposals follow in a separate increment.
	if len(result.Qualifications) > 0 {
		return reject("qualification publication requires a provenance-validating adapter")
	}
	if result.Status != c.Succeeded {
		return nil, out, nil
	}
	snapshot, err := read[c.DataSnapshot](tx, owner, "data-snapshots", g.Run.Spec.SnapshotID)
	if err != nil {
		return nil, out, err
	}
	attempt := task.Attempts[len(task.Attempts)-1].ID
	provenance := c.Provenance{RunID: g.Run.ID, TaskID: task.ID, AttemptID: attempt, SnapshotID: g.Run.Spec.SnapshotID, Trainer: g.Run.Spec.Trainer}
	records := []workflowstore.TrainingRecord{}
	add := func(kind, id string, value any) error {
		r, err := record(kind, owner, id, value)
		if err != nil {
			return invalid(err)
		}
		records = append(records, r)
		return nil
	}
	for _, result := range result.Artifacts {
		if err := c.ValidateProfile(result.Profile); err != nil {
			return nil, out, invalid(err)
		}
		if !reflect.DeepEqual(result.Profile, snapshot.Profile) {
			return reject("artifact/snapshot profile mismatch")
		}
		if len(result.Variants) == 0 {
			return reject("artifact requires at least one variant")
		}
		p := provenance
		if result.ManifestBundle != nil {
			if err := verifyFile(tx, owner, attempt, *result.ManifestBundle); err != nil {
				return nil, out, err
			}
			p.ManifestBundle = result.ManifestBundle
		}
		artifact := c.Artifact{Metadata: metadata("artifact"), Profile: result.Profile, Provenance: p}
		if err := add("artifacts", artifact.ID, artifact); err != nil {
			return nil, out, err
		}
		variants := []c.ArtifactVariant{}
		for _, spec := range result.Variants {
			if err := c.ValidateVariant(spec); err != nil {
				return nil, out, invalid(err)
			}
			for _, file := range spec.Files {
				if err := verifyFile(tx, owner, attempt, file); err != nil {
					return nil, out, err
				}
			}
			variant := c.ArtifactVariant{Metadata: metadata("variant"), ArtifactID: artifact.ID, ArtifactVariantSpec: spec}
			if err := add("artifact-variants", variant.ID, variant); err != nil {
				return nil, out, err
			}
			variants = append(variants, variant)
		}
		if err := add("artifact-variant-index", variantIndexID(artifact.ID), variants); err != nil {
			return nil, out, err
		}
		out.ArtifactIDs = append(out.ArtifactIDs, artifact.ID)
	}
	evaluationInputs := map[string]c.ArtifactVariant{}
	if len(result.Evaluations) > 0 {
		inputs, inputErr := dependencyInputs(tx, owner, g, task)
		if inputErr != nil {
			return nil, out, inputErr
		}
		for _, variant := range inputs {
			evaluationInputs[variant.ID] = variant
		}
	}
	for _, spec := range result.Evaluations {
		variant, ok := evaluationInputs[spec.VariantID]
		if !ok {
			return reject("evaluation variant must be a published dependency input of this run")
		}
		artifact, err := read[c.Artifact](tx, owner, "artifacts", variant.ArtifactID)
		if err != nil {
			return nil, out, err
		}
		evalSnapshot, err := read[c.DataSnapshot](tx, owner, "data-snapshots", spec.SnapshotID)
		if err != nil {
			return nil, out, err
		}
		if !reflect.DeepEqual(evalSnapshot.Profile, artifact.Profile) {
			return reject("evaluation profile mismatch")
		}
		if err := c.ValidateComponent(spec.Method); err != nil {
			return nil, out, invalid(err)
		}
		if len(spec.Metrics) == 0 {
			return reject("evaluation metrics are required")
		}
		for name := range spec.Metrics {
			if name == "" {
				return reject("metric name is required")
			}
		}
		evaluation := c.Evaluation{Metadata: metadata("evaluation"), Provenance: provenance, EvaluationSpec: spec}
		if err := add("evaluations", evaluation.ID, evaluation); err != nil {
			return nil, out, err
		}
		out.EvaluationIDs = append(out.EvaluationIDs, evaluation.ID)
	}
	return records, out, nil
}

func (s *Service) Variants(ctx context.Context, owner, artifactID string) ([]c.ArtifactVariant, error) {
	var out []c.ArtifactVariant
	err := s.store.UpdateTraining(ctx, func(tx *workflowstore.TrainingTx) error {
		if _, err := read[c.Artifact](tx, owner, "artifacts", artifactID); err != nil {
			return err
		}
		var err error
		out, err = read[[]c.ArtifactVariant](tx, owner, "artifact-variant-index", variantIndexID(artifactID))
		return err
	})
	return out, err
}

func variantIndexID(artifactID string) string {
	return "index_" + strings.TrimPrefix(artifactID, "artifact_")
}
