package training

import (
	"errors"
	"slices"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
	c "github.com/vllm-project/semantic-router/src/semantic-router/pkg/trainingcontract"
)

// dependencyInputs follows outputs of direct dependencies, including the exact
// variant referenced by an evaluation. It does not traverse the ancestor DAG.
func dependencyInputs(tx *workflowstore.TrainingTx, owner string, g *c.RunGraph, task *c.RunTask) ([]c.ArtifactVariant, error) {
	dependencies := map[string]bool{}
	for _, dep := range g.Tasks {
		if slices.Contains(task.Spec.DependsOn, dep.Spec.Key) {
			dependencies[dep.ID] = true
		}
	}
	artifacts := map[string]c.Artifact{}
	for _, id := range g.Outputs.ArtifactIDs {
		if _, ok := artifacts[id]; ok {
			continue
		}
		artifact, err := read[c.Artifact](tx, owner, "artifacts", id)
		if err != nil {
			return nil, err
		}
		if artifact.ID != id || artifact.Provenance.RunID != g.Run.ID {
			return nil, invalid(errors.New("input artifact must belong to this run"))
		}
		artifacts[id] = artifact
	}
	type variantIndex struct {
		variants []c.ArtifactVariant
		ids      map[string]bool
	}
	indexes := map[string]variantIndex{}
	loadIndex := func(artifactID string) (variantIndex, error) {
		if index, ok := indexes[artifactID]; ok {
			return index, nil
		}
		variants, err := read[[]c.ArtifactVariant](tx, owner, "artifact-variant-index", variantIndexID(artifactID))
		if err != nil {
			return variantIndex{}, err
		}
		index := variantIndex{variants: variants, ids: map[string]bool{}}
		for _, variant := range variants {
			if variant.ArtifactID == artifactID {
				index.ids[variant.ID] = true
			}
		}
		indexes[artifactID] = index
		return index, nil
	}
	inputs := []c.ArtifactVariant{}
	seen := map[string]bool{}
	add := func(id string) error {
		if seen[id] {
			return nil
		}
		// The index is a publication reference, not a substitute for the owned record.
		variant, err := read[c.ArtifactVariant](tx, owner, "artifact-variants", id)
		if err != nil {
			return err
		}
		if _, ok := artifacts[variant.ArtifactID]; variant.ID != id || !ok {
			return invalid(errors.New("input variant must reference an artifact published by this run"))
		}
		index, err := loadIndex(variant.ArtifactID)
		if err != nil {
			return err
		}
		if !index.ids[id] {
			return invalid(errors.New("input variant must be published in its artifact's variant index"))
		}
		seen[id] = true
		inputs = append(inputs, variant)
		return nil
	}
	for _, id := range g.Outputs.ArtifactIDs {
		artifact := artifacts[id]
		if !dependencies[artifact.Provenance.TaskID] {
			continue
		}
		index, err := loadIndex(id)
		if err != nil {
			return nil, err
		}
		for _, variant := range index.variants {
			if variant.ArtifactID != artifact.ID {
				return nil, invalid(errors.New("input variant must belong to its published artifact"))
			}
			if err := add(variant.ID); err != nil {
				return nil, err
			}
		}
	}
	for _, id := range g.Outputs.EvaluationIDs {
		evaluation, err := read[c.Evaluation](tx, owner, "evaluations", id)
		if err != nil {
			return nil, err
		}
		if evaluation.ID != id || evaluation.Provenance.RunID != g.Run.ID {
			return nil, invalid(errors.New("input evaluation must belong to this run"))
		}
		if dependencies[evaluation.Provenance.TaskID] {
			if err := add(evaluation.VariantID); err != nil {
				return nil, err
			}
		}
	}
	return inputs, nil
}
