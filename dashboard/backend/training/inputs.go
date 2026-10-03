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
	inputs := []c.ArtifactVariant{}
	seen := map[string]bool{}
	add := func(id string) error {
		if seen[id] {
			return nil
		}
		variant, err := publishedInputVariant(tx, owner, g, id)
		if err != nil {
			return err
		}
		seen[id] = true
		inputs = append(inputs, variant)
		return nil
	}
	for _, id := range g.Outputs.ArtifactIDs {
		artifact, err := read[c.Artifact](tx, owner, "artifacts", id)
		if err != nil {
			return nil, err
		}
		if artifact.ID != id || artifact.Provenance.RunID != g.Run.ID {
			return nil, invalid(errors.New("input artifact must belong to this run"))
		}
		if !dependencies[artifact.Provenance.TaskID] {
			continue
		}
		variants, err := read[[]c.ArtifactVariant](tx, owner, "artifact-variant-index", variantIndexID(id))
		if err != nil {
			return nil, err
		}
		for _, variant := range variants {
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

func publishedInputVariant(tx *workflowstore.TrainingTx, owner string, g *c.RunGraph, id string) (c.ArtifactVariant, error) {
	variant, err := read[c.ArtifactVariant](tx, owner, "artifact-variants", id)
	if err != nil {
		return c.ArtifactVariant{}, err
	}
	if variant.ID != id || !slices.Contains(g.Outputs.ArtifactIDs, variant.ArtifactID) {
		return c.ArtifactVariant{}, invalid(errors.New("input variant must reference an artifact published by this run"))
	}
	artifact, err := read[c.Artifact](tx, owner, "artifacts", variant.ArtifactID)
	if err != nil {
		return c.ArtifactVariant{}, err
	}
	if artifact.ID != variant.ArtifactID || artifact.Provenance.RunID != g.Run.ID {
		return c.ArtifactVariant{}, invalid(errors.New("input artifact must belong to this run"))
	}
	variants, err := read[[]c.ArtifactVariant](tx, owner, "artifact-variant-index", variantIndexID(artifact.ID))
	if err != nil {
		return c.ArtifactVariant{}, err
	}
	if !slices.ContainsFunc(variants, func(v c.ArtifactVariant) bool { return v.ID == id && v.ArtifactID == artifact.ID }) {
		return c.ArtifactVariant{}, invalid(errors.New("input variant must be published in its artifact's variant index"))
	}
	return variant, nil
}
