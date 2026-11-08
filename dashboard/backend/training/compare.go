package training

import (
	"context"
	"errors"
	"reflect"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
	c "github.com/vllm-project/semantic-router/src/semantic-router/pkg/trainingcontract"
)

func (s *Service) Compare(ctx context.Context, owner string, req c.ComparisonRequest) (c.ComparisonResponse, error) {
	out := c.ComparisonResponse{Entries: []c.ComparisonEntry{}}
	if len(req.RunIDs) < 2 {
		return out, invalid(errors.New("at least two unique runs are required"))
	}
	seen := map[string]bool{}
	var profile c.Profile
	err := s.store.UpdateTraining(ctx, func(tx *workflowstore.TrainingTx) error {
		for i, id := range req.RunIDs {
			if seen[id] {
				return invalid(errors.New("duplicate run ID"))
			}
			seen[id] = true
			g, err := read[c.RunGraph](tx, owner, "runs", id)
			if err != nil {
				return err
			}
			snapshot, err := read[c.DataSnapshot](tx, owner, "data-snapshots", g.Run.Spec.SnapshotID)
			if err != nil {
				return err
			}
			if i == 0 {
				profile = snapshot.Profile
			}
			if !reflect.DeepEqual(profile, snapshot.Profile) {
				return invalid(errors.New("comparison target or profile mismatch"))
			}
			entry := c.ComparisonEntry{Run: g.Run, Evaluations: []c.Evaluation{}}
			for _, id := range g.Outputs.EvaluationIDs {
				evaluation, err := read[c.Evaluation](tx, owner, "evaluations", id)
				if err != nil {
					return err
				}
				entry.Evaluations = append(entry.Evaluations, evaluation)
			}
			out.Entries = append(out.Entries, entry)
		}
		return nil
	})
	return out, err
}
