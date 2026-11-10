package training

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
	c "github.com/vllm-project/semantic-router/src/semantic-router/pkg/trainingcontract"
)

func publishInputModels(t *testing.T, s *Service, owner string, g c.RunGraph) (c.RunGraph, []c.ArtifactVariant) {
	t.Helper()
	work, err := s.StartAttempt(t.Context(), owner, g.Run.ID, g.Tasks[0].ID)
	if err != nil {
		t.Fatal(err)
	}
	file, err := s.PutOutput(t.Context(), owner, g.Run.ID, work.AttemptID, strings.NewReader("model"))
	if err != nil {
		t.Fatal(err)
	}
	result := c.WorkerResult{SchemaVersion: c.Version, Status: c.Succeeded, Artifacts: []c.ArtifactResult{{
		Profile: work.Snapshot.Profile,
		Variants: []c.ArtifactVariantSpec{
			{Format: c.Component{Name: "mock", Version: "v1"}, Files: map[string]c.File{"model.bin": file}},
			{Format: c.Component{Name: "onnx", Version: "v1"}, Files: map[string]c.File{"model.onnx": file}},
		},
	}}}
	g, err = s.Complete(t.Context(), owner, g.Run.ID, work.AttemptID, result)
	if err != nil || len(g.Outputs.ArtifactIDs) != 1 {
		t.Fatalf("publish model: graph=%+v err=%v", g, err)
	}
	variants, err := s.Variants(t.Context(), owner, g.Outputs.ArtifactIDs[0])
	if err != nil || len(variants) != 2 {
		t.Fatalf("model variants=%+v err=%v", variants, err)
	}
	return g, variants
}

func TestDependencyInputsRejectInvalidReferences(t *testing.T) {
	cases := []struct {
		name        string
		change      func(*c.Fixture)
		foreignKind string
		missingKind string
		unindexed   bool
		direct      bool
		wantErr     error
	}{
		{name: "artifact from another run", change: func(f *c.Fixture) { f.Artifact.Provenance.RunID = "run_other" }, wantErr: ErrInvalid},
		{name: "evaluation from another run", change: func(f *c.Fixture) { f.Evaluation.Provenance.RunID = "run_other" }, wantErr: ErrInvalid},
		{name: "artifact from another owner", foreignKind: "artifacts", wantErr: ErrNotFound},
		{name: "evaluation from another owner", foreignKind: "evaluations", wantErr: ErrNotFound},
		{name: "variant from another owner", foreignKind: "artifact-variants", wantErr: ErrNotFound},
		{name: "index from another owner", foreignKind: "artifact-variant-index", wantErr: ErrNotFound},
		{name: "missing variant", missingKind: "artifact-variants", wantErr: ErrNotFound},
		{name: "unpublished artifact", change: func(f *c.Fixture) { f.Graph.Outputs.ArtifactIDs = nil }, wantErr: ErrInvalid},
		{name: "unpublished variant", unindexed: true, wantErr: ErrInvalid},
		{name: "variant belongs to another artifact", change: func(f *c.Fixture) { f.Variant.ArtifactID = "artifact_other" }, wantErr: ErrInvalid},
		{name: "direct artifact missing variant", direct: true, missingKind: "artifact-variants", wantErr: ErrNotFound},
		{name: "direct artifact variant from another owner", direct: true, foreignKind: "artifact-variants", wantErr: ErrNotFound},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, store := openService(t, t.TempDir())
			defer func() { _ = store.Close() }()
			f := fixture(t, "neural")
			if tc.change != nil {
				tc.change(&f)
			}
			variants := []c.ArtifactVariant{f.Variant}
			if tc.unindexed {
				variants = nil
			}
			// Inject inconsistent stored references to verify resolution checks
			// ownership and publication itself, rather than trusting the index.
			err := store.UpdateTraining(t.Context(), func(tx *workflowstore.TrainingTx) error {
				for _, value := range []struct {
					kind, id string
					body     any
				}{
					{"artifacts", f.Artifact.ID, f.Artifact},
					{"artifact-variants", f.Variant.ID, f.Variant},
					{"artifact-variant-index", variantIndexID(f.Artifact.ID), variants},
					{"evaluations", f.Evaluation.ID, f.Evaluation},
				} {
					if value.kind == tc.missingKind {
						continue
					}
					owner := "alice"
					if value.kind == tc.foreignKind {
						owner = "bob"
					}
					if insertErr := insert(tx, value.kind, owner, value.id, value.body); insertErr != nil {
						return insertErr
					}
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			err = store.UpdateTraining(t.Context(), func(tx *workflowstore.TrainingTx) error {
				task := &f.Graph.Tasks[2]
				if tc.direct {
					task = &f.Graph.Tasks[1]
				}
				inputs, resolveErr := dependencyInputs(tx, "alice", &f.Graph, task)
				if len(inputs) != 0 {
					t.Fatal("returned inputs with an invalid reference")
				}
				return resolveErr
			})
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("resolution error=%v want=%v", err, tc.wantErr)
			}
		})
	}
}

func TestDependencyInputsCombineOnlyDirectOutputs(t *testing.T) {
	for _, tc := range []struct {
		name              string
		directArtifact    bool
		evaluatedVariants []int
		wantVariants      []int
	}{
		{name: "no evaluation output"},
		{name: "artifact and evaluation share a variant", directArtifact: true, evaluatedVariants: []int{1}, wantVariants: []int{0, 1}},
		{name: "evaluations share an artifact index", evaluatedVariants: []int{1, 0, 1}, wantVariants: []int{1, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, store := openService(t, t.TempDir())
			defer func() { _ = store.Close() }()
			req := seed(t, s, "alice", fixture(t, "selector"))
			if tc.directArtifact {
				req.Spec.Tasks[2].DependsOn = []string{"train", "evaluate"}
			}
			g, err := s.Submit(t.Context(), "alice", req)
			if err != nil {
				t.Fatal(err)
			}
			g, variants := publishInputModels(t, s, "alice", g)
			work, err := s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[1].ID)
			if err != nil {
				t.Fatal(err)
			}
			result := c.WorkerResult{SchemaVersion: c.Version, Status: c.Succeeded}
			want := []c.ArtifactVariant{}
			for _, i := range tc.evaluatedVariants {
				result.Evaluations = append(result.Evaluations, c.EvaluationSpec{VariantID: variants[i].ID, SnapshotID: req.Spec.SnapshotID, Method: c.Component{Name: "accuracy", Version: "v1"}, Metrics: map[string]float64{"accuracy": 0.9}})
			}
			for _, i := range tc.wantVariants {
				want = append(want, variants[i])
			}
			if _, operationErr := s.Complete(t.Context(), "alice", g.Run.ID, work.AttemptID, result); operationErr != nil {
				t.Fatal(operationErr)
			}
			qualify, err := s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[2].ID)
			if err != nil || !reflect.DeepEqual(qualify.Inputs, want) {
				t.Fatalf("combined inputs=%+v want=%+v err=%v", qualify.Inputs, want, err)
			}
		})
	}
}

func TestEvaluationPublicationUsesResolvedInputs(t *testing.T) {
	s, store := openService(t, t.TempDir())
	defer func() { _ = store.Close() }()
	req := seed(t, s, "alice", fixture(t, "selector"))
	req.Spec.Tasks[2].Key = "reevaluate"
	req.Spec.Tasks[2].Executor = req.Spec.Tasks[1].Executor
	g, err := s.Submit(t.Context(), "alice", req)
	if err != nil {
		t.Fatal(err)
	}
	g, variants := publishInputModels(t, s, "alice", g)
	evaluate, err := s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[1].ID)
	if err != nil {
		t.Fatal(err)
	}
	result := c.WorkerResult{SchemaVersion: c.Version, Status: c.Succeeded, Evaluations: []c.EvaluationSpec{{VariantID: variants[1].ID, SnapshotID: req.Spec.SnapshotID, Method: c.Component{Name: "accuracy", Version: "v1"}, Metrics: map[string]float64{"accuracy": 0.9}}}}
	g, err = s.Complete(t.Context(), "alice", g.Run.ID, evaluate.AttemptID, result)
	if err != nil || g.Tasks[1].Status != c.Succeeded {
		t.Fatalf("first evaluation failed: %+v %v", g, err)
	}
	work, err := s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[2].ID)
	if err != nil {
		t.Fatal(err)
	}
	// The worker may report another evaluation, but only for the variant in its
	// resolved inputs, even if an unevaluated sibling belongs to this same run.
	result.Evaluations[0].VariantID = variants[0].ID
	g, err = s.Complete(t.Context(), "alice", g.Run.ID, work.AttemptID, result)
	if err != nil || g.Run.Status != c.Failed || len(g.Outputs.EvaluationIDs) != 1 {
		t.Fatalf("accepted non-input variant: %+v %v", g, err)
	}
	if _, operationErr := s.Retry(t.Context(), "alice", g.Run.ID); operationErr != nil {
		t.Fatal(operationErr)
	}
	work, err = s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[2].ID)
	if err != nil {
		t.Fatal(err)
	}
	result.Evaluations[0].VariantID = variants[1].ID
	g, err = s.Complete(t.Context(), "alice", g.Run.ID, work.AttemptID, result)
	if err != nil || g.Run.Status != c.Succeeded || len(g.Outputs.EvaluationIDs) != 2 {
		t.Fatalf("rejected resolved input variant: %+v %v", g, err)
	}
}

func TestDependencyInputsFollowEvaluationReferences(t *testing.T) {
	for _, name := range []string{"selector", "neural"} {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			s, store := openService(t, root)
			t.Cleanup(func() { _ = store.Close() })
			req := seed(t, s, "alice", fixture(t, name))
			g, err := s.Submit(t.Context(), "alice", req)
			if err != nil {
				t.Fatal(err)
			}
			g, variants := publishInputModels(t, s, "alice", g)
			work, err := s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[1].ID)
			if err != nil || !reflect.DeepEqual(work.Inputs, variants) {
				t.Fatalf("direct artifact inputs=%+v err=%v", work.Inputs, err)
			}
			evaluation := c.EvaluationSpec{VariantID: variants[1].ID, SnapshotID: req.Spec.SnapshotID, Method: c.Component{Name: "accuracy", Version: "v1"}, Metrics: map[string]float64{"accuracy": 0.9}}
			// Two evaluations reference only one of the model's variants. Qualify
			// must receive that variant once, without including unevaluated siblings.
			g, err = s.Complete(t.Context(), "alice", g.Run.ID, work.AttemptID, c.WorkerResult{SchemaVersion: c.Version, Status: c.Succeeded, Evaluations: []c.EvaluationSpec{evaluation, evaluation}})
			if err != nil || len(g.Outputs.EvaluationIDs) != 2 || g.Tasks[1].Status != c.Succeeded {
				t.Fatalf("evaluation publication=%+v err=%v", g, err)
			}
			if operationErr := store.Close(); operationErr != nil {
				t.Fatal(operationErr)
			}
			s, store = openService(t, root)
			qualify, err := s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[2].ID)
			if err != nil || !reflect.DeepEqual(qualify.Inputs, variants[1:]) {
				t.Fatalf("evaluation-referenced inputs=%+v want=%+v err=%v", qualify.Inputs, variants[1:], err)
			}
			if !reflect.DeepEqual(graph(t, s, "alice", g.Run.ID).Run.Spec, req.Spec) {
				t.Fatal("input resolution changed the frozen dependency graph")
			}
			repeated, err := s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[2].ID)
			if err != nil || !reflect.DeepEqual(repeated, qualify) {
				t.Fatalf("repeated dispatch changed inputs: %+v %v", repeated, err)
			}
			if operationErr := store.Close(); operationErr != nil {
				t.Fatal(operationErr)
			}
			s, store = openService(t, root)
			recovery, err := s.Recover(t.Context())
			if err != nil || len(recovery) != 1 || !reflect.DeepEqual(recovery[0].Request, qualify) {
				t.Fatalf("recovery changed referenced inputs: %+v %v", recovery, err)
			}
			if _, operationErr := s.Complete(t.Context(), "alice", g.Run.ID, qualify.AttemptID, c.WorkerResult{SchemaVersion: c.Version, Status: c.Failed}); operationErr != nil {
				t.Fatal(operationErr)
			}
			if _, operationErr := s.Retry(t.Context(), "alice", g.Run.ID); operationErr != nil {
				t.Fatal(operationErr)
			}
			retried, err := s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[2].ID)
			if err != nil || retried.AttemptID == qualify.AttemptID || !reflect.DeepEqual(retried.Inputs, qualify.Inputs) {
				t.Fatalf("retry changed referenced inputs: %+v %v", retried, err)
			}
		})
	}
}
