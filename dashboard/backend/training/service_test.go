package training

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
	c "github.com/vllm-project/semantic-router/src/semantic-router/pkg/trainingcontract"
)

func openService(t *testing.T, root string) (*Service, *workflowstore.Store) {
	t.Helper()
	store, err := workflowstore.Open(filepath.Join(root, "workflow.db"))
	if err != nil {
		t.Fatal(err)
	}
	service, err := New(store, filepath.Join(root, "files"))
	if err != nil {
		t.Fatal(err)
	}
	return service, store
}

func fixture(t *testing.T, name string) c.Fixture {
	t.Helper()
	body, err := os.ReadFile("../../../src/semantic-router/pkg/trainingcontract/testdata/" + name + ".json")
	if err != nil {
		t.Fatal(err)
	}
	var f c.Fixture
	if err := json.Unmarshal(body, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func seed(t *testing.T, s *Service, owner string, f c.Fixture) c.SubmitRunRequest {
	t.Helper()
	ctx := t.Context()
	upload, err := s.Upload(ctx, owner, strings.NewReader("immutable dataset"))
	if err != nil {
		t.Fatal(err)
	}
	asset, err := s.CreateAsset(ctx, owner, f.Asset.DataAssetSpec)
	if err != nil {
		t.Fatal(err)
	}
	spec := f.Snapshot.SnapshotSpec
	spec.AssetID, spec.UploadHandle = asset.ID, upload.ID
	snapshot, err := s.CreateSnapshot(ctx, owner, spec)
	if err != nil {
		t.Fatal(err)
	}
	experiment, err := s.CreateExperiment(ctx, owner, f.Experiment.ExperimentSpec)
	if err != nil {
		t.Fatal(err)
	}
	req := f.Submit
	req.Spec.ExperimentID, req.Spec.SnapshotID = experiment.ID, snapshot.ID
	return req
}

func graph(t *testing.T, s *Service, owner, id string) c.RunGraph {
	t.Helper()
	body, err := s.Get(t.Context(), owner, "runs", id)
	if err != nil {
		t.Fatal(err)
	}
	var g c.RunGraph
	if err := json.Unmarshal(body, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func TestRestartRetainsAttemptsOutputsAndSnapshot(t *testing.T) {
	for _, name := range []string{"selector", "neural"} {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			s, store := openService(t, root)
			f := fixture(t, name)
			req := seed(t, s, "alice", f)
			req.Spec.Parameters["seed"] = int(9007199254740993)
			// Keep the fixture's target/trainer/base-model and use a small artifact -> eval DAG.
			req.Spec.Tasks = []c.TaskSpec{{Key: "train", Executor: req.Spec.Tasks[0].Executor}, {Key: "evaluate", DependsOn: []string{"train"}, Executor: req.Spec.Tasks[0].Executor}}
			g, err := s.Submit(t.Context(), "alice", req)
			if err != nil {
				t.Fatal(err)
			}
			if _, operationErr := s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[1].ID); !errors.Is(operationErr, ErrConflict) {
				t.Fatalf("dependency allowed: %v", operationErr)
			}
			request, err := s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[0].ID)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(request.Parameters, req.Spec.Parameters) {
				t.Fatalf("worker parameters=%#v want=%#v", request.Parameters, req.Spec.Parameters)
			}
			// Simulate a crash after dispatch but before storing the acknowledgement.
			if operationErr := store.Close(); operationErr != nil {
				t.Fatal(operationErr)
			}
			s, store = openService(t, root)
			recovery, err := s.Recover(t.Context())
			if err != nil || len(recovery) != 1 {
				t.Fatalf("recovery=%v err=%v", recovery, err)
			}
			if !reflect.DeepEqual(recovery[0].Request, request) || recovery[0].WorkerHandle != "" {
				t.Fatal("recovery changed frozen request or attempt ID")
			}
			if operationErr := s.Acknowledge(t.Context(), "alice", g.Run.ID, request.AttemptID, c.WorkerSubmission{SchemaVersion: c.Version, WorkerHandle: "worker_example"}); operationErr != nil {
				t.Fatal(operationErr)
			}
			file, err := s.PutOutput(t.Context(), "alice", g.Run.ID, request.AttemptID, strings.NewReader("model bytes"))
			if err != nil {
				t.Fatal(err)
			}
			result := c.WorkerResult{SchemaVersion: c.Version, Status: c.Succeeded, Artifacts: []c.ArtifactResult{{Profile: request.Snapshot.Profile, Variants: []c.ArtifactVariantSpec{{Format: c.Component{Name: "mock", Version: "v1"}, Files: map[string]c.File{"model.bin": file}}}}}}
			g, err = s.Complete(t.Context(), "alice", g.Run.ID, request.AttemptID, result)
			if err != nil || len(g.Outputs.ArtifactIDs) != 1 {
				t.Fatalf("completion=%+v err=%v", g, err)
			}
			repeated, err := s.Complete(t.Context(), "alice", g.Run.ID, request.AttemptID, result)
			if err != nil || !reflect.DeepEqual(g, repeated) {
				t.Fatalf("report replay changed output identity: %v", err)
			}
			evalRequest, err := s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[1].ID)
			if err != nil || len(evalRequest.Inputs) != 1 {
				t.Fatalf("evaluation inputs=%+v err=%v", evalRequest.Inputs, err)
			}
			if operationErr := store.Close(); operationErr != nil {
				t.Fatal(operationErr)
			}
			s, store = openService(t, root)
			defer func() { _ = store.Close() }()
			recovery, err = s.Recover(t.Context())
			if err != nil || len(recovery) != 1 || recovery[0].Request.AttemptID != evalRequest.AttemptID {
				t.Fatalf("evaluation recovery=%v err=%v", recovery, err)
			}
			g, err = s.Complete(t.Context(), "alice", g.Run.ID, evalRequest.AttemptID, c.WorkerResult{SchemaVersion: c.Version, Status: c.Succeeded, Evaluations: []c.EvaluationSpec{{VariantID: evalRequest.Inputs[0].ID, SnapshotID: req.Spec.SnapshotID, Method: c.Component{Name: "mock", Version: "v1"}, Metrics: map[string]float64{"accuracy": 0.9}}}})
			if err != nil || g.Run.Status != c.Succeeded || len(g.Outputs.EvaluationIDs) != 1 {
				t.Fatalf("evaluation completion=%+v err=%v", g, err)
			}
			again, err := s.Submit(t.Context(), "alice", req)
			if err != nil || again.Run.ID != g.Run.ID || again.Run.Status != c.Succeeded {
				t.Fatalf("restart idempotency: %v", err)
			}
			data, err := s.Download(t.Context(), "alice", file.Handle)
			if err != nil {
				t.Fatal(err)
			}
			body, err := io.ReadAll(data)
			_ = data.Close()
			if err != nil || string(body) != "model bytes" {
				t.Fatalf("download=%s err=%v", body, err)
			}
			page, err := s.Events(t.Context(), "alice", g.Run.ID, 0)
			if err != nil || len(page.Events) < 6 {
				t.Fatalf("events=%+v err=%v", page, err)
			}
			empty, err := s.Events(t.Context(), "alice", g.Run.ID, page.NextAfter)
			if err != nil || len(empty.Events) != 0 || empty.NextAfter != page.NextAfter {
				t.Fatalf("event cursor=%+v err=%v", empty, err)
			}
			if _, operationErr := s.Retry(t.Context(), "alice", g.Run.ID); !errors.Is(operationErr, ErrConflict) {
				t.Fatalf("successful retry allowed: %v", operationErr)
			}
			snapshotBody, err := s.Get(t.Context(), "alice", "data-snapshots", request.Snapshot.ID)
			if err != nil {
				t.Fatal(err)
			}
			var snapshot c.DataSnapshot
			if err := json.Unmarshal(snapshotBody, &snapshot); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(snapshot, request.Snapshot) {
				t.Fatal("snapshot changed")
			}
		})
	}
}

func TestConcurrentSubmissionAndRetryAcrossStores(t *testing.T) {
	root := t.TempDir()
	s, a := openService(t, root)
	defer func() { _ = a.Close() }()
	other, b := openService(t, root)
	defer func() { _ = b.Close() }()
	req := seed(t, s, "alice", fixture(t, "selector"))
	const count = 16
	ids := make(chan string, count)
	errs := make(chan error, count)
	var wg sync.WaitGroup
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			svc := s
			if i%2 == 0 {
				svc = other
			}
			g, err := svc.Submit(t.Context(), "alice", req)
			ids <- g.Run.ID
			errs <- err
		}(i)
	}
	wg.Wait()
	close(ids)
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	id := ""
	for value := range ids {
		if id == "" {
			id = value
		}
		if value != id {
			t.Fatal("duplicate run")
		}
	}
	if _, err := s.Cancel(t.Context(), "alice", id); err != nil {
		t.Fatal(err)
	}
	errs = make(chan error, count)
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			svc := s
			if i%2 == 0 {
				svc = other
			}
			_, err := svc.Retry(t.Context(), "alice", id)
			errs <- err
		}(i)
	}
	wg.Wait()
	close(errs)
	success := 0
	for err := range errs {
		if err == nil {
			success++
		} else if !errors.Is(err, ErrConflict) {
			t.Fatal(err)
		}
	}
	if success != 1 {
		t.Fatalf("successful retries=%d", success)
	}
	page, err := s.Events(t.Context(), "alice", id, 0)
	if err != nil {
		t.Fatal(err)
	}
	submissions := 0
	for _, event := range page.Events {
		if event.Diagnostic == "" && event.Status == c.Pending {
			submissions++
		}
	}
	if submissions != 1 {
		t.Fatalf("submission events=%d", submissions)
	}
	req.Spec.Parameters = map[string]any{"different": true}
	if _, err := s.Submit(t.Context(), "alice", req); !errors.Is(err, ErrConflict) {
		t.Fatalf("changed idempotency spec allowed: %v", err)
	}
}

func TestInvalidResultFailsAttemptAndRetryRetainsSuccess(t *testing.T) {
	s, store := openService(t, t.TempDir())
	defer func() { _ = store.Close() }()
	req := seed(t, s, "alice", fixture(t, "selector"))
	req.Spec.Tasks = []c.TaskSpec{{Key: "prepare", Executor: req.Spec.Tasks[0].Executor}, {Key: "train", DependsOn: []string{"prepare"}, Executor: req.Spec.Tasks[0].Executor}}
	g, err := s.Submit(t.Context(), "alice", req)
	if err != nil {
		t.Fatal(err)
	}
	first, err := s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[0].ID)
	if err != nil {
		t.Fatal(err)
	}
	file, err := s.PutOutput(t.Context(), "alice", g.Run.ID, first.AttemptID, strings.NewReader("prepared"))
	if err != nil {
		t.Fatal(err)
	}
	result := c.WorkerResult{SchemaVersion: c.Version, Status: c.Succeeded, Artifacts: []c.ArtifactResult{{Profile: first.Snapshot.Profile, Variants: []c.ArtifactVariantSpec{{Format: c.Component{Name: "data", Version: "v1"}, Files: map[string]c.File{"data.json": file}}}}}}
	g, err = s.Complete(t.Context(), "alice", g.Run.ID, first.AttemptID, result)
	if err != nil {
		t.Fatal(err)
	}
	second, err := s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[1].ID)
	if err != nil {
		t.Fatal(err)
	}
	// A successful report cannot claim a dependency's bytes as its own output.
	g, err = s.Complete(t.Context(), "alice", g.Run.ID, second.AttemptID, result)
	if err != nil || g.Run.Status != c.Failed || g.Tasks[1].Attempts[0].Diagnostic == "" {
		t.Fatalf("invalid final report left run stuck: %+v %v", g, err)
	}
	outputs := g.Outputs
	g, err = s.Retry(t.Context(), "alice", g.Run.ID)
	if err != nil {
		t.Fatal(err)
	}
	if g.Tasks[0].Status != c.Succeeded || !reflect.DeepEqual(outputs, g.Outputs) || len(g.Tasks[1].Attempts) != 1 {
		t.Fatal("retry erased prior success or history")
	}
	third, err := s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[1].ID)
	if err != nil {
		t.Fatal(err)
	}
	if third.AttemptID == second.AttemptID {
		t.Fatal("retry reused attempt identity")
	}
	g, err = s.Cancel(t.Context(), "alice", g.Run.ID)
	if err != nil || g.Run.Status != c.Cancelling {
		t.Fatalf("running cancellation=%+v %v", g, err)
	}
	recovery, err := s.Recover(t.Context())
	if err != nil || len(recovery) != 1 || !recovery[0].Cancel {
		t.Fatalf("cancellation recovery=%+v %v", recovery, err)
	}
	g, err = s.Complete(t.Context(), "alice", g.Run.ID, third.AttemptID, c.WorkerResult{SchemaVersion: c.Version, Status: c.Cancelled})
	if err != nil || g.Run.Status != c.Cancelled || len(g.Tasks[1].Attempts) != 2 {
		t.Fatalf("cancellation=%+v %v", g, err)
	}
}

type brokenReader struct{}

func (brokenReader) Read(p []byte) (int, error) { copy(p, "partial"); return 7, io.ErrUnexpectedEOF }

func TestOwnershipValidationAndUploadCleanup(t *testing.T) {
	root := t.TempDir()
	s, store := openService(t, root)
	defer func() { _ = store.Close() }()
	req := seed(t, s, "alice", fixture(t, "selector"))
	if _, err := s.Submit(t.Context(), "bob", req); !errors.Is(err, ErrNotFound) {
		t.Fatalf("cross-owner reference: %v", err)
	}
	req.Spec.SnapshotID = "/tmp/data.json"
	if _, err := s.Submit(t.Context(), "alice", req); !errors.Is(err, ErrInvalid) {
		t.Fatalf("raw path: %v", err)
	}
	before, _ := os.ReadDir(s.directory)
	if _, err := s.Upload(t.Context(), "alice", brokenReader{}); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("broken upload: %v", err)
	}
	revoked := auth.WithPermissionRevalidator(t.Context(), func(context.Context) error { return auth.ErrPermissionDenied })
	if _, err := s.Upload(revoked, "alice", strings.NewReader("revoked")); !errors.Is(err, auth.ErrPermissionDenied) {
		t.Fatalf("revoked upload: %v", err)
	}
	after, _ := os.ReadDir(s.directory)
	if len(before) != len(after) {
		t.Fatal("failed uploads left bytes")
	}
	if _, err := s.Get(t.Context(), "bob", "data-snapshots", req.Spec.SnapshotID); !errors.Is(err, ErrInvalid) {
		t.Fatal(err)
	}
}

func TestInvalidReportsPublishNothing(t *testing.T) {
	cases := []struct {
		name   string
		change func(*c.WorkerResult)
	}{
		{"schema", func(r *c.WorkerResult) { r.SchemaVersion = "other/v1" }},
		{"status", func(r *c.WorkerResult) { r.Status = c.Pending }},
		{"failure with outputs", func(r *c.WorkerResult) { r.Status = c.Failed }},
		{"profile", func(r *c.WorkerResult) { r.Artifacts[0].Profile.Selector.CandidateModels = []string{"different"} }},
		{"raw path", func(r *c.WorkerResult) {
			r.Artifacts[0].Variants[0].Files = map[string]c.File{"/tmp/model": r.Artifacts[0].Variants[0].Files["model.bin"]}
		}},
		{"size", func(r *c.WorkerResult) {
			f := r.Artifacts[0].Variants[0].Files["model.bin"]
			f.SizeBytes++
			r.Artifacts[0].Variants[0].Files["model.bin"] = f
		}},
		{"digest", func(r *c.WorkerResult) {
			f := r.Artifacts[0].Variants[0].Files["model.bin"]
			f.Digest = "sha256:" + strings.Repeat("0", 64)
			r.Artifacts[0].Variants[0].Files["model.bin"] = f
		}},
		{"missing digest", func(r *c.WorkerResult) {
			f := r.Artifacts[0].Variants[0].Files["model.bin"]
			f.Digest = ""
			r.Artifacts[0].Variants[0].Files["model.bin"] = f
		}},
		{"manifest digest", func(r *c.WorkerResult) {
			f := r.Artifacts[0].Variants[0].Files["model.bin"]
			f.Digest = "sha256:" + strings.Repeat("0", 64)
			r.Artifacts[0].ManifestBundle = &f
		}},
		{"unowned handle", func(r *c.WorkerResult) {
			f := r.Artifacts[0].Variants[0].Files["model.bin"]
			f.Handle = "file_missing"
			r.Artifacts[0].Variants[0].Files["model.bin"] = f
		}},
		{"qualification needs adapter", func(r *c.WorkerResult) {
			r.Qualifications = []c.QualificationSpec{{VariantID: "variant_example", Compatible: true}}
		}},
		{"partially valid output", func(r *c.WorkerResult) { r.Artifacts = append(r.Artifacts, c.ArtifactResult{}) }},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			s, store := openService(t, t.TempDir())
			defer func() { _ = store.Close() }()
			req := seed(t, s, "alice", fixture(t, "selector"))
			req.Spec.Tasks = req.Spec.Tasks[:1]
			g, err := s.Submit(t.Context(), "alice", req)
			if err != nil {
				t.Fatal(err)
			}
			worker, err := s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[0].ID)
			if err != nil {
				t.Fatal(err)
			}
			file, err := s.PutOutput(t.Context(), "alice", g.Run.ID, worker.AttemptID, strings.NewReader("model"))
			if err != nil {
				t.Fatal(err)
			}
			result := c.WorkerResult{SchemaVersion: c.Version, Status: c.Succeeded, Artifacts: []c.ArtifactResult{{Profile: worker.Snapshot.Profile, Variants: []c.ArtifactVariantSpec{{Format: c.Component{Name: "mock", Version: "v1"}, Files: map[string]c.File{"model.bin": file}}}}}}
			test.change(&result)
			g, err = s.Complete(t.Context(), "alice", g.Run.ID, worker.AttemptID, result)
			if err != nil || g.Run.Status != c.Failed || len(g.Outputs.ArtifactIDs) != 0 {
				t.Fatalf("invalid report=%+v err=%v", g, err)
			}
			records, err := store.ListTraining(t.Context(), "artifacts", "alice")
			if err != nil || len(records) != 0 {
				t.Fatalf("partial publication=%+v err=%v", records, err)
			}
			if _, err := s.Retry(t.Context(), "alice", g.Run.ID); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestDependencyFailurePropagatesWithUnorderedTasks(t *testing.T) {
	s, store := openService(t, t.TempDir())
	defer func() { _ = store.Close() }()
	req := seed(t, s, "alice", fixture(t, "selector"))
	executor := req.Spec.Tasks[0].Executor
	req.Spec.Tasks = []c.TaskSpec{{Key: "last", DependsOn: []string{"middle"}, Executor: executor}, {Key: "middle", DependsOn: []string{"first"}, Executor: executor}, {Key: "first", Executor: executor}}
	g, err := s.Submit(t.Context(), "alice", req)
	if err != nil {
		t.Fatal(err)
	}
	request, err := s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[2].ID)
	if err != nil {
		t.Fatal(err)
	}
	g, err = s.Complete(t.Context(), "alice", g.Run.ID, request.AttemptID, c.WorkerResult{SchemaVersion: c.Version, Status: c.Failed, Diagnostic: "executor failed"})
	if err != nil || g.Run.Status != c.Failed || g.Tasks[0].Status != c.Skipped || g.Tasks[1].Status != c.Skipped {
		t.Fatalf("dependency failure=%+v err=%v", g, err)
	}
}

func TestPublicationRollsBackWhenEventPersistenceFails(t *testing.T) {
	root := t.TempDir()
	s, store := openService(t, root)
	defer func() { _ = store.Close() }()
	db, err := sql.Open("sqlite3", filepath.Join(root, "workflow.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = db.Close() }()
	req := seed(t, s, "alice", fixture(t, "selector"))
	req.Spec.Tasks = req.Spec.Tasks[:1]
	g, err := s.Submit(t.Context(), "alice", req)
	if err != nil {
		t.Fatal(err)
	}
	work, err := s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[0].ID)
	if err != nil {
		t.Fatal(err)
	}
	file, err := s.PutOutput(t.Context(), "alice", g.Run.ID, work.AttemptID, strings.NewReader("model"))
	if err != nil {
		t.Fatal(err)
	}
	// Inject a storage failure after outputs and graph updates, before commit.
	_, err = db.Exec(`CREATE TRIGGER reject_training_event BEFORE INSERT ON training_events BEGIN SELECT RAISE(ABORT,'test persistence failure'); END`)
	if err != nil {
		t.Fatal(err)
	}
	result := c.WorkerResult{SchemaVersion: c.Version, Status: c.Succeeded, Artifacts: []c.ArtifactResult{{Profile: work.Snapshot.Profile, Variants: []c.ArtifactVariantSpec{{Format: c.Component{Name: "mock", Version: "v1"}, Files: map[string]c.File{"model.bin": file}}}}}}
	if _, operationErr := s.Complete(t.Context(), "alice", g.Run.ID, work.AttemptID, result); operationErr == nil {
		t.Fatal("injected failure was ignored")
	}
	saved := graph(t, s, "alice", g.Run.ID)
	artifacts, err := store.ListTraining(t.Context(), "artifacts", "alice")
	if err != nil || len(artifacts) != 0 || len(saved.Outputs.ArtifactIDs) != 0 || saved.Tasks[0].Status != c.Running {
		t.Fatalf("partial state survived failed transaction: %+v %v", saved, err)
	}
	if _, operationErr := db.Exec(`DROP TRIGGER reject_training_event`); operationErr != nil {
		t.Fatal(operationErr)
	}
	saved, err = s.Complete(t.Context(), "alice", g.Run.ID, work.AttemptID, result)
	if err != nil || saved.Run.Status != c.Succeeded || len(saved.Outputs.ArtifactIDs) != 1 {
		t.Fatalf("cannot recover failed commit: %+v %v", saved, err)
	}
}
