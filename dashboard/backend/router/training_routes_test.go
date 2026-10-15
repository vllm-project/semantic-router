package router

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/training"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
	c "github.com/vllm-project/semantic-router/src/semantic-router/pkg/trainingcontract"
)

type trainingHTTP struct {
	handler http.Handler
	service *training.Service
	store   *workflowstore.Store
}

func openTrainingHTTP(t *testing.T, root string, authService *auth.Service) trainingHTTP {
	t.Helper()
	store, err := workflowstore.Open(filepath.Join(root, "workflow.db"))
	if err != nil {
		t.Fatal(err)
	}
	service, err := training.New(store, filepath.Join(root, "files"))
	if err != nil {
		t.Fatal(err)
	}
	mux := auth.NewPolicyMux()
	registerTrainingRoutes(mux, service)
	mux.Seal()
	for _, contract := range mux.Contracts() {
		if err := auth.ValidateRouteContract(contract); err != nil {
			t.Fatal(err)
		}
		for _, policy := range contract.Policies {
			if policy.Public || policy.Permission != auth.PermMlPipeline || policy.AuditMode != auth.AuditRequired {
				t.Fatalf("training policy: %+v", policy)
			}
		}
	}
	return trainingHTTP{handler: handlers.TrainingErrorEnvelope(wrapWithAuth(mux, authService, mux)), service: service, store: store}
}

func trainingRequest(t *testing.T, h http.Handler, token, method, path string, value any, status int) *httptest.ResponseRecorder {
	t.Helper()
	body := []byte{}
	if raw, ok := value.([]byte); ok {
		body = raw
	} else if value != nil {
		var err error
		body, err = json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
	}
	req := httptest.NewRequest(method, "/api/training/v2"+path, bytes.NewReader(body))
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	req.Header.Set("Content-Type", "application/json")
	if _, ok := value.([]byte); ok && path == "/uploads" {
		req.Header.Set("Content-Type", "application/octet-stream")
	}
	response := httptest.NewRecorder()
	h.ServeHTTP(response, req)
	if response.Code != status {
		t.Fatalf("%s %s status=%d want=%d body=%s", method, path, response.Code, status, response.Body.String())
	}
	if status >= 400 {
		var err c.APIError
		if decodeErr := json.Unmarshal(response.Body.Bytes(), &err); decodeErr != nil || err.Code == "" || err.Message == "" {
			t.Fatalf("missing APIError: %s", response.Body.String())
		}
	}
	return response
}

func trainingDecode[T any](t *testing.T, response *httptest.ResponseRecorder) T {
	t.Helper()
	var value T
	if err := json.Unmarshal(response.Body.Bytes(), &value); err != nil {
		t.Fatal(err)
	}
	return value
}

func trainingTokens(t *testing.T) (*auth.Service, map[string]string) {
	t.Helper()
	store, err := auth.NewStore(filepath.Join(t.TempDir(), "auth.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	svc := auth.NewService(store, "training-test-secret", 12)
	hash, err := svc.HashPassword("training-test-password")
	if err != nil {
		t.Fatal(err)
	}
	tokens := map[string]string{}
	for _, name := range []string{"alice", "bob", "reader"} {
		role := auth.RoleWrite
		if name == "reader" {
			role = auth.RoleRead
		}
		if _, err := store.CreateUser(t.Context(), name+"@example.com", name, hash, role, "active"); err != nil {
			t.Fatal(err)
		}
		token, _, err := svc.Login(t.Context(), name+"@example.com", "training-test-password")
		if err != nil {
			t.Fatal(err)
		}
		tokens[name] = token
	}
	return svc, tokens
}

func TestTrainingHTTPFixtureLifecycleSurvivesRestart(t *testing.T) {
	authService, tokens := trainingTokens(t)
	for _, name := range []string{"selector", "neural"} {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			server := openTrainingHTTP(t, root, authService)
			body, err := os.ReadFile("../../../src/semantic-router/pkg/trainingcontract/testdata/" + name + ".json")
			if err != nil {
				t.Fatal(err)
			}
			var f c.Fixture
			if operationErr := json.Unmarshal(body, &f); operationErr != nil {
				t.Fatal(operationErr)
			}
			token := tokens["alice"]
			upload := trainingDecode[c.Upload](t, trainingRequest(t, server.handler, token, "POST", "/uploads", []byte("dataset"), 201))
			asset := trainingDecode[c.DataAsset](t, trainingRequest(t, server.handler, token, "POST", "/data-assets", f.Asset.DataAssetSpec, 201))
			spec := f.Snapshot.SnapshotSpec
			spec.AssetID, spec.UploadHandle = asset.ID, upload.ID
			if spec.Profile.Classifier != nil {
				encoded, marshalErr := json.Marshal(spec)
				if marshalErr != nil {
					t.Fatal(marshalErr)
				}
				invalid := bytes.Replace(encoded, []byte(`"negative":0`), []byte(`"negative":null`), 1)
				if bytes.Equal(invalid, encoded) {
					t.Fatal("classifier fixture must include the zero-indexed negative label")
				}
				trainingRequest(t, server.handler, token, "POST", "/data-snapshots", invalid, 400)
			}
			snapshot := trainingDecode[c.DataSnapshot](t, trainingRequest(t, server.handler, token, "POST", "/data-snapshots", spec, 201))
			experiment := trainingDecode[c.Experiment](t, trainingRequest(t, server.handler, token, "POST", "/experiments", f.Experiment.ExperimentSpec, 201))
			req := f.Submit
			req.Spec.SnapshotID, req.Spec.ExperimentID = snapshot.ID, experiment.ID
			trainingRequest(t, server.handler, token, "POST", "/runs/validate", c.ValidationRequest{SchemaVersion: c.Version, Spec: req.Spec}, 200)
			g := trainingDecode[c.RunGraph](t, trainingRequest(t, server.handler, token, "POST", "/runs", req, 202))
			if g.Run.Status != c.Pending || g.Outputs.ArtifactIDs == nil {
				t.Fatal("submitted graph not canonical")
			}
			trainingRequest(t, server.handler, tokens["bob"], "GET", "/runs/"+g.Run.ID, nil, 404)
			trainingRequest(t, server.handler, tokens["bob"], "GET", "/files/"+upload.Handle, nil, 404)
			runs := trainingDecode[[]c.TrainingRun](t, trainingRequest(t, server.handler, token, "GET", "/runs?experiment_id="+experiment.ID, nil, 200))
			if len(runs) != 1 || runs[0].ID != g.Run.ID {
				t.Fatalf("run listing=%+v", runs)
			}
			if operationErr := server.store.Close(); operationErr != nil {
				t.Fatal(operationErr)
			}
			server = openTrainingHTTP(t, root, authService)
			defer func() { _ = server.store.Close() }()
			replay := trainingDecode[c.RunGraph](t, trainingRequest(t, server.handler, token, "POST", "/runs", req, 202))
			if replay.Run.ID != g.Run.ID {
				t.Fatal("restart duplicated run")
			}
			trainingRequest(t, server.handler, token, "POST", "/runs/"+g.Run.ID+"/retry", nil, 409)
			cancelled := trainingDecode[c.RunGraph](t, trainingRequest(t, server.handler, token, "POST", "/runs/"+g.Run.ID+"/cancel", nil, 202))
			if cancelled.Run.Status != c.Cancelled {
				t.Fatalf("pending cancellation=%s", cancelled.Run.Status)
			}
			trainingRequest(t, server.handler, token, "POST", "/runs/"+g.Run.ID+"/cancel", nil, 202)
			trainingRequest(t, server.handler, token, "POST", "/runs/"+g.Run.ID+"/retry", nil, 202)
			trainingRequest(t, server.handler, token, "GET", "/runs/"+g.Run.ID+"/events?after=bad", nil, 400)
			page := trainingDecode[c.EventPage](t, trainingRequest(t, server.handler, token, "GET", "/runs/"+g.Run.ID+"/events", nil, 200))
			if len(page.Events) < 4 {
				t.Fatalf("persisted events=%+v", page)
			}
			downloaded := trainingRequest(t, server.handler, token, "GET", "/files/"+upload.Handle, nil, 200)
			if downloaded.Body.String() != "dataset" {
				t.Fatal("upload bytes changed")
			}
			if want := fmt.Sprintf("sha256:%x", sha256.Sum256(downloaded.Body.Bytes())); upload.Digest != want {
				t.Fatalf("upload digest=%s want=%s", upload.Digest, want)
			}
			req.Spec.Parameters = map[string]any{"changed": true}
			trainingRequest(t, server.handler, token, "POST", "/runs", req, 409)
			savedSnapshot := trainingDecode[c.DataSnapshot](t, trainingRequest(t, server.handler, token, "GET", "/data-snapshots/"+snapshot.ID, nil, 200))
			if savedSnapshot.Content != upload.File {
				t.Fatal("snapshot did not preserve the uploaded file metadata after restart")
			}
			trainingRequest(t, server.handler, token, "PUT", "/data-snapshots/"+snapshot.ID, spec, 405)
			// Publish through the internal mock-worker boundary, then discover everything
			// through the management API using only the original run ID.
			claims, err := authService.ParseToken(token)
			if err != nil {
				t.Fatal(err)
			}
			work, err := server.service.StartAttempt(t.Context(), claims.UserID, g.Run.ID, g.Tasks[0].ID)
			if err != nil {
				t.Fatal(err)
			}
			output, err := server.service.PutOutput(t.Context(), claims.UserID, g.Run.ID, work.AttemptID, strings.NewReader("model"))
			if err != nil {
				t.Fatal(err)
			}
			_, err = server.service.Complete(t.Context(), claims.UserID, g.Run.ID, work.AttemptID, c.WorkerResult{SchemaVersion: c.Version, Status: c.Succeeded, Artifacts: []c.ArtifactResult{{Profile: snapshot.Profile, Variants: []c.ArtifactVariantSpec{{Format: c.Component{Name: "mock", Version: "v1"}, Files: map[string]c.File{"model.bin": output}}}}}})
			if err != nil {
				t.Fatal(err)
			}
			published := trainingDecode[c.RunGraph](t, trainingRequest(t, server.handler, token, "GET", "/runs/"+g.Run.ID, nil, 200))
			artifact := trainingDecode[c.Artifact](t, trainingRequest(t, server.handler, token, "GET", "/artifacts/"+published.Outputs.ArtifactIDs[0], nil, 200))
			variants := trainingDecode[[]c.ArtifactVariant](t, trainingRequest(t, server.handler, token, "GET", "/artifacts/"+artifact.ID+"/variants", nil, 200))
			trainingRequest(t, server.handler, token, "GET", "/artifact-variants/"+variants[0].ID, nil, 200)
			download := trainingRequest(t, server.handler, token, "GET", "/files/"+variants[0].Files["model.bin"].Handle, nil, 200)
			if download.Body.String() != "model" {
				t.Fatal("artifact download changed bytes")
			}
			publishedFile := variants[0].Files["model.bin"]
			if want := fmt.Sprintf("sha256:%x", sha256.Sum256(download.Body.Bytes())); publishedFile.Digest != want || publishedFile != output {
				t.Fatalf("published file=%+v want output=%+v digest=%s", publishedFile, output, want)
			}
			trainingRequest(t, server.handler, tokens["bob"], "GET", "/artifacts/"+artifact.ID, nil, 404)
			// Complete the remaining mock tasks, cancelling just before the final
			// success arrives. Retry must settle without rerunning successful work.
			completed := published
			for i := 1; i < len(published.Tasks); i++ {
				work, err := server.service.StartAttempt(t.Context(), claims.UserID, g.Run.ID, published.Tasks[i].ID)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(work.Inputs, variants) {
					t.Fatalf("task %s did not receive the published model: %+v", published.Tasks[i].Spec.Key, work.Inputs)
				}
				result := c.WorkerResult{SchemaVersion: c.Version, Status: c.Succeeded}
				if i == 1 {
					result.Evaluations = []c.EvaluationSpec{{VariantID: variants[0].ID, SnapshotID: snapshot.ID, Method: c.Component{Name: "accuracy", Version: "v1"}, Metrics: map[string]float64{"accuracy": 0.9}}}
				}
				if i == len(published.Tasks)-1 {
					trainingRequest(t, server.handler, token, "POST", "/runs/"+g.Run.ID+"/cancel", nil, 202)
				}
				completed, err = server.service.Complete(t.Context(), claims.UserID, g.Run.ID, work.AttemptID, result)
				if err != nil {
					t.Fatal(err)
				}
			}
			if completed.Run.Status != c.Cancelled {
				t.Fatalf("late success ignored cancellation: %s", completed.Run.Status)
			}
			if len(completed.Outputs.EvaluationIDs) != 1 {
				t.Fatalf("evaluation output missing: %+v", completed.Outputs)
			}
			evaluation := trainingDecode[c.Evaluation](t, trainingRequest(t, server.handler, token, "GET", "/evaluations/"+completed.Outputs.EvaluationIDs[0], nil, 200))
			if evaluation.VariantID != variants[0].ID || evaluation.Provenance.TaskID != published.Tasks[1].ID {
				t.Fatalf("evaluation did not preserve its model reference: %+v", evaluation)
			}
			retried := trainingDecode[c.RunGraph](t, trainingRequest(t, server.handler, token, "POST", "/runs/"+g.Run.ID+"/retry", nil, 202))
			if retried.Run.Status != c.Succeeded || !reflect.DeepEqual(retried.Tasks, completed.Tasks) || !reflect.DeepEqual(retried.Outputs, completed.Outputs) {
				t.Fatalf("retry did not retain completed work: %+v", retried)
			}
			saved := trainingDecode[c.RunGraph](t, trainingRequest(t, server.handler, token, "GET", "/runs/"+g.Run.ID, nil, 200))
			if !reflect.DeepEqual(saved, retried) {
				t.Fatal("retry response and persisted graph differ")
			}
			trainingRequest(t, server.handler, token, "POST", "/runs/"+g.Run.ID+"/retry", nil, 409)
		})
	}
}

func TestTrainingHTTPRejectsBoundaryViolations(t *testing.T) {
	svc, tokens := trainingTokens(t)
	server := openTrainingHTTP(t, t.TempDir(), svc)
	defer func() { _ = server.store.Close() }()
	for _, test := range []struct {
		token, method, path string
		body                any
		status              int
	}{
		{"", "POST", "/data-assets", c.DataAssetSpec{Name: "data", TargetContract: c.Selector}, 401},
		{tokens["reader"], "GET", "/runs/run_example", nil, 403},
		{tokens["alice"], "POST", "/data-assets", []byte(`{"name":"data","target_contract":"selector.model-choice/v1","owner":"bob"}`), 400},
		{tokens["alice"], "POST", "/data-assets", []byte(`{"name":"data","target_contract":"selector.model-choice/v1"} {}`), 400},
		{tokens["alice"], "POST", "/data-assets", []byte(`null`), 400},
		{tokens["alice"], "POST", "/runs", []byte(`{"schema_version":"semantic-router.training/v2","idempotency_key":"example","spec":{"snapshot_id":"/tmp/data"}}`), 400},
		{tokens["alice"], "POST", "/uploads", c.DataAssetSpec{}, 400},
		{tokens["alice"], "GET", "/runs?experiment_id=/tmp/data", nil, 400},
		{tokens["alice"], "POST", "/binding-proposals", nil, 403},
	} {
		trainingRequest(t, server.handler, test.token, test.method, test.path, test.body, test.status)
	}
	// Shared middleware's early Content-Length rejection still uses APIError.
	req := httptest.NewRequest("POST", "/api/training/v2/uploads", strings.NewReader("small"))
	req.Header.Set("Authorization", "Bearer "+tokens["alice"])
	req.Header.Set("Content-Type", "application/octet-stream")
	req.ContentLength = training.MaxFileBytes + 1
	response := httptest.NewRecorder()
	server.handler.ServeHTTP(response, req)
	value := trainingDecode[c.APIError](t, response)
	if response.Code != 413 || value.Code != "payload_too_large" {
		t.Fatalf("early limit=%d %+v", response.Code, value)
	}
}
