package handlers

import (
	"bytes"
	"context"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/mlpipeline"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

type pausedMLBody struct {
	io.Reader
	once    sync.Once
	entered chan struct{}
	release chan struct{}
}

func (body *pausedMLBody) Read(data []byte) (int, error) {
	body.once.Do(func() {
		close(body.entered)
		<-body.release
	})
	return body.Reader.Read(data)
}

func (body *pausedMLBody) Close() error { return nil }

func TestMLPipelineRevocationDuringBodyPreventsJobSubmission(t *testing.T) {
	for _, test := range []struct {
		name string
		path string
	}{
		{"benchmark", "/api/ml-pipeline/benchmark"},
		{"train", "/api/ml-pipeline/train"},
		{"config", "/api/ml-pipeline/config"},
	} {
		t.Run(test.name, func(t *testing.T) {
			root := t.TempDir()
			uploads := filepath.Join(root, "uploads")
			if err := os.Mkdir(uploads, 0o700); err != nil {
				t.Fatal(err)
			}
			t.Setenv("TMPDIR", uploads)
			store, err := workflowstore.Open(filepath.Join(root, "workflow.sqlite"), workflowstore.Options{})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = store.Close() })
			var outbound atomic.Int32
			sidecar := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				outbound.Add(1)
				w.WriteHeader(http.StatusNoContent)
			}))
			t.Cleanup(sidecar.Close)
			runner, err := mlpipeline.NewRunner(mlpipeline.RunnerConfig{
				DataDir: filepath.Join(root, "data"), MLServiceURL: sidecar.URL, Workflow: store,
			})
			if err != nil {
				t.Fatal(err)
			}
			handler := &MLPipelineHandler{runner: runner}
			var endpoint http.HandlerFunc
			var payload bytes.Buffer
			contentType := "application/json"
			switch test.name {
			case "benchmark", "train":
				writer := multipart.NewWriter(&payload)
				fields := []string{"training_data"}
				if test.name == "benchmark" {
					fields = []string{"models_yaml", "queries_jsonl"}
					endpoint = handler.RunBenchmarkHandler()
				} else {
					endpoint = handler.RunTrainHandler()
				}
				for _, field := range fields {
					part, partErr := writer.CreateFormFile(field, field+".json")
					if partErr != nil {
						t.Fatal(partErr)
					}
					if _, writeErr := io.WriteString(part, `{}`); writeErr != nil {
						t.Fatal(writeErr)
					}
				}
				if closeErr := writer.Close(); closeErr != nil {
					t.Fatal(closeErr)
				}
				contentType = writer.FormDataContentType()
			case "config":
				endpoint = handler.GenerateConfigHandler()
				payload.WriteString(`{}`)
			}
			body := &pausedMLBody{Reader: bytes.NewReader(payload.Bytes()), entered: make(chan struct{}), release: make(chan struct{})}
			request := httptest.NewRequest(http.MethodPost, test.path, body)
			request.Header.Set("Content-Type", contentType)
			var allowed atomic.Bool
			allowed.Store(true)
			request = request.WithContext(auth.WithPermissionRevalidator(request.Context(), func(context.Context) error {
				if !allowed.Load() {
					return auth.ErrPermissionDenied
				}
				return nil
			}))
			response := httptest.NewRecorder()
			finished := make(chan struct{})
			go func() { endpoint.ServeHTTP(response, request); close(finished) }()
			select {
			case <-body.entered:
			case <-time.After(5 * time.Second):
				close(body.release)
				t.Fatal("handler did not reach body-read barrier")
			}
			allowed.Store(false)
			close(body.release)
			select {
			case <-finished:
			case <-time.After(5 * time.Second):
				t.Fatal("handler did not finish after body release")
			}
			if response.Code != http.StatusForbidden {
				t.Fatalf("status = %d, want 403: %s", response.Code, response.Body.String())
			}
			if jobs := runner.ListJobs(); len(jobs) != 0 {
				t.Fatalf("revoked request submitted %d jobs", len(jobs))
			}
			if got := outbound.Load(); got != 0 {
				t.Fatalf("revoked request called ML sidecar %d times", got)
			}
			entries, err := os.ReadDir(uploads)
			if err != nil || len(entries) != 0 {
				t.Fatalf("rejected upload artifacts = %v, %v", entries, err)
			}
		})
	}
}
