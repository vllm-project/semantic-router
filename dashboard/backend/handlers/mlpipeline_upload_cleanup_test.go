package handlers

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/dashboard/backend/mlpipeline"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

func TestMLPipelineRejectedUploadsAreRemoved(t *testing.T) {
	for _, test := range []struct {
		name       string
		benchmark  bool
		fields     []string
		config     string
		error      string
		status     int
		failStart  bool
		referenced bool
	}{
		{name: "benchmark-invalid-config", benchmark: true, fields: []string{"models_yaml", "queries_jsonl"}, config: "{", error: "Invalid config JSON", status: http.StatusBadRequest},
		{name: "benchmark-missing-queries", benchmark: true, fields: []string{"models_yaml"}, error: "Failed to save queries JSONL", status: http.StatusBadRequest},
		{name: "benchmark-missing-models", benchmark: true, error: "Failed to save models YAML", status: http.StatusBadRequest},
		{name: "train-invalid-config", fields: []string{"training_data"}, config: "{", error: "Invalid config JSON", status: http.StatusBadRequest},
		{name: "train-missing-data", error: "Failed to save training data file", status: http.StatusBadRequest},
		{name: "benchmark-start-fails", benchmark: true, fields: []string{"models_yaml", "queries_jsonl"}, failStart: true, error: "Failed to start benchmark", status: http.StatusInternalServerError},
		{name: "train-start-fails", fields: []string{"training_data"}, failStart: true, error: "Failed to start training", status: http.StatusInternalServerError},
		{name: "referenced-input-is-preserved", failStart: true, referenced: true, error: "Failed to start training", status: http.StatusInternalServerError},
	} {
		t.Run(test.name, func(t *testing.T) {
			root := t.TempDir()
			uploads := filepath.Join(root, "uploads")
			require.NoError(t, os.Mkdir(uploads, 0o700))
			for _, name := range []string{"TMPDIR", "TMP", "TEMP"} {
				t.Setenv(name, uploads)
			}
			handler := &MLPipelineHandler{}
			if test.failStart {
				store, storeErr := workflowstore.Open(filepath.Join(root, "workflow.sqlite"), workflowstore.Options{})
				require.NoError(t, storeErr)
				t.Cleanup(func() { require.NoError(t, store.Close()) })
				dataPath := filepath.Join(root, "not-a-directory")
				require.NoError(t, os.WriteFile(dataPath, []byte("existing file"), 0o600))
				runner, runnerErr := mlpipeline.NewRunner(mlpipeline.RunnerConfig{DataDir: dataPath, Workflow: store})
				require.NoError(t, runnerErr)
				handler.runner = runner
			}
			endpoint := handler.RunTrainHandler()
			if test.benchmark {
				endpoint = handler.RunBenchmarkHandler()
			}
			server := httptest.NewServer(endpoint)
			t.Cleanup(server.Close)
			var body bytes.Buffer
			var contentType string
			referencedPath := filepath.Join(root, "referenced.jsonl")
			if test.referenced {
				require.NoError(t, os.WriteFile(referencedPath, []byte("existing input"), 0o600))
				require.NoError(t, json.NewEncoder(&body).Encode(map[string]string{"benchmark_data_path": referencedPath}))
				contentType = "application/json"
			} else {
				writer := multipart.NewWriter(&body)
				for _, field := range test.fields {
					part, partErr := writer.CreateFormFile(field, "input.json")
					require.NoError(t, partErr)
					_, writeErr := io.WriteString(part, `{"input":"preserve only accepted jobs"}`)
					require.NoError(t, writeErr)
				}
				require.NoError(t, writer.WriteField("config", test.config))
				require.NoError(t, writer.Close())
				contentType = writer.FormDataContentType()
			}
			client := server.Client()
			client.Timeout = 5 * time.Second
			response, err := client.Post(server.URL, contentType, &body)
			require.NoError(t, err)
			defer response.Body.Close()
			result, err := io.ReadAll(response.Body)
			require.NoError(t, err)
			require.Equal(t, test.status, response.StatusCode)
			require.Contains(t, string(result), test.error)
			entries, err := os.ReadDir(uploads)
			require.NoError(t, err)
			t.Logf("http_status=%d retained_upload_directories=%d", response.StatusCode, len(entries))
			require.Empty(t, entries, "a rejected request retained its upload directory")
			if test.referenced {
				data, readErr := os.ReadFile(referencedPath)
				require.NoError(t, readErr)
				require.Equal(t, "existing input", string(data))
			}
		})
	}
}
