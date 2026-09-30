package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/dashboard/backend/mlpipeline"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

func TestMLPipelineUploadsArePrivateAndDistinct(t *testing.T) {
	tests := []struct {
		name   string
		fields map[string]string
	}{
		{
			name: "benchmark",
			fields: map[string]string{
				"models_yaml":   `{"models":[]}`,
				"queries_jsonl": "{\"query\":\"hello\"}\n",
			},
		},
		{
			name: "train",
			fields: map[string]string{
				"training_data": "{\"query\":\"hello\",\"model\":\"example\"}\n",
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := t.TempDir()
			uploads := filepath.Join(root, "uploads")
			require.NoError(t, os.Mkdir(uploads, 0o700))
			for _, variable := range []string{"TMPDIR", "TMP", "TEMP"} {
				t.Setenv(variable, uploads)
			}

			store, err := workflowstore.Open(filepath.Join(root, "workflow.sqlite"), workflowstore.Options{})
			require.NoError(t, err)
			t.Cleanup(func() { require.NoError(t, store.Close()) })
			type inputPaths struct {
				Models   string `json:"models_yaml_path"`
				Queries  string `json:"queries_path"`
				Training string `json:"data_file"`
			}
			received := make(chan inputPaths, 1)
			sidecar := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var paths inputPaths
				if decodeErr := json.NewDecoder(r.Body).Decode(&paths); decodeErr != nil {
					http.Error(w, decodeErr.Error(), http.StatusBadRequest)
					return
				}
				received <- paths
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = fmt.Fprint(w, "data: {\"done\":true,\"success\":false,\"percent\":100,\"message\":\"fixture stops before training\"}\n\n")
			}))
			t.Cleanup(sidecar.Close)
			runner, err := mlpipeline.NewRunner(mlpipeline.RunnerConfig{
				DataDir:      filepath.Join(root, "data"),
				MLServiceURL: sidecar.URL,
				Workflow:     store,
			})
			require.NoError(t, err)
			handler := &MLPipelineHandler{runner: runner}
			var endpoint http.HandlerFunc
			if test.name == "benchmark" {
				endpoint = handler.RunBenchmarkHandler()
			} else {
				endpoint = handler.RunTrainHandler()
			}
			server := httptest.NewServer(endpoint)
			t.Cleanup(server.Close)

			var body bytes.Buffer
			writer := multipart.NewWriter(&body)
			for field, content := range test.fields {
				part, partErr := writer.CreateFormFile(field, "inputs.json")
				require.NoError(t, partErr)
				_, writeErr := io.WriteString(part, content)
				require.NoError(t, writeErr)
			}
			require.NoError(t, writer.Close())
			request, err := http.NewRequest(http.MethodPost, server.URL, &body)
			require.NoError(t, err)
			request.Header.Set("Content-Type", writer.FormDataContentType())
			client := server.Client()
			client.Timeout = 5 * time.Second
			response, err := client.Do(request)
			require.NoError(t, err)
			t.Cleanup(func() { require.NoError(t, response.Body.Close()) })
			require.Equal(t, http.StatusCreated, response.StatusCode)
			var receipt struct {
				JobID string `json:"job_id"`
			}
			require.NoError(t, json.NewDecoder(response.Body).Decode(&receipt))
			require.NotEmpty(t, receipt.JobID)

			var paths inputPaths
			select {
			case paths = <-received:
			case <-time.After(5 * time.Second):
				t.Fatal("the actual runner did not send the uploaded paths")
			}
			require.Eventually(t, func() bool {
				events, err := runner.ListProgressEvents(receipt.JobID, 10)
				return err == nil && len(events) > 0 && events[len(events)-1].Step == "Failed"
			}, 5*time.Second, 10*time.Millisecond)

			actual := map[string]string{
				"models_yaml":   paths.Models,
				"queries_jsonl": paths.Queries,
				"training_data": paths.Training,
			}
			seen := make(map[string]bool)
			for field, expected := range test.fields {
				path := actual[field]
				require.NotEmpty(t, path)
				assert.False(t, seen[path], "different upload fields share one path")
				seen[path] = true
				content, err := os.ReadFile(path)
				require.NoError(t, err)
				assert.Equal(t, expected, string(content))
				assert.Equal(t, ".json", filepath.Ext(path))
				assert.NotEqual(t, "inputs.json", filepath.Base(path))
				file, err := os.Stat(path)
				require.NoError(t, err)
				directory, err := os.Stat(filepath.Dir(path))
				require.NoError(t, err)
				if runtime.GOOS != "windows" {
					assert.Zero(t, file.Mode().Perm()&0o077)
					assert.Zero(t, directory.Mode().Perm()&0o077)
				}
				t.Logf("field=%s preserved=%t file_mode=%o directory_mode=%o", field, string(content) == expected, file.Mode().Perm(), directory.Mode().Perm())
			}
		})
	}
}
