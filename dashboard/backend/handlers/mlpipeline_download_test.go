package handlers

import (
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/dashboard/backend/mlpipeline"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

func TestMLPipelineDownloadOutputIndex(t *testing.T) {
	handler, files := newMLDownloadFixture(t)
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	client := server.Client()
	client.Timeout = 5 * time.Second

	for _, test := range []struct {
		name   string
		suffix string
		status int
		index  int
	}{
		{name: "omitted", status: http.StatusOK},
		{name: "omitted-with-trailing-slash", suffix: "/", status: http.StatusOK},
		{name: "zero", suffix: "/0", status: http.StatusOK},
		{name: "last-file", suffix: "/1", status: http.StatusOK, index: 1},
		{name: "leading-zero", suffix: "/01", status: http.StatusOK, index: 1},
		{name: "positive-sign", suffix: "/+1", status: http.StatusOK, index: 1},
		{name: "negative-zero", suffix: "/-0", status: http.StatusOK},
		{name: "escaped-digit", suffix: "/%31", status: http.StatusOK, index: 1},
		{name: "query-does-not-change-index", suffix: "/1?index=0", status: http.StatusOK, index: 1},
		{name: "negative", suffix: "/-1", status: http.StatusBadRequest},
		{name: "minimum-int64", suffix: "/-9223372036854775808", status: http.StatusBadRequest},
		{name: "one-past-end", suffix: "/2", status: http.StatusBadRequest},
		{name: "maximum-int64", suffix: "/9223372036854775807", status: http.StatusBadRequest},
		{name: "positive-overflow", suffix: "/9223372036854775808", status: http.StatusBadRequest},
		{name: "negative-overflow", suffix: "/-9223372036854775809", status: http.StatusBadRequest},
		{name: "long-overflow", suffix: "/" + strings.Repeat("9", 256), status: http.StatusBadRequest},
		{name: "not-numeric", suffix: "/invalid", status: http.StatusBadRequest},
		{name: "zero-prefix", suffix: "/0invalid", status: http.StatusBadRequest},
		{name: "nonzero-prefix", suffix: "/1invalid", status: http.StatusBadRequest},
		{name: "decimal", suffix: "/1.0", status: http.StatusBadRequest},
		{name: "exponent", suffix: "/1e0", status: http.StatusBadRequest},
		{name: "hexadecimal", suffix: "/0x1", status: http.StatusBadRequest},
		{name: "binary", suffix: "/0b1", status: http.StatusBadRequest},
		{name: "digit-separator", suffix: "/1_0", status: http.StatusBadRequest},
		{name: "sign-only", suffix: "/+", status: http.StatusBadRequest},
		{name: "leading-space", suffix: "/%201", status: http.StatusBadRequest},
		{name: "trailing-space", suffix: "/1%20", status: http.StatusBadRequest},
		{name: "tab", suffix: "/%091", status: http.StatusBadRequest},
		{name: "newline", suffix: "/1%0A", status: http.StatusBadRequest},
		{name: "nul", suffix: "/1%00", status: http.StatusBadRequest},
		{name: "unicode-digit", suffix: "/%D9%A1", status: http.StatusBadRequest},
		{name: "extra-segment", suffix: "/1/extra", status: http.StatusBadRequest},
		{name: "empty-index-with-extra-segment", suffix: "//1", status: http.StatusBadRequest},
		{name: "trailing-index-segment", suffix: "/1/", status: http.StatusBadRequest},
		{name: "escaped-slash", suffix: "/1%2F0", status: http.StatusBadRequest},
	} {
		t.Run(test.name, func(t *testing.T) {
			response, err := client.Get(server.URL + "/api/ml-pipeline/download/download-job" + test.suffix)
			require.NoError(t, err, "invalid indices must return an HTTP error, not terminate the connection")
			defer response.Body.Close()
			body, err := io.ReadAll(response.Body)
			require.NoError(t, err)
			require.Equal(t, test.status, response.StatusCode, "body: %s", body)
			if test.status != http.StatusOK {
				require.Empty(t, response.Header.Get("Content-Disposition"))
				for _, file := range files {
					require.NotContains(t, string(body), file.content)
				}
				return
			}
			file := files[test.index]
			require.Equal(t, file.content, string(body))
			require.Equal(t, file.contentType, response.Header.Get("Content-Type"))
			require.Equal(t, "attachment; filename="+file.name, response.Header.Get("Content-Disposition"))
		})
	}
}

func TestMLPipelineDownloadOutputErrors(t *testing.T) {
	handler, _ := newMLDownloadFixture(t)
	for _, test := range []struct {
		name   string
		method string
		path   string
		status int
	}{
		{name: "missing-job-id", method: http.MethodGet, path: "/api/ml-pipeline/download/", status: http.StatusBadRequest},
		{name: "unknown-job", method: http.MethodGet, path: "/api/ml-pipeline/download/unknown/0", status: http.StatusNotFound},
		{name: "no-output-files", method: http.MethodGet, path: "/api/ml-pipeline/download/empty-job/0", status: http.StatusNotFound},
		{name: "missing-output-file", method: http.MethodGet, path: "/api/ml-pipeline/download/missing-file-job/0", status: http.StatusNotFound},
		{name: "wrong-method", method: http.MethodPost, path: "/api/ml-pipeline/download/download-job/0", status: http.StatusMethodNotAllowed},
	} {
		t.Run(test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			handler(response, httptest.NewRequest(test.method, test.path, nil))
			require.Equal(t, test.status, response.Code, "body: %s", response.Body.String())
		})
	}
}

func FuzzMLPipelineDownloadOutputIndex(f *testing.F) {
	handler, files := newMLDownloadFixture(f)
	for _, index := range []string{"", "0", "1", "+1", "01", "-0", "-1", "2", "invalid", "1suffix", "1/0", "9223372036854775808"} {
		f.Add(index)
	}
	f.Fuzz(func(t *testing.T, index string) {
		path := "/api/ml-pipeline/download/download-job"
		if index != "" {
			path += "/" + url.PathEscape(index)
		}
		response := httptest.NewRecorder()
		handler(response, httptest.NewRequest(http.MethodGet, path, nil))
		value, err := strconv.Atoi(index)
		if index == "" || (err == nil && value >= 0 && value < len(files)) {
			require.Equal(t, http.StatusOK, response.Code)
			require.Equal(t, files[value].content, response.Body.String())
		} else {
			require.Equal(t, http.StatusBadRequest, response.Code)
			require.Empty(t, response.Header().Get("Content-Disposition"))
		}
	})
}

type mlDownloadFile struct {
	name        string
	content     string
	contentType string
}

func newMLDownloadFixture(t testing.TB) (http.HandlerFunc, []mlDownloadFile) {
	t.Helper()
	root := t.TempDir()
	store, err := workflowstore.Open(filepath.Join(root, "workflow.sqlite"))
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, store.Close()) })
	files := []mlDownloadFile{
		{name: "first.yaml", content: "model: first-fixture\n", contentType: "text/yaml"},
		{name: "second.json", content: "{\"model\":\"second-fixture\"}\n", contentType: "application/json"},
	}
	paths := make([]string, len(files))
	for index, file := range files {
		paths[index] = filepath.Join(root, file.name)
		require.NoError(t, os.WriteFile(paths[index], []byte(file.content), 0o600))
	}
	for _, job := range []workflowstore.MLJobRecord{
		{ID: "download-job", OutputFiles: paths},
		{ID: "empty-job"},
		{ID: "missing-file-job", OutputFiles: []string{filepath.Join(root, "missing.yaml")}},
	} {
		job.Type = "config"
		job.Status = string(mlpipeline.StatusCompleted)
		job.CreatedAt = time.Now()
		job.CompletedAt = job.CreatedAt
		require.NoError(t, store.PutMLJob(job))
	}
	runner, err := mlpipeline.NewRunner(mlpipeline.RunnerConfig{DataDir: root, Workflow: store})
	require.NoError(t, err)
	handler := &MLPipelineHandler{runner: runner}
	return handler.DownloadOutputHandler(), files
}
