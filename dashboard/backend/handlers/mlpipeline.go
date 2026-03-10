package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/mlpipeline"
)

// MLPipelineHandler holds dependencies for ML pipeline endpoints.
type MLPipelineHandler struct {
	runner       *mlpipeline.Runner
	sseClients   sync.Map // map[jobID]*sync.Map (clientID -> chan ProgressUpdate)
	lastProgress sync.Map // map[jobID]mlpipeline.ProgressUpdate — last known progress per job
}

// NewMLPipelineHandler creates a new ML pipeline handler.
func NewMLPipelineHandler(runner *mlpipeline.Runner) *MLPipelineHandler {
	h := &MLPipelineHandler{
		runner: runner,
	}
	// Forward progress updates to SSE clients
	go h.forwardProgressUpdates()
	return h
}

// forwardProgressUpdates forwards progress updates from the runner to SSE clients.
func (h *MLPipelineHandler) forwardProgressUpdates() {
	for update := range h.runner.ProgressUpdates() {
		h.broadcastProgress(update)
	}
}

// broadcastProgress sends a progress update to all subscribed clients for a job.
func (h *MLPipelineHandler) broadcastProgress(update mlpipeline.ProgressUpdate) {
	// Store last progress so new clients can catch up (skip 0% to avoid reset on reconnect)
	if update.Percent > 0 {
		h.lastProgress.Store(update.JobID, update)
	}

	if clientsMap, ok := h.sseClients.Load(update.JobID); ok {
		clients := clientsMap.(*sync.Map)
		clients.Range(func(key, value interface{}) bool {
			ch := value.(chan mlpipeline.ProgressUpdate)
			select {
			case ch <- update:
			default:
			}
			return true
		})
	}
}

// ListJobsHandler returns all ML pipeline jobs.
func (h *MLPipelineHandler) ListJobsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		jobs := h.runner.ListJobs()
		if jobs == nil {
			jobs = []*mlpipeline.Job{}
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(jobs); err != nil {
			log.Printf("Error encoding ML jobs response: %v", err)
		}
	}
}

// GetJobHandler returns a specific job by ID.
func (h *MLPipelineHandler) GetJobHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/ml-pipeline/jobs/"), "/")
		if len(pathParts) == 0 || pathParts[0] == "" {
			http.Error(w, "Job ID required", http.StatusBadRequest)
			return
		}
		jobID := pathParts[0]

		job := h.runner.GetJob(jobID)
		if job == nil {
			http.Error(w, "Job not found", http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(job); err != nil {
			log.Printf("Error encoding ML job response: %v", err)
		}
	}
}

// RunBenchmarkHandler starts a benchmark job (Layer 1).
func (h *MLPipelineHandler) RunBenchmarkHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Parse multipart form (models YAML + queries JSONL + config)
		if err := r.ParseMultipartForm(32 << 20); err != nil {
			http.Error(w, fmt.Sprintf("Failed to parse form: %v", err), http.StatusBadRequest)
			return
		}

		// Save uploaded files to job dir
		tempDir := filepath.Join(os.TempDir(), fmt.Sprintf("ml-bench-%d", time.Now().UnixMilli()))
		if err := os.MkdirAll(tempDir, 0o755); err != nil {
			http.Error(w, fmt.Sprintf("Failed to create temp dir: %v", err), http.StatusInternalServerError)
			return
		}

		modelsPath, err := saveUploadedFile(r, "models_yaml", tempDir)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to save models YAML: %v", err), http.StatusBadRequest)
			return
		}

		queriesPath, err := saveUploadedFile(r, "queries_jsonl", tempDir)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to save queries JSONL: %v", err), http.StatusBadRequest)
			return
		}

		// Parse config from form
		var req mlpipeline.BenchmarkRequest
		if configJSON := r.FormValue("config"); configJSON != "" {
			if unmarshalErr := json.Unmarshal([]byte(configJSON), &req); unmarshalErr != nil {
				http.Error(w, fmt.Sprintf("Invalid config JSON: %v", unmarshalErr), http.StatusBadRequest)
				return
			}
		}

		ctx := context.Background()
		jobID, err := h.runner.RunBenchmark(ctx, modelsPath, queriesPath, req)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to start benchmark: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"job_id": jobID,
			"status": "started",
		})
	}
}

// RunTrainHandler starts a training job (Layer 2).
// Supports two modes:
//   - JSON body: {"benchmark_job_id": "...", "config": {...}} — uses output from a previous benchmark job
//   - Multipart form: file field "training_data" + form field "config" JSON — uses an uploaded JSONL file directly
func (h *MLPipelineHandler) RunTrainHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var benchmarkDataPath string
		var trainConfig mlpipeline.TrainRequest

		contentType := r.Header.Get("Content-Type")

		if strings.HasPrefix(contentType, "multipart/form-data") {
			// ── Multipart upload mode: user uploads a training data file directly ──
			if err := r.ParseMultipartForm(64 << 20); err != nil {
				http.Error(w, fmt.Sprintf("Failed to parse form: %v", err), http.StatusBadRequest)
				return
			}

			// Save the uploaded training data file
			tempDir := filepath.Join(os.TempDir(), fmt.Sprintf("ml-train-upload-%d", time.Now().UnixMilli()))
			if err := os.MkdirAll(tempDir, 0o755); err != nil {
				http.Error(w, fmt.Sprintf("Failed to create temp dir: %v", err), http.StatusInternalServerError)
				return
			}

			uploadedPath, err := saveUploadedFile(r, "training_data", tempDir)
			if err != nil {
				http.Error(w, fmt.Sprintf("Failed to save training data file: %v", err), http.StatusBadRequest)
				return
			}
			benchmarkDataPath = uploadedPath

			// Parse config from form field
			if configJSON := r.FormValue("config"); configJSON != "" {
				if err := json.Unmarshal([]byte(configJSON), &trainConfig); err != nil {
					http.Error(w, fmt.Sprintf("Invalid config JSON: %v", err), http.StatusBadRequest)
					return
				}
			}
		} else {
			// ── JSON body mode: reference a previous benchmark job ──
			var body struct {
				BenchmarkDataPath string                  `json:"benchmark_data_path"`
				BenchmarkJobID    string                  `json:"benchmark_job_id"`
				Config            mlpipeline.TrainRequest `json:"config"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
				return
			}

			benchmarkDataPath = body.BenchmarkDataPath
			trainConfig = body.Config

			if benchmarkDataPath == "" && body.BenchmarkJobID != "" {
				benchJob := h.runner.GetJob(body.BenchmarkJobID)
				if benchJob != nil && len(benchJob.OutputFiles) > 0 {
					benchmarkDataPath = benchJob.OutputFiles[0]
				}
			}
		}

		if benchmarkDataPath == "" {
			http.Error(w, "Training data is required: upload a file (training_data) or provide benchmark_job_id/benchmark_data_path", http.StatusBadRequest)
			return
		}

		if len(trainConfig.Algorithms) == 0 {
			trainConfig.Algorithms = []string{"knn", "kmeans", "svm", "mlp"}
		}

		ctx := context.Background()
		jobID, err := h.runner.RunTrain(ctx, benchmarkDataPath, trainConfig)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to start training: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"job_id": jobID,
			"status": "started",
		})
	}
}

// GenerateConfigHandler generates deployment config (Layer 3).
func (h *MLPipelineHandler) GenerateConfigHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req mlpipeline.ConfigRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
			return
		}

		jobID, err := h.runner.GenerateConfig(req)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to generate config: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"job_id": jobID,
			"status": "started",
		})
	}
}

// DownloadOutputHandler serves a job's output file.
func (h *MLPipelineHandler) DownloadOutputHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/ml-pipeline/download/"), "/")
		if len(pathParts) == 0 || pathParts[0] == "" {
			http.Error(w, "Job ID required", http.StatusBadRequest)
			return
		}
		jobID := pathParts[0]

		job := h.runner.GetJob(jobID)
		if job == nil {
			http.Error(w, "Job not found", http.StatusNotFound)
			return
		}

		if len(job.OutputFiles) == 0 {
			http.Error(w, "No output files available", http.StatusNotFound)
			return
		}

		// Serve the first output file (or specific one if index is provided)
		fileIdx := 0
		if len(pathParts) > 1 && pathParts[1] != "" {
			_, _ = fmt.Sscanf(pathParts[1], "%d", &fileIdx)
		}
		if fileIdx >= len(job.OutputFiles) {
			http.Error(w, "File index out of range", http.StatusBadRequest)
			return
		}

		filePath := job.OutputFiles[fileIdx]
		filename := filepath.Base(filePath)

		w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%s", filename))
		if strings.HasSuffix(filename, ".yaml") || strings.HasSuffix(filename, ".yml") {
			w.Header().Set("Content-Type", "text/yaml")
		} else if strings.HasSuffix(filename, ".json") {
			w.Header().Set("Content-Type", "application/json")
		} else if strings.HasSuffix(filename, ".jsonl") {
			w.Header().Set("Content-Type", "application/x-jsonlines")
		} else {
			w.Header().Set("Content-Type", "application/octet-stream")
		}

		http.ServeFile(w, r, filePath)
	}
}

// StreamProgressHandler provides SSE for job progress updates.
func (h *MLPipelineHandler) StreamProgressHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/ml-pipeline/stream/"), "/")
		if len(pathParts) == 0 || pathParts[0] == "" {
			http.Error(w, "Job ID required", http.StatusBadRequest)
			return
		}
		jobID := pathParts[0]

		// Set SSE headers
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("Access-Control-Allow-Origin", "*")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		// Create client channel
		clientID := fmt.Sprintf("%d", time.Now().UnixNano())
		clientChan := make(chan mlpipeline.ProgressUpdate, 10)

		// Register client
		var clients *sync.Map
		if existing, ok := h.sseClients.Load(jobID); ok {
			clients = existing.(*sync.Map)
		} else {
			clients = &sync.Map{}
			h.sseClients.Store(jobID, clients)
		}
		clients.Store(clientID, clientChan)

		// Clean up on disconnect
		defer func() {
			clients.Delete(clientID)
			close(clientChan)
		}()

		// Send initial connection message
		_, _ = fmt.Fprintf(w, "event: connected\ndata: {\"job_id\":\"%s\"}\n\n", jobID)
		flusher.Flush()

		// Send last-known progress so client catches up on missed events
		if lastProg, ok := h.lastProgress.Load(jobID); ok {
			data, err := json.Marshal(lastProg.(mlpipeline.ProgressUpdate))
			if err == nil {
				_, _ = fmt.Fprintf(w, "event: progress\ndata: %s\n\n", data)
				flusher.Flush()
				// If the job already finished, send completed immediately
				if lastProg.(mlpipeline.ProgressUpdate).Percent >= 100 {
					_, _ = fmt.Fprintf(w, "event: completed\ndata: {\"job_id\":\"%s\"}\n\n", jobID)
					flusher.Flush()
					return
				}
			}
		}

		// Heartbeat ticker keeps the connection alive during long silent periods
		heartbeat := time.NewTicker(15 * time.Second)
		defer heartbeat.Stop()

		ctx := r.Context()
		for {
			select {
			case <-ctx.Done():
				return
			case <-heartbeat.C:
				// SSE comment line — keeps connection alive without triggering events
				_, _ = fmt.Fprintf(w, ": heartbeat\n\n")
				flusher.Flush()
			case update, ok := <-clientChan:
				if !ok {
					return
				}
				data, err := json.Marshal(update)
				if err != nil {
					log.Printf("Error marshaling progress update: %v", err)
					continue
				}
				_, _ = fmt.Fprintf(w, "event: progress\ndata: %s\n\n", data)
				flusher.Flush()

				if update.Percent >= 100 {
					_, _ = fmt.Fprintf(w, "event: completed\ndata: {\"job_id\":\"%s\"}\n\n", jobID)
					flusher.Flush()
					return
				}
			}
		}
	}
}

// saveUploadedFile saves a multipart file to the target directory and returns its path.
func saveUploadedFile(r *http.Request, fieldName, targetDir string) (string, error) {
	file, header, err := r.FormFile(fieldName)
	if err != nil {
		return "", fmt.Errorf("missing file field '%s': %w", fieldName, err)
	}
	defer file.Close()

	destPath := filepath.Join(targetDir, header.Filename)
	out, err := os.Create(destPath)
	if err != nil {
		return "", fmt.Errorf("failed to create file: %w", err)
	}
	defer out.Close()

	if _, err := io.Copy(out, file); err != nil {
		return "", fmt.Errorf("failed to write file: %w", err)
	}

	return destPath, nil
}
