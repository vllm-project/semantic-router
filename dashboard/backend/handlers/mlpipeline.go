package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/mlpipeline"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

const (
	mlBenchmarkMultipartMemory  = 8 << 20
	mlBenchmarkBodyLimit        = 70 << 20
	mlModelsFileLimit           = 4 << 20
	mlQueriesFileLimit          = 64 << 20
	mlTrainingMultipartMemory   = 8 << 20
	mlTrainingBodyLimit         = 66 << 20
	mlTrainingFileLimit         = 64 << 20
	mlFormConfigLimit           = 64 << 10
	mlJSONBodyLimit             = 1 << 20
	mlSSEClientsPerJobLimit     = 16
	mlSSEClientsGlobalLimit     = 128
	mlTerminalProgressRetention = 512
)

// MLPipelineHandler holds dependencies for ML pipeline endpoints.
type MLPipelineHandler struct {
	runner *mlpipeline.Runner

	sseMu             sync.Mutex
	sseClients        map[string]map[string]chan mlpipeline.ProgressUpdate
	sseClientCount    int
	sseClientSequence atomic.Uint64
	lastProgress      map[string]mlpipeline.ProgressUpdate
	terminalProgress  map[string]struct{}
	terminalOrder     []string
}

type trainHandlerInput struct {
	benchmarkDataPath string
	config            mlpipeline.TrainRequest
	uploadDir         string
}

// NewMLPipelineHandler creates a new ML pipeline handler.
func NewMLPipelineHandler(runner *mlpipeline.Runner) *MLPipelineHandler {
	h := &MLPipelineHandler{
		runner:           runner,
		sseClients:       make(map[string]map[string]chan mlpipeline.ProgressUpdate),
		lastProgress:     make(map[string]mlpipeline.ProgressUpdate),
		terminalProgress: make(map[string]struct{}),
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
	h.sseMu.Lock()
	defer h.sseMu.Unlock()

	if update.Percent > 0 {
		h.lastProgress[update.JobID] = update
		if update.Percent >= 100 {
			if _, retained := h.terminalProgress[update.JobID]; !retained {
				h.terminalProgress[update.JobID] = struct{}{}
				h.terminalOrder = append(h.terminalOrder, update.JobID)
			}
			h.pruneTerminalProgressLocked()
		}
	}

	if clients := h.sseClients[update.JobID]; clients != nil {
		for _, ch := range clients {
			offerProgressUpdate(ch, update)
		}
	}
}

func offerProgressUpdate(ch chan mlpipeline.ProgressUpdate, update mlpipeline.ProgressUpdate) {
	select {
	case ch <- update:
		return
	default:
	}
	if update.Percent < 100 {
		return
	}
	select {
	case <-ch:
	default:
	}
	select {
	case ch <- update:
	default:
	}
}

func (h *MLPipelineHandler) pruneTerminalProgressLocked() {
	attempts := len(h.terminalOrder)
	for len(h.lastProgress) > mlTerminalProgressRetention && len(h.terminalOrder) > 0 && attempts > 0 {
		jobID := h.terminalOrder[0]
		h.terminalOrder = h.terminalOrder[1:]
		attempts--
		if len(h.sseClients[jobID]) > 0 {
			h.terminalOrder = append(h.terminalOrder, jobID)
			continue
		}
		delete(h.terminalProgress, jobID)
		delete(h.lastProgress, jobID)
	}
}

func (h *MLPipelineHandler) registerSSEClient(jobID string) (string, chan mlpipeline.ProgressUpdate, mlpipeline.ProgressUpdate, bool, error) {
	h.sseMu.Lock()
	defer h.sseMu.Unlock()
	clients := h.sseClients[jobID]
	if h.sseClientCount >= mlSSEClientsGlobalLimit || len(clients) >= mlSSEClientsPerJobLimit {
		return "", nil, mlpipeline.ProgressUpdate{}, false, errors.New("ML pipeline SSE capacity exceeded")
	}
	if clients == nil {
		clients = make(map[string]chan mlpipeline.ProgressUpdate)
		h.sseClients[jobID] = clients
	}
	clientID := strconv.FormatUint(h.sseClientSequence.Add(1), 10)
	clientChan := make(chan mlpipeline.ProgressUpdate, 10)
	clients[clientID] = clientChan
	h.sseClientCount++
	last, hasLast := h.lastProgress[jobID]
	return clientID, clientChan, last, hasLast, nil
}

func (h *MLPipelineHandler) unregisterSSEClient(jobID, clientID string) {
	h.sseMu.Lock()
	defer h.sseMu.Unlock()
	clients := h.sseClients[jobID]
	if clients == nil {
		return
	}
	if _, exists := clients[clientID]; exists {
		delete(clients, clientID)
		h.sseClientCount--
	}
	if len(clients) == 0 {
		delete(h.sseClients, jobID)
		h.pruneTerminalProgressLocked()
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

// GetJobHandler returns a specific job by ID, or GET .../jobs/{id}/events for typed progress history.
func (h *MLPipelineHandler) GetJobHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		pathParts := strings.Split(strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/ml-pipeline/jobs/"), "/"), "/")
		if len(pathParts) == 0 || pathParts[0] == "" {
			http.Error(w, "Job ID required", http.StatusBadRequest)
			return
		}
		jobID := pathParts[0]
		if !validMLJobID(jobID) || len(pathParts) > 2 {
			http.Error(w, "Invalid job path", http.StatusBadRequest)
			return
		}

		if len(pathParts) >= 2 && pathParts[1] == "events" {
			if h.runner.GetJob(jobID) == nil {
				http.Error(w, "Job not found", http.StatusNotFound)
				return
			}
			events, err := h.runner.ListProgressEvents(jobID, 500)
			if err != nil {
				http.Error(w, "Failed to load progress events", http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(struct {
				JobID  string                          `json:"job_id"`
				Events []workflowstore.MLProgressEvent `json:"events"`
			}{JobID: jobID, Events: events})
			return
		}
		if len(pathParts) == 2 {
			http.Error(w, "Invalid job path", http.StatusBadRequest)
			return
		}

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

		defer cleanupMultipartForm(r)
		if status, err := parseBoundedMultipart(w, r, mlBenchmarkBodyLimit, mlBenchmarkMultipartMemory); err != nil {
			http.Error(w, "Invalid multipart form", status)
			return
		}

		uploadDir, err := h.runner.CreateUploadDir("benchmark-")
		if err != nil {
			http.Error(w, "Failed to prepare benchmark upload", http.StatusInternalServerError)
			return
		}
		runnerOwnsUpload := false
		defer func() {
			if !runnerOwnsUpload {
				_ = h.runner.RemoveUploadDir(uploadDir)
			}
		}()

		modelsPath, err := saveUploadedFile(r, "models_yaml", uploadDir, "models.yaml", mlModelsFileLimit)
		if err != nil {
			http.Error(w, "Invalid models YAML upload", http.StatusBadRequest)
			return
		}

		queriesPath, err := saveUploadedFile(r, "queries_jsonl", uploadDir, "queries.jsonl", mlQueriesFileLimit)
		if err != nil {
			http.Error(w, "Invalid queries JSONL upload", http.StatusBadRequest)
			return
		}

		var req mlpipeline.BenchmarkRequest
		if configJSON := r.FormValue("config"); configJSON != "" {
			if len(configJSON) > mlFormConfigLimit {
				http.Error(w, "Benchmark config is too large", http.StatusRequestEntityTooLarge)
				return
			}
			if _, decodeErr := decodeStrictJSONBytes([]byte(configJSON), &req); decodeErr != nil {
				http.Error(w, "Invalid benchmark config", http.StatusBadRequest)
				return
			}
		}
		if validationErr := mlpipeline.ValidateBenchmarkRequest(req); validationErr != nil {
			http.Error(w, "Invalid benchmark config", http.StatusBadRequest)
			return
		}

		ctx := context.Background()
		jobID, err := h.runner.RunBenchmark(ctx, modelsPath, queriesPath, req)
		if err != nil {
			if errors.Is(err, mlpipeline.ErrJobCapacityExceeded) {
				w.Header().Set("Retry-After", "5")
				http.Error(w, "ML pipeline is busy", http.StatusTooManyRequests)
				return
			}
			if errors.Is(err, mlpipeline.ErrPipelineInputRejected) {
				http.Error(w, "Invalid benchmark input", http.StatusBadRequest)
				return
			}
			http.Error(w, "Failed to start benchmark", http.StatusInternalServerError)
			return
		}
		runnerOwnsUpload = true

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

		input, ok := h.parseTrainingInput(w, r)
		if !ok {
			return
		}
		runnerOwnsUpload := false
		defer func() {
			if input.uploadDir != "" && !runnerOwnsUpload {
				_ = h.runner.RemoveUploadDir(input.uploadDir)
			}
		}()

		if input.benchmarkDataPath == "" {
			http.Error(w, "Training data is required: upload training_data or provide benchmark_job_id", http.StatusBadRequest)
			return
		}

		if len(input.config.Algorithms) == 0 {
			input.config.Algorithms = []string{"knn", "kmeans", "svm", "mlp"}
		}
		if err := mlpipeline.ValidateTrainRequest(input.config); err != nil {
			http.Error(w, "Invalid training config", http.StatusBadRequest)
			return
		}

		ctx := context.Background()
		jobID, err := h.runner.RunTrain(ctx, input.benchmarkDataPath, input.config)
		if err != nil {
			if errors.Is(err, mlpipeline.ErrJobCapacityExceeded) {
				w.Header().Set("Retry-After", "5")
				http.Error(w, "ML pipeline is busy", http.StatusTooManyRequests)
				return
			}
			if errors.Is(err, mlpipeline.ErrPipelineInputRejected) {
				http.Error(w, "Invalid training input", http.StatusBadRequest)
				return
			}
			http.Error(w, "Failed to start training", http.StatusInternalServerError)
			return
		}
		runnerOwnsUpload = input.uploadDir != ""

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"job_id": jobID,
			"status": "started",
		})
	}
}

func (h *MLPipelineHandler) parseTrainingInput(w http.ResponseWriter, r *http.Request) (trainHandlerInput, bool) {
	mediaType, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil {
		http.Error(w, "Unsupported request content type", http.StatusUnsupportedMediaType)
		return trainHandlerInput{}, false
	}

	switch mediaType {
	case "multipart/form-data":
		return h.parseMultipartTrainingInput(w, r)
	case "application/json":
		return h.parseJSONTrainingInput(w, r)
	default:
		http.Error(w, "Unsupported request content type", http.StatusUnsupportedMediaType)
		return trainHandlerInput{}, false
	}
}

func (h *MLPipelineHandler) parseMultipartTrainingInput(w http.ResponseWriter, r *http.Request) (trainHandlerInput, bool) {
	defer cleanupMultipartForm(r)
	if status, err := parseBoundedMultipart(w, r, mlTrainingBodyLimit, mlTrainingMultipartMemory); err != nil {
		http.Error(w, "Invalid multipart form", status)
		return trainHandlerInput{}, false
	}

	uploadDir, err := h.runner.CreateUploadDir("training-")
	if err != nil {
		http.Error(w, "Failed to prepare training upload", http.StatusInternalServerError)
		return trainHandlerInput{}, false
	}
	accepted := false
	defer func() {
		if !accepted {
			_ = h.runner.RemoveUploadDir(uploadDir)
		}
	}()

	uploadedPath, err := saveUploadedFile(r, "training_data", uploadDir, "training.jsonl", mlTrainingFileLimit)
	if err != nil {
		http.Error(w, "Invalid training data upload", http.StatusBadRequest)
		return trainHandlerInput{}, false
	}
	input := trainHandlerInput{benchmarkDataPath: uploadedPath, uploadDir: uploadDir}
	if configJSON := r.FormValue("config"); configJSON != "" {
		if len(configJSON) > mlFormConfigLimit {
			http.Error(w, "Training config is too large", http.StatusRequestEntityTooLarge)
			return trainHandlerInput{}, false
		}
		if _, decodeErr := decodeStrictJSONBytes([]byte(configJSON), &input.config); decodeErr != nil {
			http.Error(w, "Invalid training config", http.StatusBadRequest)
			return trainHandlerInput{}, false
		}
	}
	accepted = true
	return input, true
}

func (h *MLPipelineHandler) parseJSONTrainingInput(w http.ResponseWriter, r *http.Request) (trainHandlerInput, bool) {
	var body struct {
		BenchmarkJobID string                  `json:"benchmark_job_id"`
		Config         mlpipeline.TrainRequest `json:"config"`
	}
	if status, err := decodeBoundedJSON(w, r, mlJSONBodyLimit, &body); err != nil {
		http.Error(w, "Invalid request body", status)
		return trainHandlerInput{}, false
	}
	if !validMLJobID(body.BenchmarkJobID) {
		http.Error(w, "A valid benchmark_job_id is required", http.StatusBadRequest)
		return trainHandlerInput{}, false
	}
	benchJob := h.runner.GetJob(body.BenchmarkJobID)
	if benchJob == nil || benchJob.Type != "benchmark" || benchJob.Status != mlpipeline.StatusCompleted || len(benchJob.OutputFiles) == 0 {
		http.Error(w, "Completed benchmark job not found", http.StatusBadRequest)
		return trainHandlerInput{}, false
	}
	validatedPath, err := h.runner.ValidateJobOutputFile(benchJob.ID, benchJob.Type, benchJob.OutputFiles[0])
	if err != nil {
		http.Error(w, "Benchmark output is unavailable", http.StatusBadRequest)
		return trainHandlerInput{}, false
	}
	return trainHandlerInput{benchmarkDataPath: validatedPath, config: body.Config}, true
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
		if status, err := decodeBoundedJSON(w, r, mlJSONBodyLimit, &req); err != nil {
			http.Error(w, "Invalid request body", status)
			return
		}
		if err := mlpipeline.ValidateConfigRequest(req); err != nil {
			http.Error(w, "Invalid config request", http.StatusBadRequest)
			return
		}

		jobID, err := h.runner.GenerateConfig(req)
		if err != nil {
			http.Error(w, "Failed to generate config", http.StatusInternalServerError)
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

		pathParts := strings.Split(strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/ml-pipeline/download/"), "/"), "/")
		if len(pathParts) == 0 || len(pathParts) > 2 || !validMLJobID(pathParts[0]) {
			http.Error(w, "Job ID required", http.StatusBadRequest)
			return
		}
		jobID := pathParts[0]

		job := h.runner.GetJob(jobID)
		if job == nil {
			http.Error(w, "Job not found", http.StatusNotFound)
			return
		}

		if job.Status != mlpipeline.StatusCompleted || len(job.OutputFiles) == 0 {
			http.Error(w, "No output files available", http.StatusNotFound)
			return
		}

		// Serve the first output file (or specific one if index is provided)
		fileIdx := 0
		if len(pathParts) > 1 && pathParts[1] != "" {
			parsedIndex, err := strconv.Atoi(pathParts[1])
			if err != nil || parsedIndex < 0 {
				http.Error(w, "Invalid file index", http.StatusBadRequest)
				return
			}
			fileIdx = parsedIndex
		}
		if fileIdx >= len(job.OutputFiles) {
			http.Error(w, "File index out of range", http.StatusBadRequest)
			return
		}

		filePath := job.OutputFiles[fileIdx]
		filename := filepath.Base(filePath)
		file, info, err := h.runner.OpenJobOutputFile(job.ID, job.Type, filePath)
		if err != nil {
			http.Error(w, "Output file unavailable", http.StatusNotFound)
			return
		}
		defer file.Close()

		w.Header().Set("Content-Disposition", mime.FormatMediaType("attachment", map[string]string{"filename": filename}))
		if strings.HasSuffix(filename, ".yaml") || strings.HasSuffix(filename, ".yml") {
			w.Header().Set("Content-Type", "text/yaml")
		} else if strings.HasSuffix(filename, ".json") {
			w.Header().Set("Content-Type", "application/json")
		} else if strings.HasSuffix(filename, ".jsonl") {
			w.Header().Set("Content-Type", "application/x-jsonlines")
		} else {
			w.Header().Set("Content-Type", "application/octet-stream")
		}

		http.ServeContent(w, r, filename, info.ModTime(), file)
	}
}

// StreamProgressHandler provides SSE for job progress updates.
func (h *MLPipelineHandler) StreamProgressHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		pathParts := strings.Split(strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/ml-pipeline/stream/"), "/"), "/")
		if len(pathParts) != 1 || !validMLJobID(pathParts[0]) {
			http.Error(w, "Job ID required", http.StatusBadRequest)
			return
		}
		job := h.runner.GetJob(pathParts[0])
		if job == nil {
			http.Error(w, "Job not found", http.StatusNotFound)
			return
		}
		jobID := job.ID

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		clientID, clientChan, lastProgress, hasLastProgress, err := h.registerSSEClient(jobID)
		if err != nil {
			w.Header().Set("Retry-After", "5")
			http.Error(w, "Too many progress stream clients", http.StatusTooManyRequests)
			return
		}
		defer h.unregisterSSEClient(jobID, clientID)

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		// Send initial connection message
		connectedData, _ := json.Marshal(map[string]string{"job_id": jobID})
		_, _ = fmt.Fprintf(w, "event: connected\ndata: %s\n\n", connectedData)
		flusher.Flush()

		// Send last-known progress so client catches up on missed events
		if hasLastProgress {
			data, err := json.Marshal(lastProgress)
			if err == nil {
				_, _ = fmt.Fprintf(w, "event: progress\ndata: %s\n\n", data)
				flusher.Flush()
				// If the job already finished, send completed immediately
				if lastProgress.Percent >= 100 {
					_, _ = fmt.Fprintf(w, "event: completed\ndata: %s\n\n", connectedData)
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
					_, _ = fmt.Fprintf(w, "event: completed\ndata: %s\n\n", connectedData)
					flusher.Flush()
					return
				}
			}
		}
	}
}

func parseBoundedMultipart(w http.ResponseWriter, r *http.Request, maxBodyBytes, maxMemoryBytes int64) (int, error) {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
	if err := r.ParseMultipartForm(maxMemoryBytes); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			return http.StatusRequestEntityTooLarge, errors.New("multipart request is too large")
		}
		return http.StatusBadRequest, errors.New("multipart request is invalid")
	}
	return 0, nil
}

func cleanupMultipartForm(r *http.Request) {
	if r != nil && r.MultipartForm != nil {
		_ = r.MultipartForm.RemoveAll()
	}
}

// saveUploadedFile ignores the caller-supplied filename and writes one bounded
// upload to a server-selected name in a private directory.
func saveUploadedFile(r *http.Request, fieldName, targetDir, targetName string, maxBytes int64) (string, error) {
	file, header, err := r.FormFile(fieldName)
	if err != nil {
		return "", fmt.Errorf("missing file field '%s': %w", fieldName, err)
	}
	defer file.Close()
	if header.Size > maxBytes {
		return "", errors.New("uploaded file exceeds its limit")
	}

	destPath := filepath.Join(targetDir, targetName)
	out, err := os.OpenFile(destPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return "", fmt.Errorf("failed to create file: %w", err)
	}
	keepFile := false
	defer func() {
		_ = out.Close()
		if !keepFile {
			_ = os.Remove(destPath)
		}
	}()

	written, err := io.Copy(out, io.LimitReader(file, maxBytes+1))
	if err != nil {
		return "", fmt.Errorf("failed to write file: %w", err)
	}
	if written > maxBytes {
		return "", errors.New("uploaded file exceeds its limit")
	}
	if err := out.Close(); err != nil {
		return "", fmt.Errorf("failed to close file: %w", err)
	}
	keepFile = true

	return destPath, nil
}

func validMLJobID(jobID string) bool {
	if len(jobID) == 0 || len(jobID) > 128 {
		return false
	}
	for _, r := range jobID {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			continue
		}
		return false
	}
	return true
}
