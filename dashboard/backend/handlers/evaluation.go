package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/evaluation"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

// EvaluationHandler holds dependencies for evaluation endpoints.
type EvaluationHandler struct {
	db           *evaluation.DB
	runner       *evaluation.Runner
	readonlyMode bool
	routerAPIURL string   // Router API URL for signal evaluation
	envoyURL     string   // Envoy URL for model evaluation
	cancelFuncs  sync.Map // map[taskID]context.CancelFunc
	runMu        sync.Mutex
	activeRuns   map[string]struct{}
	sseMu        sync.Mutex
	sseClients   map[string]*evaluationSSETaskRegistry
	sseTotal     int
}

// NewEvaluationHandler creates a new evaluation handler.
func NewEvaluationHandler(db *evaluation.DB, runner *evaluation.Runner, readonlyMode bool, routerAPIURL, envoyURL string) *EvaluationHandler {
	h := &EvaluationHandler{
		db:           db,
		runner:       runner,
		readonlyMode: readonlyMode,
		routerAPIURL: routerAPIURL,
		envoyURL:     envoyURL,
		activeRuns:   make(map[string]struct{}),
		sseClients:   make(map[string]*evaluationSSETaskRegistry),
	}

	// Start background goroutine to forward progress updates to SSE clients
	go h.forwardProgressUpdates()

	return h
}

// forwardProgressUpdates forwards progress updates from the runner to SSE clients.
func (h *EvaluationHandler) forwardProgressUpdates() {
	for update := range h.runner.ProgressUpdates() {
		h.broadcastProgress(update)
	}
}

// broadcastProgress sends a progress update to all subscribed clients for a task.
func (h *EvaluationHandler) broadcastProgress(update models.ProgressUpdate) {
	h.broadcastEvaluationProgress(update)
}

// ListTasksHandler returns all evaluation tasks.
func (h *EvaluationHandler) ListTasksHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		status := strings.TrimSpace(r.URL.Query().Get("status"))
		if !validEvaluationStatusFilter(status) {
			http.Error(w, "Invalid task status", http.StatusBadRequest)
			return
		}
		tasks, err := h.db.ListTasks(status)
		if err != nil {
			log.Printf("Evaluation task list failed (%T)", err)
			http.Error(w, "Failed to list tasks", http.StatusInternalServerError)
			return
		}

		if tasks == nil {
			tasks = []*models.EvaluationTask{}
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(tasks); err != nil {
			log.Printf("Evaluation response encoding failed (%T)", err)
		}
	}
}

// GetTaskHandler returns a specific task by ID.
func (h *EvaluationHandler) GetTaskHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		taskID, ok := evaluationTaskIDFromPath(r.URL.Path, "/api/evaluation/tasks/")
		if !ok {
			http.Error(w, "Invalid task ID", http.StatusBadRequest)
			return
		}

		task, err := h.db.GetTask(taskID)
		if err != nil {
			log.Printf("Evaluation task %s lookup failed (%T)", taskID, err)
			http.Error(w, "Failed to get task", http.StatusInternalServerError)
			return
		}

		if task == nil {
			http.Error(w, "Task not found", http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(task); err != nil {
			log.Printf("Evaluation response encoding failed (%T)", err)
		}
	}
}

// CreateTaskHandler creates a new evaluation task.
func (h *EvaluationHandler) CreateTaskHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if h.readonlyMode {
			http.Error(w, "Operation not allowed in readonly mode", http.StatusForbidden)
			return
		}
		var req models.CreateTaskRequest
		if status, err := decodeBoundedJSON(w, r, smallJSONRequestBodyLimit, &req); err != nil {
			http.Error(w, "Invalid request body", status)
			return
		}
		normalizeEvaluationCreateConfig(&req.Config)
		if msg, code := validateEvaluationCreateRequest(&req); msg != "" {
			http.Error(w, msg, code)
			return
		}
		h.applyEvaluationCreateDefaults(&req.Config)
		task := &models.EvaluationTask{
			Name:        req.Name,
			Description: req.Description,
			Config:      req.Config,
		}
		if err := h.db.CreateTask(task); err != nil {
			log.Printf("Evaluation task creation failed (%T)", err)
			http.Error(w, "Failed to create task", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		if err := json.NewEncoder(w).Encode(task); err != nil {
			log.Printf("Evaluation response encoding failed (%T)", err)
		}
	}
}

// DeleteTaskHandler deletes an evaluation task.
func (h *EvaluationHandler) DeleteTaskHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodDelete {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if h.readonlyMode {
			http.Error(w, "Operation not allowed in readonly mode", http.StatusForbidden)
			return
		}

		taskID, ok := evaluationTaskIDFromPath(r.URL.Path, "/api/evaluation/tasks/")
		if !ok {
			http.Error(w, "Invalid task ID", http.StatusBadRequest)
			return
		}

		h.runMu.Lock()
		defer h.runMu.Unlock()
		if _, running := h.activeRuns[taskID]; running {
			http.Error(w, "Cannot delete a running task", http.StatusConflict)
			return
		}
		task, err := h.db.GetTask(taskID)
		if err != nil {
			log.Printf("Evaluation task %s lookup before delete failed (%T)", taskID, err)
			http.Error(w, "Failed to delete task", http.StatusInternalServerError)
			return
		}
		if task == nil {
			http.Error(w, "Task not found", http.StatusNotFound)
			return
		}
		if task.Status == models.StatusRunning {
			http.Error(w, "Cannot delete a running task", http.StatusConflict)
			return
		}
		if err := h.db.DeleteTask(taskID); err != nil {
			if strings.Contains(err.Error(), "not found") {
				http.Error(w, "Task not found", http.StatusNotFound)
				return
			}
			log.Printf("Evaluation task %s deletion failed (%T)", taskID, err)
			http.Error(w, "Failed to delete task", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
	}
}

// RunTaskHandler starts running an evaluation task.
func (h *EvaluationHandler) RunTaskHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if h.readonlyMode {
			http.Error(w, "Operation not allowed in readonly mode", http.StatusForbidden)
			return
		}

		var req models.RunTaskRequest
		if status, err := decodeBoundedJSON(w, r, smallJSONRequestBodyLimit, &req); err != nil {
			http.Error(w, "Invalid request body", status)
			return
		}

		req.TaskID = strings.TrimSpace(req.TaskID)
		if !validEvaluationTaskID(req.TaskID) {
			http.Error(w, "Invalid task ID", http.StatusBadRequest)
			return
		}

		h.runMu.Lock()
		defer h.runMu.Unlock()
		if _, running := h.activeRuns[req.TaskID]; running {
			http.Error(w, "Task is already running", http.StatusConflict)
			return
		}
		if len(h.activeRuns) >= h.runner.MaxConcurrent() {
			http.Error(w, "Too many evaluation tasks are running", http.StatusTooManyRequests)
			return
		}

		task, err := h.db.GetTask(req.TaskID)
		if err != nil {
			log.Printf("Evaluation task %s lookup before run failed (%T)", req.TaskID, err)
			http.Error(w, "Failed to get task", http.StatusInternalServerError)
			return
		}
		if task == nil {
			http.Error(w, "Task not found", http.StatusNotFound)
			return
		}
		if task.Status != models.StatusPending && task.Status != models.StatusFailed {
			http.Error(w, "Task cannot be started in its current state", http.StatusConflict)
			return
		}

		// Transition the task before returning so reruns do not briefly show the previous failed state.
		if err := h.db.UpdateTaskStatus(req.TaskID, models.StatusRunning, ""); err != nil {
			log.Printf("Evaluation task %s running transition failed (%T)", req.TaskID, err)
			http.Error(w, "Failed to start task", http.StatusInternalServerError)
			return
		}
		if err := h.db.UpdateTaskProgress(req.TaskID, 0, "Starting evaluation"); err != nil {
			log.Printf("Evaluation task %s progress reset failed (%T)", req.TaskID, err)
			_ = h.db.UpdateTaskStatus(req.TaskID, models.StatusFailed, "Failed to initialize evaluation")
			http.Error(w, "Failed to initialize task progress", http.StatusInternalServerError)
			return
		}

		ctx, cancel := context.WithCancel(context.Background())
		h.cancelFuncs.Store(req.TaskID, cancel)
		h.activeRuns[req.TaskID] = struct{}{}

		go func(taskID string) {
			defer h.finishEvaluationRun(taskID)
			defer func() {
				if recovered := recover(); recovered != nil {
					log.Printf("Evaluation task %s panicked (panic_type=%T)", taskID, recovered)
					_ = h.db.UpdateTaskStatus(taskID, models.StatusFailed, "Evaluation failed")
				}
			}()
			if err := h.runner.RunTask(ctx, taskID); err != nil {
				log.Printf("Evaluation task %s failed (error_type=%T)", taskID, err)
			}
		}(req.TaskID)

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{
			"status":  "started",
			"task_id": req.TaskID,
		})
	}
}

// CancelTaskHandler cancels a running evaluation task.
func (h *EvaluationHandler) CancelTaskHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if h.readonlyMode {
			http.Error(w, "Operation not allowed in readonly mode", http.StatusForbidden)
			return
		}

		taskID, ok := evaluationTaskIDFromPath(r.URL.Path, "/api/evaluation/cancel/")
		if !ok {
			http.Error(w, "Invalid task ID", http.StatusBadRequest)
			return
		}

		h.runMu.Lock()
		defer h.runMu.Unlock()
		task, err := h.db.GetTask(taskID)
		if err != nil {
			log.Printf("Evaluation task %s lookup before cancellation failed (%T)", taskID, err)
			http.Error(w, "Failed to cancel task", http.StatusInternalServerError)
			return
		}
		if task == nil {
			http.Error(w, "Task not found", http.StatusNotFound)
			return
		}
		if task.Status != models.StatusRunning {
			http.Error(w, "Task is not running", http.StatusConflict)
			return
		}
		if cancelFunc, ok := h.cancelFuncs.Load(taskID); ok {
			cancelFunc.(context.CancelFunc)()
		}

		if err := h.runner.CancelTask(taskID); err != nil {
			log.Printf("Evaluation task %s cancellation failed (%T)", taskID, err)
			http.Error(w, "Failed to cancel task", http.StatusInternalServerError)
			return
		}
		h.finishEvaluationSSETask(taskID, models.StatusCancelled)

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "cancelled"})
	}
}

// StreamProgressHandler provides SSE for task progress updates.
func (h *EvaluationHandler) StreamProgressHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		taskID, ok := evaluationTaskIDFromPath(r.URL.Path, "/api/evaluation/stream/")
		if !ok {
			http.Error(w, "Invalid task ID", http.StatusBadRequest)
			return
		}

		// Serialize terminal-state observation with task completion. If a stream
		// registers first, completion will detach and notify it; if completion
		// wins, the terminal state is observed here and no registry is created.
		h.runMu.Lock()
		task, err := h.db.GetTask(taskID)
		if err != nil {
			h.runMu.Unlock()
			log.Printf("Evaluation task %s lookup before stream failed (%T)", taskID, err)
			http.Error(w, "Failed to open progress stream", http.StatusInternalServerError)
			return
		}
		if task == nil {
			h.runMu.Unlock()
			http.Error(w, "Task not found", http.StatusNotFound)
			return
		}
		if task.Status == models.StatusCompleted || task.Status == models.StatusFailed || task.Status == models.StatusCancelled {
			h.runMu.Unlock()
			http.Error(w, "Task is already finished", http.StatusConflict)
			return
		}
		clientID, client, err := h.registerEvaluationSSEClient(taskID)
		h.runMu.Unlock()
		if err != nil {
			http.Error(w, err.Error(), http.StatusTooManyRequests)
			return
		}
		defer h.unregisterEvaluationSSEClient(taskID, clientID, client)

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		_, _ = fmt.Fprintf(w, "event: connected\ndata: {\"task_id\":\"%s\"}\n\n", taskID)
		flusher.Flush()

		ctx := r.Context()
		heartbeat := time.NewTicker(15 * time.Second)
		defer heartbeat.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-heartbeat.C:
				_, _ = fmt.Fprint(w, ": heartbeat\n\n")
				flusher.Flush()
			case status := <-client.terminal:
				data, _ := json.Marshal(map[string]string{"task_id": taskID, "status": string(status)})
				_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", status, data)
				flusher.Flush()
				return
			case update := <-client.updates:
				data, err := json.Marshal(update)
				if err != nil {
					log.Printf("Evaluation progress encoding failed (%T)", err)
					continue
				}
				_, _ = fmt.Fprintf(w, "event: progress\ndata: %s\n\n", data)
				flusher.Flush()
			}
		}
	}
}

// GetResultsHandler returns results for a completed task.
func (h *EvaluationHandler) GetResultsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		taskID, ok := evaluationTaskIDFromPath(r.URL.Path, "/api/evaluation/results/")
		if !ok {
			http.Error(w, "Invalid task ID", http.StatusBadRequest)
			return
		}

		// Get task to check status
		task, err := h.db.GetTask(taskID)
		if err != nil {
			log.Printf("Evaluation task %s result lookup failed (%T)", taskID, err)
			http.Error(w, "Failed to get task", http.StatusInternalServerError)
			return
		}
		if task == nil {
			http.Error(w, "Task not found", http.StatusNotFound)
			return
		}

		// Get results
		results, err := h.db.GetResults(taskID)
		if err != nil {
			log.Printf("Evaluation task %s result query failed (%T)", taskID, err)
			http.Error(w, "Failed to get results", http.StatusInternalServerError)
			return
		}

		if results == nil {
			results = []*models.EvaluationResult{}
		}

		response := map[string]interface{}{
			"task":    task,
			"results": results,
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			log.Printf("Evaluation response encoding failed (%T)", err)
		}
	}
}

// ExportResultsHandler exports results in the specified format.
func (h *EvaluationHandler) ExportResultsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		taskID, ok := evaluationTaskIDFromPath(r.URL.Path, "/api/evaluation/export/")
		if !ok {
			http.Error(w, "Invalid task ID", http.StatusBadRequest)
			return
		}

		format := models.ExportFormat(strings.TrimSpace(r.URL.Query().Get("format")))
		if format == "" {
			format = models.ExportJSON
		}
		if format != models.ExportJSON && format != models.ExportCSV {
			http.Error(w, "Unsupported export format", http.StatusBadRequest)
			return
		}

		data, contentType, err := h.runner.ExportResults(taskID, format)
		if err != nil {
			log.Printf("Evaluation task %s export failed (%T)", taskID, err)
			http.Error(w, "Failed to export results", http.StatusInternalServerError)
			return
		}

		// Set filename for download
		filename := fmt.Sprintf("evaluation_%s.%s", taskID[:8], format)
		w.Header().Set("Content-Type", contentType)
		w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%s", filename))
		_, _ = w.Write(data)
	}
}

// GetDatasetsHandler returns available datasets grouped by dimension.
// This is a standalone function that doesn't require database initialization,
// allowing datasets to be served even when the evaluation DB fails to initialize.
func GetDatasetsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		datasets := evaluation.GetAvailableDatasets()

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(datasets); err != nil {
			log.Printf("Evaluation response encoding failed (%T)", err)
		}
	}
}

// GetHistoryHandler returns historical metrics for trend analysis.
func (h *EvaluationHandler) GetHistoryHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		metricName := strings.TrimSpace(r.URL.Query().Get("metric"))
		if metricName == "" {
			http.Error(w, "Metric name is required", http.StatusBadRequest)
			return
		}
		if len([]byte(metricName)) > 128 || containsUnicodeControl(metricName) {
			http.Error(w, "Invalid metric name", http.StatusBadRequest)
			return
		}

		limit := 100 // Default limit
		if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
			parsed, err := strconv.Atoi(limitStr)
			if err != nil || parsed < 1 || parsed > 1000 {
				http.Error(w, "limit must be between 1 and 1000", http.StatusBadRequest)
				return
			}
			limit = parsed
		}

		entries, err := h.db.GetHistoryForMetric(metricName, limit)
		if err != nil {
			log.Printf("Evaluation history query failed (%T)", err)
			http.Error(w, "Failed to get history", http.StatusInternalServerError)
			return
		}

		if entries == nil {
			entries = []*models.EvaluationHistoryEntry{}
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(entries); err != nil {
			log.Printf("Evaluation response encoding failed (%T)", err)
		}
	}
}
