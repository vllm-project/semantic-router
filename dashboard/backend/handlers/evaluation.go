package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
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
	sseClients   sync.Map // map[taskID]map[clientID]chan models.ProgressUpdate
	cancelFuncs  sync.Map // map[taskID]context.CancelFunc
}

// NewEvaluationHandler creates a new evaluation handler.
func NewEvaluationHandler(db *evaluation.DB, runner *evaluation.Runner, readonlyMode bool) *EvaluationHandler {
	h := &EvaluationHandler{
		db:           db,
		runner:       runner,
		readonlyMode: readonlyMode,
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
	if clientsMap, ok := h.sseClients.Load(update.TaskID); ok {
		clients := clientsMap.(*sync.Map)
		clients.Range(func(key, value interface{}) bool {
			ch := value.(chan models.ProgressUpdate)
			select {
			case ch <- update:
			default:
				// Client channel full, skip
			}
			return true
		})
	}
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

		status := r.URL.Query().Get("status")
		tasks, err := h.db.ListTasks(status)
		if err != nil {
			log.Printf("Failed to list tasks: %v", err)
			http.Error(w, fmt.Sprintf("Failed to list tasks: %v", err), http.StatusInternalServerError)
			return
		}

		if tasks == nil {
			tasks = []*models.EvaluationTask{}
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(tasks); err != nil {
			log.Printf("Error encoding response: %v", err)
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

		// Extract task ID from URL path: /api/evaluation/tasks/{id}
		pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/evaluation/tasks/"), "/")
		if len(pathParts) == 0 || pathParts[0] == "" {
			http.Error(w, "Task ID required", http.StatusBadRequest)
			return
		}
		taskID := pathParts[0]

		task, err := h.db.GetTask(taskID)
		if err != nil {
			log.Printf("Failed to get task: %v", err)
			http.Error(w, fmt.Sprintf("Failed to get task: %v", err), http.StatusInternalServerError)
			return
		}

		if task == nil {
			http.Error(w, "Task not found", http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(task); err != nil {
			log.Printf("Error encoding response: %v", err)
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

		var req models.CreateTaskRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
			return
		}

		// Validate request
		if req.Name == "" {
			http.Error(w, "Task name is required", http.StatusBadRequest)
			return
		}
		if len(req.Config.Dimensions) == 0 {
			http.Error(w, "At least one evaluation dimension is required", http.StatusBadRequest)
			return
		}

		// Set defaults
		if req.Config.MaxSamples <= 0 {
			req.Config.MaxSamples = 50
		}
		if req.Config.Endpoint == "" {
			req.Config.Endpoint = "http://localhost:8888"
		}
		if req.Config.SamplesPerCat <= 0 {
			req.Config.SamplesPerCat = 10
		}

		task := &models.EvaluationTask{
			Name:        req.Name,
			Description: req.Description,
			Config:      req.Config,
		}

		if err := h.db.CreateTask(task); err != nil {
			log.Printf("Failed to create task: %v", err)
			http.Error(w, fmt.Sprintf("Failed to create task: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		if err := json.NewEncoder(w).Encode(task); err != nil {
			log.Printf("Error encoding response: %v", err)
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

		// Extract task ID from URL path
		pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/evaluation/tasks/"), "/")
		if len(pathParts) == 0 || pathParts[0] == "" {
			http.Error(w, "Task ID required", http.StatusBadRequest)
			return
		}
		taskID := pathParts[0]

		if err := h.db.DeleteTask(taskID); err != nil {
			if strings.Contains(err.Error(), "not found") {
				http.Error(w, "Task not found", http.StatusNotFound)
				return
			}
			log.Printf("Failed to delete task: %v", err)
			http.Error(w, fmt.Sprintf("Failed to delete task: %v", err), http.StatusInternalServerError)
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

		var req models.RunTaskRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
			return
		}

		if req.TaskID == "" {
			http.Error(w, "Task ID is required", http.StatusBadRequest)
			return
		}

		// Check if task exists and is in pending state
		task, err := h.db.GetTask(req.TaskID)
		if err != nil {
			log.Printf("Failed to get task: %v", err)
			http.Error(w, fmt.Sprintf("Failed to get task: %v", err), http.StatusInternalServerError)
			return
		}
		if task == nil {
			http.Error(w, "Task not found", http.StatusNotFound)
			return
		}
		if task.Status != models.StatusPending && task.Status != models.StatusFailed {
			http.Error(w, fmt.Sprintf("Task is already %s", task.Status), http.StatusConflict)
			return
		}

		// Create cancellation context
		ctx, cancel := context.WithCancel(context.Background())
		h.cancelFuncs.Store(req.TaskID, cancel)

		// Run task in background
		go func() {
			defer h.cancelFuncs.Delete(req.TaskID)
			if err := h.runner.RunTask(ctx, req.TaskID); err != nil {
				log.Printf("Task %s failed: %v", req.TaskID, err)
			}
		}()

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

		// Extract task ID from URL path
		pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/evaluation/cancel/"), "/")
		if len(pathParts) == 0 || pathParts[0] == "" {
			http.Error(w, "Task ID required", http.StatusBadRequest)
			return
		}
		taskID := pathParts[0]

		// Cancel the context
		if cancelFunc, ok := h.cancelFuncs.Load(taskID); ok {
			cancelFunc.(context.CancelFunc)()
			h.cancelFuncs.Delete(taskID)
		}

		// Also tell the runner to cancel
		if err := h.runner.CancelTask(taskID); err != nil {
			log.Printf("Failed to cancel task: %v", err)
			http.Error(w, fmt.Sprintf("Failed to cancel task: %v", err), http.StatusInternalServerError)
			return
		}

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

		// Extract task ID from URL path
		pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/evaluation/stream/"), "/")
		if len(pathParts) == 0 || pathParts[0] == "" {
			http.Error(w, "Task ID required", http.StatusBadRequest)
			return
		}
		taskID := pathParts[0]

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
		clientChan := make(chan models.ProgressUpdate, 10)

		// Register client
		var clients *sync.Map
		if existing, ok := h.sseClients.Load(taskID); ok {
			clients = existing.(*sync.Map)
		} else {
			clients = &sync.Map{}
			h.sseClients.Store(taskID, clients)
		}
		clients.Store(clientID, clientChan)

		// Clean up on disconnect
		defer func() {
			clients.Delete(clientID)
			close(clientChan)
		}()

		// Send initial connection message
		fmt.Fprintf(w, "event: connected\ndata: {\"task_id\":\"%s\"}\n\n", taskID)
		flusher.Flush()

		// Stream updates
		ctx := r.Context()
		for {
			select {
			case <-ctx.Done():
				return
			case update, ok := <-clientChan:
				if !ok {
					return
				}
				data, err := json.Marshal(update)
				if err != nil {
					log.Printf("Error marshaling progress update: %v", err)
					continue
				}
				fmt.Fprintf(w, "event: progress\ndata: %s\n\n", data)
				flusher.Flush()

				// Close stream if task completed
				if update.ProgressPercent >= 100 {
					fmt.Fprintf(w, "event: completed\ndata: {\"task_id\":\"%s\"}\n\n", taskID)
					flusher.Flush()
					return
				}
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

		// Extract task ID from URL path
		pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/evaluation/results/"), "/")
		if len(pathParts) == 0 || pathParts[0] == "" {
			http.Error(w, "Task ID required", http.StatusBadRequest)
			return
		}
		taskID := pathParts[0]

		// Get task to check status
		task, err := h.db.GetTask(taskID)
		if err != nil {
			log.Printf("Failed to get task: %v", err)
			http.Error(w, fmt.Sprintf("Failed to get task: %v", err), http.StatusInternalServerError)
			return
		}
		if task == nil {
			http.Error(w, "Task not found", http.StatusNotFound)
			return
		}

		// Get results
		results, err := h.db.GetResults(taskID)
		if err != nil {
			log.Printf("Failed to get results: %v", err)
			http.Error(w, fmt.Sprintf("Failed to get results: %v", err), http.StatusInternalServerError)
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
			log.Printf("Error encoding response: %v", err)
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

		// Extract task ID from URL path
		pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/evaluation/export/"), "/")
		if len(pathParts) == 0 || pathParts[0] == "" {
			http.Error(w, "Task ID required", http.StatusBadRequest)
			return
		}
		taskID := pathParts[0]

		format := models.ExportFormat(r.URL.Query().Get("format"))
		if format == "" {
			format = models.ExportJSON
		}

		data, contentType, err := h.runner.ExportResults(taskID, format)
		if err != nil {
			log.Printf("Failed to export results: %v", err)
			http.Error(w, fmt.Sprintf("Failed to export results: %v", err), http.StatusInternalServerError)
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
			log.Printf("Error encoding response: %v", err)
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

		metricName := r.URL.Query().Get("metric")
		if metricName == "" {
			http.Error(w, "Metric name is required", http.StatusBadRequest)
			return
		}

		limit := 100 // Default limit
		if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
			_, _ = fmt.Sscanf(limitStr, "%d", &limit)
		}

		entries, err := h.db.GetHistoryForMetric(metricName, limit)
		if err != nil {
			log.Printf("Failed to get history: %v", err)
			http.Error(w, fmt.Sprintf("Failed to get history: %v", err), http.StatusInternalServerError)
			return
		}

		if entries == nil {
			entries = []*models.EvaluationHistoryEntry{}
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(entries); err != nil {
			log.Printf("Error encoding response: %v", err)
		}
	}
}
