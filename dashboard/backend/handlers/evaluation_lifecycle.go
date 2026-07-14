package handlers

import (
	"errors"
	"strings"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

const (
	evaluationMaxSSEClientsPerTask = 16
	evaluationMaxSSEClientsGlobal  = 256
)

var (
	errEvaluationSSETaskLimit   = errors.New("too many progress streams for this task")
	errEvaluationSSEGlobalLimit = errors.New("too many progress streams")
)

type evaluationSSEClient struct {
	updates  chan models.ProgressUpdate
	terminal chan models.EvaluationStatus
}

type evaluationSSETaskRegistry struct {
	clients map[string]*evaluationSSEClient
}

func validEvaluationTaskID(taskID string) bool {
	return taskID != "" && uuid.Validate(taskID) == nil
}

func validEvaluationStatusFilter(status string) bool {
	switch models.EvaluationStatus(status) {
	case models.StatusPending, models.StatusRunning, models.StatusCompleted, models.StatusFailed, models.StatusCancelled:
		return true
	default:
		return status == ""
	}
}

func evaluationTaskIDFromPath(path, prefix string) (string, bool) {
	if !strings.HasPrefix(path, prefix) {
		return "", false
	}
	taskID := strings.TrimPrefix(path, prefix)
	if strings.Contains(taskID, "/") || !validEvaluationTaskID(taskID) {
		return "", false
	}
	return taskID, true
}

func (h *EvaluationHandler) registerEvaluationSSEClient(taskID string) (string, *evaluationSSEClient, error) {
	h.sseMu.Lock()
	defer h.sseMu.Unlock()

	if h.sseTotal >= evaluationMaxSSEClientsGlobal {
		return "", nil, errEvaluationSSEGlobalLimit
	}
	registry := h.sseClients[taskID]
	if registry == nil {
		registry = &evaluationSSETaskRegistry{clients: make(map[string]*evaluationSSEClient)}
		h.sseClients[taskID] = registry
	}
	if len(registry.clients) >= evaluationMaxSSEClientsPerTask {
		return "", nil, errEvaluationSSETaskLimit
	}

	clientID := uuid.NewString()
	client := &evaluationSSEClient{
		updates:  make(chan models.ProgressUpdate, 16),
		terminal: make(chan models.EvaluationStatus, 1),
	}
	registry.clients[clientID] = client
	h.sseTotal++
	return clientID, client, nil
}

func (h *EvaluationHandler) unregisterEvaluationSSEClient(taskID, clientID string, expected *evaluationSSEClient) {
	h.sseMu.Lock()
	defer h.sseMu.Unlock()

	registry := h.sseClients[taskID]
	if registry == nil || registry.clients[clientID] != expected {
		return
	}
	delete(registry.clients, clientID)
	if h.sseTotal > 0 {
		h.sseTotal--
	}
	if len(registry.clients) == 0 {
		delete(h.sseClients, taskID)
	}
}

func (h *EvaluationHandler) broadcastEvaluationProgress(update models.ProgressUpdate) {
	h.sseMu.Lock()
	registry := h.sseClients[update.TaskID]
	clients := make([]*evaluationSSEClient, 0)
	if registry != nil {
		clients = make([]*evaluationSSEClient, 0, len(registry.clients))
		for _, client := range registry.clients {
			clients = append(clients, client)
		}
	}
	h.sseMu.Unlock()

	for _, client := range clients {
		select {
		case client.updates <- update:
		default:
			// Slow consumers lose intermediate progress updates; terminal status
			// uses a dedicated channel and therefore cannot be starved by them.
		}
	}
}

// finishEvaluationSSETask detaches the task-owned registry atomically, so a
// terminal task cannot retain an empty outer map entry. Channels are never
// closed: a broadcast that already copied a client remains safe.
func (h *EvaluationHandler) finishEvaluationSSETask(taskID string, status models.EvaluationStatus) {
	h.sseMu.Lock()
	registry := h.sseClients[taskID]
	if registry != nil {
		delete(h.sseClients, taskID)
		h.sseTotal -= len(registry.clients)
		if h.sseTotal < 0 {
			h.sseTotal = 0
		}
	}
	h.sseMu.Unlock()

	if registry == nil {
		return
	}
	for _, client := range registry.clients {
		select {
		case client.terminal <- status:
		default:
		}
	}
}

func (h *EvaluationHandler) finishEvaluationRun(taskID string) {
	h.cancelFuncs.Delete(taskID)
	h.runMu.Lock()
	defer h.runMu.Unlock()

	delete(h.activeRuns, taskID)
	status := models.StatusFailed
	if task, err := h.db.GetTask(taskID); err == nil && task != nil {
		status = task.Status
		if status == models.StatusRunning || status == models.StatusPending {
			_ = h.db.UpdateTaskStatus(taskID, models.StatusFailed, "Evaluation failed")
			status = models.StatusFailed
		}
	}
	h.finishEvaluationSSETask(taskID, status)
}
