package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/modelresearch"
)

type ModelResearchHandler struct {
	manager    *modelresearch.Manager
	sseClients sync.Map // map[campaignID]*sync.Map (clientID -> chan modelresearch.CampaignEvent)
}

func NewModelResearchHandler(manager *modelresearch.Manager) *ModelResearchHandler {
	handler := &ModelResearchHandler{manager: manager}
	go handler.broadcastLoop()
	return handler
}

func (h *ModelResearchHandler) broadcastLoop() {
	for update := range h.manager.Events() {
		value, ok := h.sseClients.Load(update.CampaignID)
		if !ok {
			continue
		}
		clients := value.(*sync.Map)
		clients.Range(func(_, clientValue any) bool {
			client := clientValue.(chan modelresearch.CampaignEvent)
			select {
			case client <- update.Event:
			default:
			}
			return true
		})
	}
}

func (h *ModelResearchHandler) RecipesHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(h.manager.Recipes())
	}
}

func (h *ModelResearchHandler) ListCampaignsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(h.manager.ListCampaigns())
	}
}

func (h *ModelResearchHandler) CreateCampaignHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req modelresearch.CreateCampaignRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
			return
		}

		campaign, err := h.manager.StartCampaign(req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(campaign)
	}
}

func (h *ModelResearchHandler) GetCampaignHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		id := strings.TrimPrefix(r.URL.Path, "/api/model-research/campaigns/")
		id = strings.TrimSuffix(id, "/")
		if id == "" {
			http.Error(w, "Campaign ID required", http.StatusBadRequest)
			return
		}

		campaign := h.manager.GetCampaign(id)
		if campaign == nil {
			http.Error(w, "Campaign not found", http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(campaign)
	}
}

func (h *ModelResearchHandler) StopCampaignHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		id := strings.TrimPrefix(r.URL.Path, "/api/model-research/campaigns/")
		id = strings.TrimSuffix(id, "/stop")
		id = strings.TrimSuffix(id, "/")
		if id == "" {
			http.Error(w, "Campaign ID required", http.StatusBadRequest)
			return
		}

		if err := h.manager.StopCampaign(id); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "stopping"})
	}
}

func (h *ModelResearchHandler) StreamEventsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		id := strings.TrimPrefix(r.URL.Path, "/api/model-research/campaigns/")
		id = strings.TrimSuffix(id, "/events")
		id = strings.TrimSuffix(id, "/")
		if id == "" {
			http.Error(w, "Campaign ID required", http.StatusBadRequest)
			return
		}

		campaign := h.manager.GetCampaign(id)
		if campaign == nil {
			http.Error(w, "Campaign not found", http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		clientID := fmt.Sprintf("%d", time.Now().UnixNano())
		clientChan := make(chan modelresearch.CampaignEvent, 16)
		value, _ := h.sseClients.LoadOrStore(id, &sync.Map{})
		clients := value.(*sync.Map)
		clients.Store(clientID, clientChan)
		defer func() {
			clients.Delete(clientID)
			close(clientChan)
		}()

		_, _ = fmt.Fprintf(w, "event: connected\ndata: {\"campaign_id\":\"%s\"}\n\n", id)
		flusher.Flush()

		for _, event := range campaign.Events {
			payload, err := json.Marshal(event)
			if err != nil {
				continue
			}
			_, _ = fmt.Fprintf(w, "event: event\ndata: %s\n\n", payload)
			flusher.Flush()
		}

		if campaign.Status == modelresearch.StatusCompleted || campaign.Status == modelresearch.StatusFailed || campaign.Status == modelresearch.StatusStopped || campaign.Status == modelresearch.StatusBlocked {
			_, _ = fmt.Fprintf(w, "event: completed\ndata: {\"campaign_id\":\"%s\"}\n\n", id)
			flusher.Flush()
			return
		}

		heartbeat := time.NewTicker(15 * time.Second)
		defer heartbeat.Stop()

		for {
			select {
			case <-r.Context().Done():
				return
			case <-heartbeat.C:
				_, _ = fmt.Fprint(w, ": heartbeat\n\n")
				flusher.Flush()
			case event := <-clientChan:
				payload, err := json.Marshal(event)
				if err != nil {
					continue
				}
				_, _ = fmt.Fprintf(w, "event: event\ndata: %s\n\n", payload)
				if event.Percent >= 100 && (event.Kind == modelresearch.EventStatus || event.Kind == modelresearch.EventProgress) {
					_, _ = fmt.Fprintf(w, "event: completed\ndata: {\"campaign_id\":\"%s\"}\n\n", id)
					flusher.Flush()
					return
				}
				flusher.Flush()
			}
		}
	}
}
