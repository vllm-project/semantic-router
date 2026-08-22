package handlers

import (
	"database/sql"
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

const recipeDraftsPath = "/api/recipe-drafts"

func RecipeDraftsHandler(store *workflowstore.Store) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		name, err := recipeDraftName(r.URL.Path)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		switch r.Method {
		case http.MethodGet:
			if name == "" {
				items, listErr := store.ListRecipeDrafts(r.Context())
				if listErr != nil {
					http.Error(w, "could not load recipe drafts", http.StatusInternalServerError)
					return
				}
				writeRecipeDraftJSON(w, map[string]any{"items": items})
				return
			}
			item, getErr := store.GetRecipeDraft(r.Context(), name)
			if errors.Is(getErr, sql.ErrNoRows) {
				http.Error(w, "recipe draft not found", http.StatusNotFound)
				return
			}
			if getErr != nil {
				http.Error(w, "could not load recipe draft", http.StatusInternalServerError)
				return
			}
			writeRecipeDraftJSON(w, item)
		case http.MethodPost, http.MethodPut:
			var item workflowstore.RecipeDraft
			if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 2<<20)).Decode(&item); err != nil {
				http.Error(w, "invalid recipe draft", http.StatusBadRequest)
				return
			}
			if name != "" {
				if item.Name != "" && strings.TrimSpace(item.Name) != name {
					http.Error(w, "recipe name does not match request path", http.StatusBadRequest)
					return
				}
				item.Name = name
			}
			saved, saveErr := store.SaveRecipeDraft(r.Context(), item)
			if saveErr != nil {
				http.Error(w, saveErr.Error(), http.StatusBadRequest)
				return
			}
			writeRecipeDraftJSON(w, saved)
		case http.MethodDelete:
			if name == "" {
				http.Error(w, "recipe name is required", http.StatusBadRequest)
				return
			}
			if err := store.DeleteRecipeDraft(r.Context(), name); errors.Is(err, sql.ErrNoRows) {
				http.Error(w, "recipe draft not found", http.StatusNotFound)
				return
			} else if err != nil {
				http.Error(w, "could not delete recipe draft", http.StatusInternalServerError)
				return
			}
			w.WriteHeader(http.StatusNoContent)
		default:
			w.Header().Set("Allow", "GET, POST, PUT, DELETE, OPTIONS")
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})
}

func recipeDraftName(path string) (string, error) {
	remainder := strings.Trim(strings.TrimPrefix(path, recipeDraftsPath), "/")
	if remainder == "" {
		return "", nil
	}
	if strings.Contains(remainder, "/") {
		return "", errors.New("invalid recipe draft path")
	}
	return url.PathUnescape(remainder)
}

func writeRecipeDraftJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(value); err != nil {
		http.Error(w, "could not encode recipe draft", http.StatusInternalServerError)
	}
}
