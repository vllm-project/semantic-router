package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
)

// ToolsDBHandler reads and serves the tools_db.json file
func ToolsDBHandler(toolsDBPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Only allow GET requests
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Read the tools database file
		data, err := os.ReadFile(toolsDBPath)
		if err != nil {
			// Return 404 if file doesn't exist, 500 for other errors
			if os.IsNotExist(err) {
				log.Printf("tools database not found at %s: %v", toolsDBPath, err)
				http.Error(w, "Tools database not found", http.StatusNotFound)
			} else {
				log.Printf("failed to read tools database from %s: %v", toolsDBPath, err)
				http.Error(w, "Failed to read tools database", http.StatusInternalServerError)
			}
			return
		}

		// Parse JSON to validate it
		var tools interface{}
		if err := json.Unmarshal(data, &tools); err != nil {
			log.Printf("failed to parse tools database from %s: %v", toolsDBPath, err)
			http.Error(w, "Failed to parse tools database", http.StatusInternalServerError)
			return
		}

		// Send response
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(tools); err != nil {
			log.Printf("failed to encode tools database: %v", err)
			http.Error(w, "Failed to encode tools", http.StatusInternalServerError)
			return
		}
	}
}
