package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
)

// ToolsDBHandler reads and serves the tools_db.json file
func ToolsDBHandler(configDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Only allow GET requests
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Construct the tools_db.json path
		toolsDBPath := filepath.Join(configDir, "tools_db.json")

		// Read the tools database file
		data, err := os.ReadFile(toolsDBPath)
		if err != nil {
			log.Printf("Error reading tools_db.json: %v", err)
			http.Error(w, fmt.Sprintf("Failed to read tools database: %v", err), http.StatusInternalServerError)
			return
		}

		// Parse JSON to validate it
		var tools interface{}
		if err := json.Unmarshal(data, &tools); err != nil {
			log.Printf("Error parsing tools_db.json: %v", err)
			http.Error(w, fmt.Sprintf("Failed to parse tools database: %v", err), http.StatusInternalServerError)
			return
		}

		// Send response
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(tools); err != nil {
			log.Printf("Error encoding tools to JSON: %v", err)
			http.Error(w, fmt.Sprintf("Failed to encode tools: %v", err), http.StatusInternalServerError)
			return
		}
	}
}
