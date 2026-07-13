package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const maxOpenClawSkillsPerWorker = 64

// --- Skills ---

func (h *OpenClawHandler) SkillsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		skills, err := h.loadSkills()
		if err != nil {
			log.Printf("Warning: failed to load skills config: %v", err)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte("[]"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(skills); err != nil {
			log.Printf("openclaw: skills encode error: %v", err)
		}
	}
}

func (h *OpenClawHandler) loadSkills() ([]SkillTemplate, error) {
	candidates := make([]string, 0, 12)
	if p := strings.TrimSpace(os.Getenv("OPENCLAW_SKILLS_PATH")); p != "" {
		candidates = append(candidates, p)
	}

	candidates = append(candidates,
		filepath.Join(h.dataDir, "skills.json"),
		filepath.Join(h.dataDir, "..", "..", "config", "openclaw-skills.json"),
		"/app/config/openclaw-skills.json",
		"/app/dashboard/backend/config/openclaw-skills.json",
		"./config/openclaw-skills.json",
	)

	if wd, err := os.Getwd(); err == nil {
		candidates = append(candidates, filepath.Join(wd, "config", "openclaw-skills.json"))
	}
	if exe, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exe)
		candidates = append(candidates,
			filepath.Join(exeDir, "config", "openclaw-skills.json"),
			filepath.Join(exeDir, "..", "config", "openclaw-skills.json"),
		)
	}

	seen := make(map[string]struct{}, len(candidates))
	for _, rawPath := range candidates {
		configPath := strings.TrimSpace(rawPath)
		if configPath == "" {
			continue
		}
		cleanPath := filepath.Clean(configPath)
		if _, ok := seen[cleanPath]; ok {
			continue
		}
		seen[cleanPath] = struct{}{}

		data, err := os.ReadFile(configPath)
		if err != nil {
			continue
		}
		var skills []SkillTemplate
		if err := json.Unmarshal(data, &skills); err != nil {
			return nil, fmt.Errorf("invalid %s: %w", configPath, err)
		}
		log.Printf("openclaw: loaded %d skills from %s", len(skills), configPath)
		return skills, nil
	}
	return []SkillTemplate{}, nil
}

func validateRequestedOpenClawSkills(skillIDs []string) ([]string, error) {
	if len(skillIDs) > maxOpenClawSkillsPerWorker {
		return nil, fmt.Errorf("at most %d skills may be provisioned", maxOpenClawSkillsPerWorker)
	}
	validated := make([]string, 0, len(skillIDs))
	seen := make(map[string]struct{}, len(skillIDs))
	for _, rawID := range skillIDs {
		skillID := strings.TrimSpace(rawID)
		if !validOpenClawSkillID(skillID) {
			return nil, errors.New("skill IDs must use 1-128 ASCII letters, digits, dot, underscore, or hyphen without path segments")
		}
		if _, exists := seen[skillID]; exists {
			continue
		}
		seen[skillID] = struct{}{}
		validated = append(validated, skillID)
	}
	return validated, nil
}

func validOpenClawSkillID(skillID string) bool {
	if skillID == "" || len(skillID) > 128 || skillID == "." || skillID == ".." {
		return false
	}
	for index := 0; index < len(skillID); index++ {
		value := skillID[index]
		if (value >= 'a' && value <= 'z') ||
			(value >= 'A' && value <= 'Z') ||
			(value >= '0' && value <= '9') ||
			(value == '.' && index > 0) || value == '_' || value == '-' {
			continue
		}
		return false
	}
	return true
}
