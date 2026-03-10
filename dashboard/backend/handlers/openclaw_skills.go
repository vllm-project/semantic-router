package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

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

func (h *OpenClawHandler) fetchSkillContent(skillID, baseImage string) string {
	containerPaths := []string{
		"/app/skills/" + skillID + "/SKILL.md",
		"/app/extensions/" + skillID + "/SKILL.md",
	}
	for _, p := range containerPaths {
		out, err := h.containerOutput("run", "--rm", baseImage, "cat", p)
		if err == nil && len(out) > 0 {
			return string(out)
		}
	}
	skills, err := h.loadSkills()
	if err != nil {
		return ""
	}
	for _, s := range skills {
		if s.ID == skillID {
			return fmt.Sprintf("---\nname: %s\ndescription: %q\nuser-invocable: true\n---\n\n# %s\n\n%s\n",
				s.ID, s.Description, s.Name, s.Description)
		}
	}
	return ""
}
