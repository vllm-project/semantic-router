package handlers

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const defaultOpenClawClawHubSkill = "sonoscli"

func defaultOpenClawClawHubSkills() []string {
	raw, hasEnv := os.LookupEnv("OPENCLAW_DEFAULT_CLAWHUB_SKILLS")
	if !hasEnv {
		return []string{defaultOpenClawClawHubSkill}
	}

	return normalizeOpenClawClawHubSkills(raw)
}

func normalizeOpenClawClawHubSkills(raw string) []string {
	trimmed := strings.TrimSpace(raw)
	switch strings.ToLower(trimmed) {
	case "", "none", "off", "disabled", "false":
		return nil
	}

	fields := strings.FieldsFunc(trimmed, func(r rune) bool {
		return r == ',' || r == '\n' || r == '\r' || r == '\t' || r == ' '
	})

	seen := make(map[string]struct{}, len(fields))
	skills := make([]string, 0, len(fields))
	for _, field := range fields {
		skill := strings.TrimSpace(field)
		if skill == "" {
			continue
		}
		if _, ok := seen[skill]; ok {
			continue
		}
		seen[skill] = struct{}{}
		skills = append(skills, skill)
	}
	return skills
}

func (h *OpenClawHandler) installDefaultClawHubSkills(workspaceDir, baseImage string) error {
	for _, skillSlug := range defaultOpenClawClawHubSkills() {
		if err := h.installClawHubSkill(workspaceDir, baseImage, skillSlug); err != nil {
			return err
		}
	}
	return nil
}

func (h *OpenClawHandler) installClawHubSkill(workspaceDir, baseImage, skillSlug string) error {
	workspaceDir = strings.TrimSpace(workspaceDir)
	baseImage = strings.TrimSpace(baseImage)
	skillSlug = strings.TrimSpace(skillSlug)
	if workspaceDir == "" || baseImage == "" || skillSlug == "" {
		return nil
	}

	absWorkspaceDir, err := filepath.Abs(workspaceDir)
	if err != nil {
		return fmt.Errorf("failed to resolve workspace path for ClawHub install: %w", err)
	}

	args := []string{
		"run", "--rm",
		"--entrypoint", "npx",
		"-v", absWorkspaceDir + ":/workspace",
		"-e", "HOME=/tmp",
		"-e", "npm_config_cache=/tmp/.npm",
		baseImage,
		"--yes",
		"clawhub@latest",
		"--workdir", "/workspace",
		"--no-input",
		"install",
		skillSlug,
		"--force",
	}

	out, err := h.containerCombinedOutput(args...)
	if err != nil {
		detail := strings.TrimSpace(string(out))
		if detail == "" {
			detail = err.Error()
		}
		return fmt.Errorf("failed to install default ClawHub skill %q: %s", skillSlug, detail)
	}
	return nil
}
