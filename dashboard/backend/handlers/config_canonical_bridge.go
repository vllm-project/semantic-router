package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

var _canonicalDashboardLegacyKeys = map[string]struct{}{
	"categories":               {},
	"complexity_rules":         {},
	"context_rules":            {},
	"default_model":            {},
	"default_reasoning_effort": {},
	"embedding_rules":          {},
	"fact_check_rules":         {},
	"jailbreak":                {},
	"keyword_rules":            {},
	"language_rules":           {},
	"modality_rules":           {},
	"model_config":             {},
	"pii":                      {},
	"preference_rules":         {},
	"reasoning_families":       {},
	"role_bindings":            {},
	"user_feedback_rules":      {},
	"vllm_endpoints":           {},
}

func loadDashboardConfig(configPath string) (interface{}, error) {
	if config, err := loadCanonicalDashboardConfigWithPython(configPath); err == nil {
		return config, nil
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}

	var config interface{}
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, err
	}
	return config, nil
}

func loadCanonicalDashboardConfigWithPython(configPath string) (map[string]interface{}, error) {
	output, err := runDashboardPythonBridge(configPath, `
import json
import logging
import sys

logging.disable(logging.CRITICAL)
sys.path.insert(0, CLI_ROOT)

from cli.dashboard_bridge import load_dashboard_config

print(json.dumps(load_dashboard_config(CONFIG_PATH)))
`)
	if err != nil {
		return nil, err
	}

	var config map[string]interface{}
	if err := json.Unmarshal([]byte(output), &config); err != nil {
		return nil, fmt.Errorf("failed to decode canonical dashboard config: %w", err)
	}
	return config, nil
}

func renderCanonicalDashboardConfigWithPython(configData map[string]interface{}) ([]byte, error) {
	payload, err := json.Marshal(configData)
	if err != nil {
		return nil, fmt.Errorf("failed to encode canonical dashboard config: %w", err)
	}

	output, err := runDashboardPythonBridge("", fmt.Sprintf(`
import json
import logging
import sys

logging.disable(logging.CRITICAL)
sys.path.insert(0, CLI_ROOT)

from cli.dashboard_bridge import render_dashboard_yaml

config_data = json.loads(%q)
print(render_dashboard_yaml(config_data), end="")
`, string(payload)))
	if err != nil {
		return nil, err
	}
	return []byte(output), nil
}

func shouldMergeDashboardConfigCanonically(configData map[string]interface{}) bool {
	if len(configData) == 0 {
		return false
	}

	if isCanonicalDashboardFullConfig(configData) {
		return true
	}

	hasCanonicalKey := false
	for _, key := range []string{"providers", "signals", "decisions", "listeners"} {
		if _, ok := configData[key]; ok {
			hasCanonicalKey = true
			break
		}
	}
	if !hasCanonicalKey {
		return false
	}

	for key := range configData {
		if _, isLegacy := _canonicalDashboardLegacyKeys[key]; isLegacy {
			return false
		}
	}

	return true
}

func isCanonicalDashboardFullConfig(configData map[string]interface{}) bool {
	_, hasVersion := configData["version"]
	return hasVersion
}

func runDashboardPythonBridge(configPath string, pythonScript string) (string, error) {
	cliRoot := detectPythonCLIRoot()
	if cliRoot == "" {
		return "", fmt.Errorf("python CLI not available")
	}
	pythonBin := detectPythonCLIExecutable()

	script := strings.ReplaceAll(pythonScript, "CLI_ROOT", fmt.Sprintf("%q", cliRoot))
	script = strings.ReplaceAll(script, "CONFIG_PATH", fmt.Sprintf("%q", configPath))

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, pythonBin, "-c", script)
	if configPath != "" {
		cmd.Dir = filepath.Dir(configPath)
	}
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("python CLI dashboard bridge failed: %w (output: %s)", err, strings.TrimSpace(string(output)))
	}
	return string(output), nil
}
