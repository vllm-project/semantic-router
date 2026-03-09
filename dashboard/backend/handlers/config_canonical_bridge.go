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

var _canonicalDashboardTopLevelKeys = map[string]struct{}{
	"api":                           {},
	"authz":                         {},
	"auto_model_name":               {},
	"bert_model":                    {},
	"classifier":                    {},
	"clear_route_cache":             {},
	"config_source":                 {},
	"decisions":                     {},
	"embedding_models":              {},
	"feedback_detector":             {},
	"hallucination_mitigation":      {},
	"image_gen_backends":            {},
	"include_config_models_in_list": {},
	"listeners":                     {},
	"looper":                        {},
	"memory":                        {},
	"modality_detector":             {},
	"model_selection":               {},
	"mom_registry":                  {},
	"observability":                 {},
	"prompt_guard":                  {},
	"provider_profiles":             {},
	"providers":                     {},
	"ratelimit":                     {},
	"response_api":                  {},
	"router_replay":                 {},
	"semantic_cache":                {},
	"signals":                       {},
	"strategy":                      {},
	"streamed_body_mode":            {},
	"streamed_body_timeout_sec":     {},
	"tools":                         {},
	"vector_store":                  {},
	"version":                       {},
	"max_streamed_body_bytes":       {},
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

func mergeCanonicalDashboardConfigWithPython(configPath string, configPatch map[string]interface{}) ([]byte, error) {
	payload, err := json.Marshal(configPatch)
	if err != nil {
		return nil, fmt.Errorf("failed to encode canonical dashboard config patch: %w", err)
	}

	output, err := runDashboardPythonBridge(configPath, fmt.Sprintf(`
import json
import logging
import sys

logging.disable(logging.CRITICAL)
sys.path.insert(0, CLI_ROOT)

from cli.dashboard_bridge import render_merged_dashboard_yaml

config_patch = json.loads(%q)
print(render_merged_dashboard_yaml(CONFIG_PATH, config_patch), end="")
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

	hasCanonicalKey := false
	for key := range configData {
		if _, isLegacy := _canonicalDashboardLegacyKeys[key]; isLegacy {
			return false
		}
		if _, isCanonical := _canonicalDashboardTopLevelKeys[key]; isCanonical {
			hasCanonicalKey = true
		}
	}

	return hasCanonicalKey
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
