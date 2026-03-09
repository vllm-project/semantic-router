package authoring

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var supportedRuntimeAuthoringTopLevelKeys = map[string]struct{}{
	"version":   {},
	"listeners": {},
	"signals":   {},
	"providers": {},
	"decisions": {},
}

// LoadRuntimeCompatibleConfig loads either the canonical authoring first slice
// or the legacy runtime RouterConfig file format for router file-based startup.
func LoadRuntimeCompatibleConfig(path string) (*routerconfig.RouterConfig, error) {
	resolved, data, err := readConfigFile(path)
	if err != nil {
		return nil, err
	}

	shape, unsupported, err := classifyRuntimeConfigShape(data)
	if err != nil {
		// Fall back to the legacy runtime parser so syntax errors still surface
		// through the long-standing config.Parse error path.
		return routerconfig.Parse(path)
	}
	if shape == runtimeConfigShapeLegacy {
		return routerconfig.Parse(path)
	}
	if len(unsupported) > 0 {
		return nil, fmt.Errorf(
			"canonical authoring config in %s contains unsupported top-level keys for direct router runtime load: %s; supported keys today: %s",
			resolved,
			strings.Join(unsupported, ", "),
			strings.Join(sortedAuthoringKeys(supportedRuntimeAuthoringTopLevelKeys), ", "),
		)
	}

	cfg, err := Parse(data)
	if err != nil {
		return nil, err
	}
	runtimeCfg, err := CompileRuntime(cfg)
	if err != nil {
		return nil, err
	}
	if err := routerconfig.FinalizeParsedConfig(runtimeCfg); err != nil {
		return nil, err
	}
	return runtimeCfg, nil
}

type runtimeConfigShape int

const (
	runtimeConfigShapeLegacy runtimeConfigShape = iota
	runtimeConfigShapeAuthoring
)

func readConfigFile(path string) (string, []byte, error) {
	resolved, _ := filepath.EvalSymlinks(path)
	if resolved == "" {
		resolved = path
	}
	data, err := os.ReadFile(resolved)
	if err != nil {
		return resolved, nil, fmt.Errorf("failed to read config file: %w", err)
	}
	return resolved, data, nil
}

func classifyRuntimeConfigShape(data []byte) (runtimeConfigShape, []string, error) {
	var raw map[string]interface{}
	if err := yaml.Unmarshal(data, &raw); err != nil {
		return runtimeConfigShapeLegacy, nil, err
	}
	if _, ok := raw["providers"]; !ok {
		return runtimeConfigShapeLegacy, nil, nil
	}

	var unsupported []string
	for key := range raw {
		if _, ok := supportedRuntimeAuthoringTopLevelKeys[key]; !ok {
			unsupported = append(unsupported, key)
		}
	}
	sort.Strings(unsupported)
	return runtimeConfigShapeAuthoring, unsupported, nil
}

func sortedAuthoringKeys(keys map[string]struct{}) []string {
	names := make([]string, 0, len(keys))
	for key := range keys {
		names = append(names, key)
	}
	sort.Strings(names)
	return names
}
