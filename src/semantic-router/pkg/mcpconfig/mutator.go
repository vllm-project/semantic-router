package mcpconfig

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const maxBackups = 10

var configVersionPattern = regexp.MustCompile(`^[0-9]{8}-[0-9]{6}$`)

var deployMu sync.Mutex

// Mutator reads and writes router config through the same validation path as the apiserver.
type Mutator struct {
	configPath string
}

func NewMutator(configPath string) *Mutator {
	return &Mutator{configPath: configPath}
}

func (m *Mutator) paths() persistencePaths {
	return resolvePersistencePaths(m.configPath)
}

// GetDocument returns the current router config document.
func (m *Mutator) GetDocument() (map[string]any, error) {
	if m.configPath == "" {
		return nil, fmt.Errorf("router config path is not set")
	}
	paths := m.paths()
	data, err := os.ReadFile(paths.sourcePath)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	return decodeYAMLDocument(data)
}

// ValidateDocument validates a config document without persisting it.
func (m *Mutator) ValidateDocument(doc map[string]any, mode MutationMode) (map[string]any, error) {
	if m.configPath == "" {
		return nil, fmt.Errorf("router config path is not set")
	}
	paths := m.paths()
	_, yamlBytes, err := m.prepareMutationPayload(doc, paths.sourcePath, mode)
	if err != nil {
		return nil, err
	}
	validated, err := decodeYAMLDocument(yamlBytes)
	if err != nil {
		return nil, err
	}
	return validated, nil
}

// DiffDocument merges patch into the current config and returns before/after views.
func (m *Mutator) DiffDocument(patch map[string]any, mode MutationMode) (map[string]any, error) {
	if m.configPath == "" {
		return nil, fmt.Errorf("router config path is not set")
	}
	paths := m.paths()
	existingDoc, _, err := readConfigDocument(paths.sourcePath)
	if err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("read existing config: %w", err)
	}

	nextDoc := patch
	if mode == MutationMerge {
		nextDoc = mergeConfigDocuments(existingDoc, patch)
	}

	mergedYAML, err := normalizeRouterConfigDocument(nextDoc)
	if err != nil {
		return map[string]any{
			"valid":            false,
			"validation_error": err.Error(),
			"base":             existingDoc,
			"patch":            patch,
			"merged":           nextDoc,
		}, nil
	}

	mergedDoc, err := decodeYAMLDocument(mergedYAML)
	if err != nil {
		return nil, err
	}

	return map[string]any{
		"valid":  true,
		"base":   existingDoc,
		"patch":  patch,
		"merged": mergedDoc,
	}, nil
}

// ApplyPatch persists a config mutation using merge or replace semantics.
func (m *Mutator) ApplyPatch(patch map[string]any, mode MutationMode, dsl string) (*ApplyResult, error) {
	if m.configPath == "" {
		return nil, fmt.Errorf("router config path is not set")
	}
	if !deployMu.TryLock() {
		return nil, fmt.Errorf("another config update operation is in progress")
	}
	defer deployMu.Unlock()

	paths := m.paths()
	yamlBytes, existingData, err := m.prepareMutationPayload(patch, paths.sourcePath, mode)
	if err != nil {
		return nil, err
	}

	version, backupDir := recordConfigArtifacts(paths.sourcePath, existingData, dsl)
	if err := writeConfigFiles(paths, yamlBytes); err != nil {
		return nil, err
	}
	cleanupBackups(backupDir)

	logging.Infof(
		"Router config %s via MCP: version=%s, size=%d bytes, source=%s",
		mode,
		version,
		len(yamlBytes),
		paths.sourcePath,
	)

	return &ApplyResult{
		Status:  "success",
		Version: version,
		Message: mutationMessage(mode),
	}, nil
}

// ApplyResult is returned after a successful config write.
type ApplyResult struct {
	Status  string `json:"status"`
	Version string `json:"version"`
	Message string `json:"message,omitempty"`
}

type MutationMode string

const (
	MutationMerge   MutationMode = "merge"
	MutationReplace MutationMode = "replace"
)

func (m *Mutator) prepareMutationPayload(
	patchDoc map[string]any,
	sourceConfigPath string,
	mode MutationMode,
) ([]byte, []byte, error) {
	existingDoc, existingData, err := readConfigDocument(sourceConfigPath)
	if err != nil && !os.IsNotExist(err) {
		return nil, nil, fmt.Errorf("read existing config: %w", err)
	}

	nextDoc := patchDoc
	if mode == MutationMerge {
		nextDoc = mergeConfigDocuments(existingDoc, patchDoc)
	}

	yamlBytes, err := normalizeRouterConfigDocument(nextDoc)
	if err != nil {
		return nil, nil, err
	}
	return yamlBytes, existingData, nil
}

func normalizeRouterConfigDocument(doc map[string]any) ([]byte, error) {
	rawYAML, err := yaml.Marshal(doc)
	if err != nil {
		return nil, fmt.Errorf("encode router config: %w", err)
	}

	parsedCfg, err := config.ParseYAMLBytes(rawYAML)
	if err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}

	canonicalCfg := config.CanonicalConfigFromRouterConfig(parsedCfg)
	yamlBytes, err := yaml.Marshal(canonicalCfg)
	if err != nil {
		return nil, fmt.Errorf("normalize router config: %w", err)
	}
	return yamlBytes, nil
}

func readConfigDocument(path string) (map[string]any, []byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return map[string]any{}, nil, err
		}
		return nil, nil, err
	}
	doc, err := decodeYAMLDocument(data)
	if err != nil {
		return nil, nil, err
	}
	return doc, data, nil
}

func decodeYAMLDocument(data []byte) (map[string]any, error) {
	var doc map[string]any
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("decode yaml: %w", err)
	}
	if doc == nil {
		return map[string]any{}, nil
	}
	return doc, nil
}

func mergeConfigDocuments(base map[string]any, patch map[string]any) map[string]any {
	merged, ok := mergeYAMLValue(base, patch).(map[string]any)
	if !ok {
		return map[string]any{}
	}
	return merged
}

func mergeYAMLValue(base any, patch any) any {
	if patch == nil {
		return nil
	}

	patchMap, ok := patch.(map[string]any)
	if !ok {
		return cloneYAMLValue(patch)
	}

	merged := map[string]any{}
	if baseMap, ok := base.(map[string]any); ok {
		for key, value := range baseMap {
			merged[key] = cloneYAMLValue(value)
		}
	}

	for key, value := range patchMap {
		if value == nil {
			delete(merged, key)
			continue
		}
		merged[key] = mergeYAMLValue(merged[key], value)
	}
	return merged
}

func cloneYAMLValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		cloned := make(map[string]any, len(typed))
		for key, nested := range typed {
			cloned[key] = cloneYAMLValue(nested)
		}
		return cloned
	case []any:
		cloned := make([]any, len(typed))
		for i, nested := range typed {
			cloned[i] = cloneYAMLValue(nested)
		}
		return cloned
	default:
		return typed
	}
}

func recordConfigArtifacts(sourceConfigPath string, existingData []byte, dsl string) (string, string) {
	configDir := filepath.Dir(sourceConfigPath)
	backupDir := filepath.Join(configDir, ".vllm-sr", "config-backups")
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		logging.Warnf("Failed to create backup directory: %v", err)
	}

	version := time.Now().Format("20060102-150405")
	if len(existingData) > 0 {
		backupFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", version))
		if err := os.WriteFile(backupFile, existingData, 0o644); err != nil {
			logging.Warnf("Failed to create backup: %v", err)
		}
	}

	if strings.TrimSpace(dsl) != "" {
		dslDir := filepath.Join(configDir, ".vllm-sr")
		dslFile := filepath.Join(dslDir, "config.dsl")
		if err := os.WriteFile(dslFile, []byte(dsl), 0o644); err != nil {
			logging.Warnf("Failed to archive DSL source: %v", err)
		}
	}

	return version, backupDir
}

func writeConfigFiles(paths persistencePaths, yamlBytes []byte) error {
	if err := writeConfigAtomically(paths.sourcePath, yamlBytes); err != nil {
		return fmt.Errorf("write source config: %w", err)
	}
	if !paths.usesRuntimeOverride() {
		return nil
	}
	return fmt.Errorf("runtime config sync is not available from MCP config tools; write source config at %s", paths.sourcePath)
}

func writeConfigAtomically(configPath string, yamlBytes []byte) error {
	tmpConfigFile := configPath + ".tmp"
	if err := os.WriteFile(tmpConfigFile, yamlBytes, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmpConfigFile, configPath); err != nil {
		if writeErr := os.WriteFile(configPath, yamlBytes, 0o644); writeErr != nil {
			return writeErr
		}
	}
	return nil
}

func cleanupBackups(backupDir string) {
	entries, err := os.ReadDir(backupDir)
	if err != nil {
		return
	}

	var backups []os.DirEntry
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasPrefix(entry.Name(), "config.") && strings.HasSuffix(entry.Name(), ".yaml") {
			backups = append(backups, entry)
		}
	}

	if len(backups) <= maxBackups {
		return
	}

	sort.Slice(backups, func(i, j int) bool {
		return backups[i].Name() < backups[j].Name()
	})

	toRemove := len(backups) - maxBackups
	for i := 0; i < toRemove; i++ {
		path := filepath.Join(backupDir, backups[i].Name())
		if err := os.Remove(path); err != nil {
			logging.Warnf("Failed to remove old backup %s: %v", path, err)
		}
	}
}

func mutationMessage(mode MutationMode) string {
	if mode == MutationMerge {
		return "Router config merged successfully. Router will reload automatically via fsnotify."
	}
	return "Router config replaced successfully. Router will reload automatically via fsnotify."
}

// ConfigVersionPattern exposes the backup version format for tests.
func ConfigVersionPattern() *regexp.Regexp {
	return configVersionPattern
}
