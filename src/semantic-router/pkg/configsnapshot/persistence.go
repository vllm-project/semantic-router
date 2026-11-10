package configsnapshot

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Deployment variables that relocate where a Router's configuration is
// persisted.
const (
	// RuntimeConfigPathEnv names the document the Router loads when a
	// deployment generates it from the source document.
	RuntimeConfigPathEnv = "VLLM_SR_RUNTIME_CONFIG_PATH"
	// SourceConfigPathEnv names the document the management API persists.
	SourceConfigPathEnv = "VLLM_SR_SOURCE_CONFIG_PATH"
	// HistoryDirEnv names the directory that holds the configuration
	// history; it must be absolute.
	HistoryDirEnv = "VLLM_SR_CONFIG_BACKUP_DIR"
)

// Persistence locates a Router's configuration on disk.
type Persistence struct {
	// Source is the document the management API persists.
	Source string
	// Runtime is the document the Router loads. It is Source unless the
	// deployment generates one from the other.
	Runtime string
	// HistoryDir holds the recorded versions.
	HistoryDir string
}

// ResolvePersistence locates the configuration of the Router that loads
// activeConfigPath.
func ResolvePersistence(activeConfigPath string) Persistence {
	active := filepath.Clean(activeConfigPath)
	source := strings.TrimSpace(os.Getenv(SourceConfigPathEnv))
	if source == "" {
		source = derivedSourcePath(active)
	}
	if source == "" {
		source = active
	}
	source = filepath.Clean(source)
	runtime := strings.TrimSpace(os.Getenv(RuntimeConfigPathEnv))
	if runtime == "" {
		runtime = active
	}
	return Persistence{Source: source, Runtime: filepath.Clean(runtime), HistoryDir: HistoryDir(source)}
}

// workspaceDocument is the name of the document a vllm-sr workspace holds.
const workspaceDocument = "config.yaml"

// HistoryDir is the history directory of the document persisted at
// sourcePath. Every document has its own, so Routers whose documents share a
// directory keep separate histories: <base>/.vllm-sr/config-backups for a
// workspace's config.yaml, where the management API and the Dashboard have
// kept its backups, and a directory named after the file inside it for any
// other document. HistoryDirEnv replaces it, as replicas that must not share
// a history do with per-replica state.
func HistoryDir(sourcePath string) string {
	if configured := strings.TrimSpace(os.Getenv(HistoryDirEnv)); configured != "" && filepath.IsAbs(configured) {
		return filepath.Clean(configured)
	}
	dir := filepath.Join(persistenceBaseDir(sourcePath), ".vllm-sr", "config-backups")
	if name := filepath.Base(filepath.Clean(sourcePath)); name != workspaceDocument {
		dir = filepath.Join(dir, name)
	}
	return dir
}

// persistenceBaseDir is the project directory of a source document: its
// directory, or the one above when the document is a generated
// .vllm-sr/runtime-config*.yaml.
func persistenceBaseDir(sourcePath string) string {
	if configured := strings.TrimSpace(os.Getenv(config.ConfigBaseDirEnv)); configured != "" && filepath.IsAbs(configured) {
		return filepath.Clean(configured)
	}
	cleaned := filepath.Clean(sourcePath)
	parent := filepath.Dir(cleaned)
	if isGeneratedRuntimeConfig(cleaned) {
		return filepath.Dir(parent)
	}
	return parent
}

// derivedSourcePath is the source document a generated runtime document is
// produced from, or "" when activePath is not generated.
func derivedSourcePath(activePath string) string {
	if filepath.Base(activePath) == workspaceDocument {
		return activePath
	}
	if isGeneratedRuntimeConfig(activePath) {
		return filepath.Join(filepath.Dir(filepath.Dir(activePath)), workspaceDocument)
	}
	return ""
}

func isGeneratedRuntimeConfig(path string) bool {
	base := filepath.Base(path)
	return filepath.Base(filepath.Dir(path)) == ".vllm-sr" &&
		strings.HasPrefix(base, "runtime-config") && strings.HasSuffix(base, ".yaml")
}
