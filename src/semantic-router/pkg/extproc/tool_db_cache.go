package extproc

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

func (r *OpenAIRouter) resolveToolDatabasePath(path string) string {
	if r == nil || r.Config == nil {
		return strings.TrimSpace(path)
	}
	path = strings.TrimSpace(path)
	if path == "" {
		path = strings.TrimSpace(r.Config.Tools.ToolsDBPath)
	}
	if path == "" {
		return ""
	}
	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}
	base := strings.TrimSpace(r.Config.ConfigBaseDir)
	if base == "" {
		return filepath.Clean(path)
	}
	return filepath.Clean(filepath.Join(base, path))
}

func (r *OpenAIRouter) getOrLoadToolDatabaseForSelection(absPath string) (*tools.ToolsDatabase, error) {
	if r == nil || r.Config == nil {
		return nil, fmt.Errorf("tool_selection: router configuration missing")
	}
	if absPath == "" {
		return nil, fmt.Errorf("tool_selection: tools database path resolved empty")
	}

	primaryAbs := r.resolveToolDatabasePath(r.Config.Tools.ToolsDBPath)
	if r.ToolsDatabase != nil && absPath == primaryAbs && r.ToolsDatabase.IsEnabled() {
		return r.ToolsDatabase, nil
	}

	r.toolSelectionDBMu.Lock()
	defer r.toolSelectionDBMu.Unlock()
	if r.toolSelectionDBByPath == nil {
		r.toolSelectionDBByPath = make(map[string]*tools.ToolsDatabase)
	}
	if db, ok := r.toolSelectionDBByPath[absPath]; ok && db != nil {
		return db, nil
	}

	emb := r.Config.EmbeddingModels
	provider, err := toolsEmbeddingProvider(r.Config)
	if err != nil {
		return nil, err
	}
	db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
		Enabled:             true,
		SimilarityThreshold: 0,
		ModelType:           emb.EmbeddingConfig.ModelType,
		TargetDimension:     emb.EmbeddingConfig.TargetDimension,
		Provider:            provider,
	})
	if err := db.LoadToolsFromFile(absPath); err != nil {
		return nil, err
	}
	r.toolSelectionDBByPath[absPath] = db
	logging.Infof("tool_selection: loaded tools database %q (%d tools)", absPath, db.GetToolCount())
	return db, nil
}

// closeToolSelectionDatabases releases the tool databases getOrLoadToolDatabaseForSelection
// built for decisions pointing at a non-primary tools_db_path. Each carries its
// own embedding provider, which is an io.Closer against a remote embedding
// backend, so a decision using tool_selection would otherwise leak one provider
// client per reload. The primary database is not in this map — it is owned and
// closed by the build — so nothing here is closed twice.
//
// The map is cleared as it is drained, both so a second call is a no-op and so a
// request arriving after teardown reloads rather than reusing a closed database.
func (r *OpenAIRouter) closeToolSelectionDatabases() error {
	if r == nil {
		return nil
	}

	r.toolSelectionDBMu.Lock()
	databases := r.toolSelectionDBByPath
	r.toolSelectionDBByPath = nil
	r.toolSelectionDBMu.Unlock()

	var errs []error
	for path, db := range databases {
		if err := db.Close(); err != nil {
			errs = append(errs, fmt.Errorf("tool_selection: closing tools database %q: %w", path, err))
		}
	}
	return errors.Join(errs...)
}

func (r *OpenAIRouter) toolDatabaseForSelectionPlugin(sel *config.ToolSelectionPluginConfig) (*tools.ToolsDatabase, bool, error) {
	if r == nil || sel == nil {
		return nil, false, fmt.Errorf("tool_selection: invalid selection config")
	}
	resolved := r.resolveToolDatabasePath(sel.ToolsDBPath)
	if resolved == "" {
		return nil, false, fmt.Errorf("tool_selection add mode: tools_db_path and global tools.tools_db_path are empty")
	}

	primaryAbs := r.resolveToolDatabasePath(r.Config.Tools.ToolsDBPath)
	if resolved == primaryAbs && r.ToolsDatabase != nil && r.ToolsDatabase.IsEnabled() {
		return r.ToolsDatabase, false, nil
	}

	db, err := r.getOrLoadToolDatabaseForSelection(resolved)
	if err != nil {
		return nil, false, err
	}
	return db, true, nil
}
