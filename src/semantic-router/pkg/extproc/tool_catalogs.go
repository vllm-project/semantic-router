package extproc

import (
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// toolCatalog is a catalog the tools plugin's add mode chooses from, and
// when it was read, the availability evidence of its trusted facts.
type toolCatalog struct {
	tools    []config.CatalogTool
	loadedAt time.Time
}

// loadToolCatalogs reads the catalog of every add-mode selection a request
// runs, a decision's or the default plugins', once per tools_db_path. A
// catalog the generation cannot read fails its build.
func loadToolCatalogs(cfg *config.RouterConfig) (map[string]*toolCatalog, error) {
	catalogs := map[string]*toolCatalog{}
	var err error
	cfg.VisitEveryPlugins(func(plugins []config.DecisionPlugin) {
		spec := config.ToolsPluginOf(plugins).ActiveSelect()
		if err != nil || spec == nil || spec.Mode != config.ToolsSelectModeAdd || catalogs[spec.ToolsDBPath] != nil {
			return
		}
		tools, loadErr := config.LoadToolCatalog(spec.ToolsDBPath)
		if loadErr != nil {
			err = fmt.Errorf("tools catalog %s: %w", spec.ToolsDBPath, loadErr)
			return
		}
		catalogs[spec.ToolsDBPath] = &toolCatalog{tools: tools, loadedAt: time.Now()}
	})
	if err != nil {
		return nil, err
	}
	return catalogs, nil
}
