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

// loadToolCatalogs reads the catalog of every add-mode selection, once per
// tools_db_path. A catalog the generation cannot read fails its build.
func loadToolCatalogs(cfg *config.RouterConfig) (map[string]*toolCatalog, error) {
	catalogs := map[string]*toolCatalog{}
	for _, decision := range cfg.AllRoutingDecisions() {
		spec := decision.GetToolsConfig().ActiveSelect()
		if spec == nil || spec.Mode != config.ToolsSelectModeAdd || catalogs[spec.ToolsDBPath] != nil {
			continue
		}
		tools, err := config.LoadToolCatalog(spec.ToolsDBPath)
		if err != nil {
			return nil, fmt.Errorf("decision %q: tools catalog %s: %w", decision.Name, spec.ToolsDBPath, err)
		}
		catalogs[spec.ToolsDBPath] = &toolCatalog{tools: tools, loadedAt: time.Now()}
	}
	return catalogs, nil
}
