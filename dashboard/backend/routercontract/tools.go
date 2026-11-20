package routercontract

import routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// ToolSelection exposes the dashboard-owned view of router tool configuration.
type ToolSelection struct {
	ToolsDBPath string
}

// ToolSelectionFromConfig reads the stable dashboard-facing tool contract from
// a canonical router config: the catalog of the first decision whose tools
// plugin adds tools from one, empty when none does.
func ToolSelectionFromConfig(cfg *routerconfig.RouterConfig) ToolSelection {
	for _, decision := range cfg.AllRoutingDecisions() {
		if spec := decision.GetToolsConfig().ActiveSelect(); spec != nil && spec.Mode == routerconfig.ToolsSelectModeAdd {
			return ToolSelection{ToolsDBPath: spec.ToolsDBPath}
		}
	}
	return ToolSelection{}
}
