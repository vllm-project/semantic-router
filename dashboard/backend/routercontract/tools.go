package routercontract

import routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// ToolSelection exposes the dashboard-owned view of router tool configuration.
type ToolSelection struct {
	ToolsDBPath string
}

// ReadToolSelection reads the stable dashboard-facing tool contract from a
// canonical router config file: the catalog of the first decision whose tools
// plugin adds tools from one, empty when none does.
func ReadToolSelection(configPath string) (ToolSelection, error) {
	cfg, err := routerconfig.Parse(configPath)
	if err != nil {
		return ToolSelection{}, err
	}
	for _, decision := range cfg.AllRoutingDecisions() {
		if spec := decision.GetToolsConfig().ActiveSelect(); spec != nil && spec.Mode == routerconfig.ToolsSelectModeAdd {
			return ToolSelection{ToolsDBPath: spec.ToolsDBPath}, nil
		}
	}
	return ToolSelection{}, nil
}
