package routercontract

import routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// ToolSelection exposes the dashboard-owned view of router tool configuration.
type ToolSelection struct {
	ToolsDBPath string
}

// ReadToolSelection reads the stable dashboard-facing tool contract from a
// canonical router config file.
func ReadToolSelection(configPath string) (ToolSelection, error) {
	cfg, err := routerconfig.Parse(configPath)
	if err != nil {
		return ToolSelection{}, err
	}
	return ToolSelection{
		ToolsDBPath: cfg.ToolSelection.Tools.ToolsDBPath,
	}, nil
}
