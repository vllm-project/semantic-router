package config

import (
	"fmt"

	"gopkg.in/yaml.v3"
)

// routerReplayForms are the forms of shadow traffic before router_replay
// owned it: the shadow_dispatch plugin, now the shadow block of the
// decision's router_replay plugin, which records its outcome.
var routerReplayForms = []ConfigForm{
	{
		Code:    CodeRemovedPluginType,
		Paths:   RoutingScopes("decisions[].plugins[]"),
		Match:   hasType("shadow_dispatch"),
		What:    "plugin shadow_dispatch is removed",
		Becomes: RemovedPluginTypes["shadow_dispatch"],
		Migrate: migrateShadowDispatchPlugin,
	},
}

// migrateShadowDispatchPlugin moves an enabled shadow_dispatch plugin's
// settings into the shadow block of the decision's router_replay plugin, or
// makes it a router_replay plugin that records the decision. A disabled one
// sent nothing.
func migrateShadowDispatchPlugin(m *Migration, at FormMatch) error {
	plugins := at.Parent
	dispatch := mappingValue(at.Node, "configuration")
	if !yamlTrue(mappingValue(dispatch, "enabled")) {
		removeItem(plugins, at.Node)
		return nil
	}
	shadow := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	for i := 0; i+1 < len(dispatch.Content); i += 2 {
		if key := dispatch.Content[i].Value; key != "enabled" {
			m.Set(shadow, key, dispatch.Content[i+1])
		}
	}
	for _, item := range plugins.Content {
		if scalarAt(item, "type") != DecisionPluginRouterReplay {
			continue
		}
		replay := mappingValue(item, "configuration")
		if replay == nil {
			replay = &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
			m.Set(item, "configuration", replay)
		}
		if mappingValue(replay, "shadow") != nil {
			return fmt.Errorf("the decision's router_replay plugin already has a shadow")
		}
		if enabled := mappingValue(replay, "enabled"); enabled != nil && !yamlTrue(enabled) {
			m.Note(at.Path, "the decision's router_replay plugin is disabled, and a shadow runs only while the decision records replays; enable it to keep the shadow")
		}
		m.Set(replay, "shadow", shadow)
		removeItem(plugins, at.Node)
		return nil
	}
	replay := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	if err := m.SetValue(replay, "enabled", true); err != nil {
		return err
	}
	m.Set(replay, "shadow", shadow)
	m.Set(at.Node, "type", Scalar(DecisionPluginRouterReplay))
	m.Set(at.Node, "configuration", replay)
	m.Changed(at.Path, "router_replay now records this decision, so that the replay record keeps the shadow's outcome")
	return nil
}
