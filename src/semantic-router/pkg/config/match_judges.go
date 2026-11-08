package config

// MatchJudges are the judges of the match blocks a decision's enabled
// plugins run, defaults included. The deployments they ask must be served,
// and their questions prepared, before a request asks them.
func (d *Decision) MatchJudges() []MatchJudge {
	return MatchJudgesOf(d.Plugins)
}

// MatchJudgesOf are the judges of the match blocks a request's enabled
// plugins run, defaults included: a decision's, or the default plugins of a
// request that matches no decision.
func MatchJudgesOf(plugins []DecisionPlugin) []MatchJudge {
	var judges []MatchJudge
	if spec := decodePlugin(plugins, DecisionPluginTools, &ToolsPluginConfig{}).ActiveSelect(); spec != nil {
		judges = append(judges, *spec.EffectiveMatch().Judge)
	}
	if plugin := ContextPluginOf(plugins); plugin != nil {
		judges = append(judges, plugin.MatchJudges()...)
	}
	if plugin := decodePlugin(plugins, DecisionPluginResponseCache, &ResponseCachePluginConfig{}); plugin != nil && plugin.Enabled && plugin.Mode != ResponseCacheModeExact {
		if judge := MatchResponseCache.Effective(plugin.SemanticMatch()).Judge; judge != nil {
			judges = append(judges, *judge)
		}
	}
	return judges
}

// MatchJudges are the judges of a context plugin's sources and compress
// block, defaults included.
func (c *ContextPluginConfig) MatchJudges() []MatchJudge {
	var judges []MatchJudge
	for index := range c.Sources {
		source := &c.Sources[index]
		if judge := source.MatchFeature().Effective(source.Match).Judge; judge != nil {
			judges = append(judges, *judge)
		}
	}
	if c.Compress.CompressesRequest() {
		if judge := MatchCompress.Effective(c.Compress.Match).Judge; judge != nil {
			judges = append(judges, *judge)
		}
	}
	return judges
}
