package config

// MatchJudges are the judges of the match blocks a decision's enabled
// plugins run, defaults included. The deployments they ask must be served,
// and their questions prepared, before a request asks them.
func (d *Decision) MatchJudges() []MatchJudge {
	var judges []MatchJudge
	if spec := d.GetToolsConfig().ActiveSelect(); spec != nil {
		judges = append(judges, *spec.EffectiveMatch().Judge)
	}
	if plugin := d.GetContextConfig(); plugin != nil && plugin.Enabled {
		judges = append(judges, plugin.MatchJudges()...)
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
	if c.Compress != nil {
		if judge := MatchCompress.Effective(c.Compress.Match).Judge; judge != nil {
			judges = append(judges, *judge)
		}
	}
	return judges
}
