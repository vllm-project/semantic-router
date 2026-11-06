package config

// MatchJudges are the judges of the match blocks a decision's enabled
// plugins run, defaults included. The deployments they ask must be served,
// and their questions prepared, before a request asks them.
func (d *Decision) MatchJudges() []MatchJudge {
	var judges []MatchJudge
	if spec := d.GetToolsConfig().ActiveSelect(); spec != nil {
		judges = append(judges, *spec.EffectiveMatch().Judge)
	}
	return judges
}
