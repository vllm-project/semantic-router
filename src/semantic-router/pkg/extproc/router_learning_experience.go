package extproc

import "sort"

const (
	routerLearningExperienceStatusUsed    = "used"
	routerLearningExperienceStatusMissing = "missing"

	routerLearningExperienceSourceLookupTable = "internal_lookup_table"
)

type routerLearningExperienceSnapshot struct {
	views map[string]routerLearningExperienceView
}

type routerLearningExperienceView struct {
	name        string
	method      routerLearningMethod
	status      string
	source      string
	version     string
	freshness   string
	sampleCount int
}

func (r *OpenAIRouter) routerLearningExperienceSnapshot() routerLearningExperienceSnapshot {
	status := routerLearningExperienceStatusMissing
	source := ""
	if r != nil && r.LookupTable != nil {
		status = routerLearningExperienceStatusUsed
		source = routerLearningExperienceSourceLookupTable
	}
	return newRouterLearningExperienceSnapshot([]routerLearningExperienceView{
		{
			name:   "handoff_penalty",
			method: routerLearningMethodSessionAware,
			status: status,
			source: source,
		},
		{
			name:   "quality_gap",
			method: routerLearningMethodSessionAware,
			status: status,
			source: source,
		},
		{
			name:   "remaining_turn_estimate",
			method: routerLearningMethodSessionAware,
			status: status,
			source: source,
		},
		{
			name:   "reward_stats",
			method: routerLearningMethodBandit,
			status: routerLearningExperienceStatusMissing,
		},
		{
			name:   "elo_rating",
			method: routerLearningMethodElo,
			status: routerLearningExperienceStatusMissing,
		},
		{
			name:   "interaction_graph",
			method: routerLearningMethodPersonalization,
			status: routerLearningExperienceStatusMissing,
		},
	})
}

func newRouterLearningExperienceSnapshot(views []routerLearningExperienceView) routerLearningExperienceSnapshot {
	snapshot := routerLearningExperienceSnapshot{views: map[string]routerLearningExperienceView{}}
	for _, view := range views {
		if view.method == "" || view.name == "" {
			continue
		}
		snapshot.views[string(view.method)+"."+view.name] = view
	}
	return snapshot
}

func (s routerLearningExperienceSnapshot) diagnostics(method routerLearningMethod) map[string]interface{} {
	result := map[string]interface{}{}
	if method == "" {
		return result
	}
	keys := make([]string, 0, len(s.views))
	for key, view := range s.views {
		if view.method == method {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	for _, key := range keys {
		view := s.views[key]
		result[view.name] = view.diagnostics()
	}
	return result
}

func (v routerLearningExperienceView) diagnostics() map[string]interface{} {
	status := v.status
	if status == "" {
		status = routerLearningExperienceStatusMissing
	}
	result := map[string]interface{}{
		"status": status,
	}
	if v.source != "" {
		result["source"] = v.source
	}
	if v.version != "" {
		result["version"] = v.version
	}
	if v.freshness != "" {
		result["freshness"] = v.freshness
	}
	if v.sampleCount > 0 {
		result["sample_count"] = v.sampleCount
	}
	return result
}

func attachRouterLearningExperience(
	result routerLearningAdaptationResult,
	snapshot routerLearningExperienceSnapshot,
) routerLearningAdaptationResult {
	if result.method == "" {
		return result
	}
	experience := snapshot.diagnostics(result.method)
	if len(experience) > 0 {
		result.policy.Set("experience", experience)
	}
	return result
}
