package extproc

import "sort"

const (
	routerLearningExperienceStatusUsed    routerLearningExperienceStatus = "used"
	routerLearningExperienceStatusMissing routerLearningExperienceStatus = "missing"

	routerLearningExperienceSourceLookupTable routerLearningExperienceSource = "internal_lookup_table"
)

type (
	routerLearningExperienceStatus string
	routerLearningExperienceSource string
)

type routerLearningExperienceSnapshot struct {
	views map[string]routerLearningExperienceView
}

type routerLearningExperienceDiagnostics struct {
	views []routerLearningExperienceView
}

type routerLearningExperienceView struct {
	name        string
	method      routerLearningMethod
	status      routerLearningExperienceStatus
	source      routerLearningExperienceSource
	version     string
	freshness   string
	sampleCount int
}

func (r *OpenAIRouter) routerLearningExperienceSnapshot() routerLearningExperienceSnapshot {
	status := routerLearningExperienceStatusMissing
	var source routerLearningExperienceSource
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

func (s routerLearningExperienceSnapshot) diagnostics(method routerLearningMethod) routerLearningExperienceDiagnostics {
	result := routerLearningExperienceDiagnostics{}
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
		result.views = append(result.views, s.views[key])
	}
	return result
}

func (d routerLearningExperienceDiagnostics) Empty() bool {
	return len(d.views) == 0
}

func (d routerLearningExperienceDiagnostics) appendLearningPolicyFields(fields *routerLearningPolicyFields) {
	if len(d.views) == 0 {
		return
	}
	experience := make(map[string]interface{}, len(d.views))
	for _, view := range d.views {
		if view.name == "" {
			continue
		}
		experience[view.name] = view.diagnostics()
	}
	if len(experience) > 0 {
		fields.Set(learningPolicyFieldExperience, experience)
	}
}

func (v routerLearningExperienceView) diagnostics() map[string]interface{} {
	status := v.status
	if status == "" {
		status = routerLearningExperienceStatusMissing
	}
	result := map[string]interface{}{
		"status": string(status),
	}
	if v.source != "" {
		result["source"] = string(v.source)
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
	result.policy.Experience = snapshot.diagnostics(result.method)
	return result
}
