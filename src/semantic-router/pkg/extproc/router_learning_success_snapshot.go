package extproc

import (
	"strings"
	"time"
)

type routerLearningEvidenceSnapshot struct {
	takenAt  time.Time
	identity string
	decision string
	tier     int
	cohort   string
	byKey    map[string]routerLearningModelExperience
}

func (rt *routerLearningRuntime) freezeEvidenceSnapshot(
	decision string,
	tier int,
	models []string,
	cohort string,
) routerLearningEvidenceSnapshot {
	snap := routerLearningEvidenceSnapshot{
		takenAt:  time.Now().UTC(),
		decision: strings.TrimSpace(decision),
		tier:     tier,
		cohort:   strings.TrimSpace(cohort),
		byKey:    map[string]routerLearningModelExperience{},
	}
	if rt != nil && rt.shared != nil {
		rt.shared.mu.Lock()
		for _, model := range models {
			model = strings.TrimSpace(model)
			if model == "" {
				continue
			}
			for _, key := range snapshotExperienceKeys(snap.decision, snap.tier, snap.cohort, model) {
				if exp := rt.shared.experience[key]; exp != nil {
					snap.byKey[key] = *exp
				}
			}
		}
		rt.shared.mu.Unlock()
	}
	snap.identity = snapshotIdentity(snap.takenAt)
	return snap
}

func snapshotExperienceKeys(decision string, tier int, cohort string, model string) []string {
	keys := make([]string, 0, 4)
	if cohort != "" {
		keys = append(keys, modelExperienceKey(cohort, tier, model))
	}
	if decision != "" {
		keys = append(keys, modelExperienceKey(decision, tier, model))
	}
	if tier != 0 {
		keys = append(keys, modelExperienceKey("", tier, model))
	}
	keys = append(keys, modelExperienceKey("", 0, model))
	return keys
}

func snapshotIdentity(takenAt time.Time) string {
	if takenAt.IsZero() {
		takenAt = time.Now().UTC()
	}
	return takenAt.UTC().Format(time.RFC3339Nano)
}
