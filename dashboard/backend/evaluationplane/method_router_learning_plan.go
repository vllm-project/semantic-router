package evaluationplane

import (
	"fmt"
	"slices"
	"sort"
)

// Frozen router-learning-core-v2 coordinates. The Python replay parity test
// binds its packaged corpus and manifest to this independent server contract.
var routerLearningSeedOffsets = [routerLearningTrialCount]int64{
	11, 29, 47, 71, 101, 131, 173, 211,
	251, 293, 337, 383, 431, 479, 523, 571,
	619, 661, 709, 757, 809, 853, 907, 953,
	1009, 1061, 1109, 1153, 1201, 1259, 1307, 1361,
}

var routerLearningCaseOrder = []string{
	"learning-format-01", "learning-knowledge-02", "learning-context-03",
	"learning-reasoning-04", "learning-format-05", "learning-code-06",
	"learning-knowledge-07", "learning-tool-08", "learning-safety-09",
	"learning-reasoning-10", "learning-format-11", "learning-code-12",
}

func validateRouterLearningTrialSets(policies map[string]map[string]*routerLearningTrialRows) error {
	for policy, trials := range policies {
		seeds := make(map[int64]string)
		var reference []string
		for trialID, trial := range trials {
			if prior, duplicate := seeds[trial.seed]; duplicate {
				return fmt.Errorf("router learning %s/%s repeats seed from %s", policy, trialID, prior)
			}
			seeds[trial.seed] = trialID
			sort.Slice(trial.rows, func(i, j int) bool {
				return trial.rows[i].RouterLearning.RoundIndex < trial.rows[j].RouterLearning.RoundIndex
			})
			caseIDs := make([]string, len(trial.rows))
			seen := make(map[string]bool)
			for index, row := range trial.rows {
				if seen[row.CaseID] {
					return fmt.Errorf("router learning %s/%s repeats case %s", policy, trialID, row.CaseID)
				}
				seen[row.CaseID] = true
				caseIDs[index] = row.CaseID
			}
			if reference == nil {
				reference = caseIDs
			} else if !slices.Equal(reference, caseIDs) {
				return fmt.Errorf("router learning %s/%s has an incomplete or reordered case sequence", policy, trialID)
			}
		}
	}
	return nil
}

func validateRouterLearningRunPlan(records []executionRecordEvidence, planned map[string]struct{}, seed int64) error {
	ordered := make([]string, 0, len(planned))
	for _, caseID := range routerLearningCaseOrder {
		if _, selected := planned[caseID]; selected {
			ordered = append(ordered, caseID)
		}
	}
	if len(ordered) == 0 || len(ordered) != len(planned) {
		return fmt.Errorf("router learning plan contains no cases or an unknown corpus case")
	}
	byPolicy := make(map[string]map[string]map[int64]executionRecordEvidence)
	for _, record := range records {
		method := record.RouterLearning
		if method == nil || !slices.Contains(routerLearningPolicyIDs, method.PolicyID) {
			return fmt.Errorf("router learning plan requires a supported policy on every row")
		}
		if byPolicy[method.PolicyID] == nil {
			byPolicy[method.PolicyID] = make(map[string]map[int64]executionRecordEvidence)
		}
		if byPolicy[method.PolicyID][method.TrialID] == nil {
			byPolicy[method.PolicyID][method.TrialID] = make(map[int64]executionRecordEvidence)
		}
		rounds := byPolicy[method.PolicyID][method.TrialID]
		if _, duplicate := rounds[method.RoundIndex]; duplicate {
			return fmt.Errorf("router learning %s/%s repeats round %d", method.PolicyID, method.TrialID, method.RoundIndex)
		}
		rounds[method.RoundIndex] = record
	}
	for _, policy := range routerLearningPolicyIDs {
		trials := byPolicy[policy]
		for index, offset := range routerLearningSeedOffsets {
			trialID := fmt.Sprintf("trial-%02d", index+1)
			rounds := trials[trialID]
			expectedSeed := (seed + offset) % (1 << 32)
			for round, caseID := range ordered {
				row, present := rounds[int64(round)]
				if !present || row.CaseID != caseID {
					return fmt.Errorf("router learning %s/%s requires case %s at round %d; rerun the complete trial", policy, trialID, caseID, round)
				}
				if row.RouterLearning.TrialSeed != expectedSeed {
					return fmt.Errorf("router learning %s/%s seed does not match run plan: want %d", policy, trialID, expectedSeed)
				}
			}
			if len(rounds) != len(ordered) {
				return fmt.Errorf("router learning %s/%s contains unplanned rounds", policy, trialID)
			}
		}
		if len(trials) != routerLearningTrialCount {
			return fmt.Errorf("router learning %s contains unplanned trials", policy)
		}
	}
	return nil
}
