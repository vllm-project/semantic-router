package evaluationplane

import (
	"fmt"
	"math"
	"testing"
)

func routerLearningPlannedTestRecords(seed int64) ([]executionRecordEvidence, map[string]struct{}) {
	rows := routerLearningTestRecords()
	planned := map[string]struct{}{routerLearningCaseOrder[0]: {}, routerLearningCaseOrder[1]: {}}
	for index := range rows {
		row := &rows[index]
		method := row.RouterLearning
		row.CaseID = routerLearningCaseOrder[method.RoundIndex]
		for trial, offset := range routerLearningSeedOffsets {
			if method.TrialID == fmt.Sprintf("trial-%02d", trial+1) {
				method.TrialSeed = (seed + offset) % (1 << 32)
			}
		}
	}
	return rows, planned
}

func TestRouterLearningPlanRejectsIncompleteAndForgedTrials(t *testing.T) {
	for _, name := range []string{
		"missing-tail", "missing-tail-every-trial", "missing-policy", "missing-trial",
		"duplicate-case", "duplicate-round", "reordered-cases", "duplicate-seeds",
		"wrong-seed", "wrong-trial-id", "extra-round",
	} {
		t.Run(name, func(t *testing.T) {
			rows, planned := routerLearningPlannedTestRecords(17)
			kept := make([]executionRecordEvidence, 0, len(rows))
			for _, row := range rows {
				method := row.RouterLearning
				switch name {
				case "missing-tail":
					if method.TrialID == "trial-01" && method.RoundIndex == 1 {
						continue
					}
				case "missing-tail-every-trial":
					if method.RoundIndex == 1 {
						continue
					}
				case "missing-policy":
					if method.PolicyID == "static-base" {
						continue
					}
				case "missing-trial":
					if method.TrialID == "trial-01" {
						continue
					}
				case "duplicate-case":
					row.CaseID = routerLearningCaseOrder[0]
				case "duplicate-round":
					method.RoundIndex = 0
				case "reordered-cases":
					row.CaseID = routerLearningCaseOrder[1-method.RoundIndex]
				case "duplicate-seeds":
					method.TrialSeed = 28
				case "wrong-seed":
					method.TrialSeed++
				case "wrong-trial-id":
					method.TrialID += "-renamed"
				case "extra-round":
					if method.RoundIndex == 1 {
						extra := row
						extraMethod := *method
						extraMethod.RoundIndex = 2
						extra.RouterLearning = &extraMethod
						kept = append(kept, extra)
					}
				}
				kept = append(kept, row)
			}
			if err := validateRouterLearningRunPlan(kept, planned, 17); err == nil {
				t.Fatal("invalid replay plan accepted")
			}
		})
	}
}

func TestRouterLearningPlanAcceptsSampledCasesAndWrappedSeeds(t *testing.T) {
	for _, seed := range []int64{17, 1<<32 - 5} {
		rows, planned := routerLearningPlannedTestRecords(seed)
		if err := validateRouterLearningRunPlan(rows, planned, seed); err != nil {
			t.Fatal(err)
		}
	}
}

func TestRouterLearningReducerRejectsTruncationAndRepeatedSeeds(t *testing.T) {
	for _, name := range []string{"missing-tail", "duplicate-seeds", "duplicate-case"} {
		rows := routerLearningTestRecords()
		kept := rows[:0]
		for _, row := range rows {
			if name == "missing-tail" && row.RouterLearning.TrialID == "trial-01" && row.RouterLearning.RoundIndex == 1 {
				continue
			}
			if name == "duplicate-seeds" {
				row.RouterLearning.TrialSeed = 11
			}
			if name == "duplicate-case" {
				row.CaseID = "same-case"
			}
			kept = append(kept, row)
		}
		if _, err := reduceRouterLearningMethod(kept); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
}

func TestRouterLearningStudentTInterval(t *testing.T) {
	values := make([]float64, routerLearningTrialCount)
	for i := range values {
		values[i] = float64(i)
	}
	interval := routerLearningInterval(values, false)
	// Mean 15.5, unbiased sample variance 88, t(0.975, 31)=2.0395134464.
	margin := 2.0395134463964077 * math.Sqrt(88.0/32)
	if math.Abs(interval[0]-(15.5-margin)) > 1e-12 || math.Abs(interval[1]-(15.5+margin)) > 1e-12 {
		t.Fatalf("unexpected Student t interval: %v", interval)
	}
}
