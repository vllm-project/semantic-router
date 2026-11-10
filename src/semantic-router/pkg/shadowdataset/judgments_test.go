package shadowdataset

import (
	"strings"
	"testing"
)

func judgedTasks(t *testing.T) JudgeTaskSet {
	t.Helper()
	manifest, texts := judgedManifest(t)
	return buildTasksOrFail(t, manifest, texts)
}

func judgmentSet(tasks JudgeTaskSet, judgments ...Judgment) JudgmentSet {
	return JudgmentSet{
		Version:        JudgmentsVersion,
		ManifestDigest: tasks.ManifestDigest,
		Judge:          JudgeIdentity{Model: "judge-model", ModelRevision: "rev-1", RubricVersion: "rubric-1"},
		Judgments:      judgments,
	}
}

// ordersOf returns the two tasks of one pair.
func ordersOf(tasks JudgeTaskSet, pair string) (JudgeTask, JudgeTask) {
	var found []JudgeTask
	for _, task := range tasks.Tasks {
		if task.Pair == pair {
			found = append(found, task)
		}
	}
	return found[0], found[1]
}

func pick(task JudgeTask, arm string) Judgment {
	return Judgment{Task: task.ID, Outcome: JudgmentPreferred, Preferred: arm}
}

func validOrFail(t *testing.T, tasks JudgeTaskSet, set JudgmentSet) {
	t.Helper()
	if err := ValidateJudgments(tasks, set); err != nil {
		t.Fatalf("ValidateJudgments: %v", err)
	}
}

// A judge that picks the same answer in both orders is consistent. One that
// picks whatever sat first is following the slot, which is the position bias
// the swapped orders exist to expose.
func TestReportJudgmentsSeparatesConsistentPairsFromSlotFollowing(t *testing.T) {
	tasks := judgedTasks(t)
	one, swapped := ordersOf(tasks, tasks.Tasks[0].Pair)
	var otherPair string
	for _, task := range tasks.Tasks {
		if task.Pair != one.Pair {
			otherPair = task.Pair
		}
	}
	two, twoSwapped := ordersOf(tasks, otherPair)

	set := judgmentSet(tasks,
		pick(one, one.First.Arm), pick(swapped, one.First.Arm),
		pick(two, two.First.Arm), pick(twoSwapped, twoSwapped.First.Arm),
	)
	validOrFail(t, tasks, set)
	report := ReportJudgments(tasks, set)

	if report.ConsistentPairs != 1 || report.SlotFollowingPairs != 1 || report.InconsistentPairs != 0 {
		t.Fatalf("report %+v, want one consistent and one slot-following pair", report)
	}
	if report.FirstSlotPicks != 3 || report.SecondSlotPicks != 1 {
		t.Fatalf("slot picks %d first and %d second, want 3 and 1", report.FirstSlotPicks, report.SecondSlotPicks)
	}
}

// A pair with an abstention or a malformed answer is not a position finding in
// either direction. It stays in the outcome counts as evidence about the judge.
func TestReportJudgmentsKeepsAbstentionsAsEvidence(t *testing.T) {
	tasks := judgedTasks(t)
	one, swapped := ordersOf(tasks, tasks.Tasks[0].Pair)

	set := judgmentSet(tasks,
		pick(one, one.Second.Arm),
		Judgment{Task: swapped.ID, Outcome: JudgmentMalformed, MalformedReason: "no verdict token"},
	)
	validOrFail(t, tasks, set)
	report := ReportJudgments(tasks, set)

	if report.IncompletePairs != 2 || report.ConsistentPairs+report.SlotFollowingPairs+report.InconsistentPairs != 0 {
		t.Fatalf("report %+v, want both pairs incomplete", report)
	}
	if report.Outcomes[JudgmentMalformed] != 1 || report.Outcomes[JudgmentPreferred] != 1 {
		t.Fatalf("outcomes %v, want the malformed answer counted beside the pick", report.Outcomes)
	}
}

func TestReportJudgmentsCountsTiesAndMixedVerdicts(t *testing.T) {
	tasks := judgedTasks(t)
	one, swapped := ordersOf(tasks, tasks.Tasks[0].Pair)
	var otherPair string
	for _, task := range tasks.Tasks {
		if task.Pair != one.Pair {
			otherPair = task.Pair
		}
	}
	two, twoSwapped := ordersOf(tasks, otherPair)

	set := judgmentSet(tasks,
		Judgment{Task: one.ID, Outcome: JudgmentTie}, Judgment{Task: swapped.ID, Outcome: JudgmentTie},
		pick(two, two.First.Arm), Judgment{Task: twoSwapped.ID, Outcome: JudgmentTie},
	)
	validOrFail(t, tasks, set)
	report := ReportJudgments(tasks, set)

	if report.ConsistentPairs != 1 || report.InconsistentPairs != 1 {
		t.Fatalf("report %+v, want two ties consistent and a pick against a tie inconsistent", report)
	}
}

func TestReportJudgmentsSumsBudgetAndCountsLeakedTasks(t *testing.T) {
	tasks := judgedTasks(t)
	tasks.Tasks[0].Second.NamesOwnModel = true
	judgment := pick(tasks.Tasks[0], tasks.Tasks[0].First.Arm)
	judgment.InputTokens, judgment.OutputTokens, judgment.CostUSD = 100, 5, 0.25
	other := Judgment{Task: tasks.Tasks[1].ID, Outcome: JudgmentAbstain, InputTokens: 90, CostUSD: 0.5}

	report := ReportJudgments(tasks, judgmentSet(tasks, judgment, other))
	if report.InputTokens != 190 || report.OutputTokens != 5 || report.CostUSD != 0.75 {
		t.Fatalf("budget %d in, %d out, %v cost, want 190, 5 and 0.75", report.InputTokens, report.OutputTokens, report.CostUSD)
	}
	if report.IdentityLeakedTasks != 1 {
		t.Fatalf("counted %d leaked tasks, want the one whose answer names its model", report.IdentityLeakedTasks)
	}
}

// Everything here fails before a judgment set is published. A verdict that
// cannot be read against its task would otherwise become evidence for a
// comparison it never made.
func TestValidateJudgmentsRejectsASetItCannotReadAgainstItsTasks(t *testing.T) {
	tasks := judgedTasks(t)
	task := tasks.Tasks[0]
	outside := 1.5

	for name, mutate := range map[string]func(*JudgmentSet){
		"wrong version":            func(s *JudgmentSet) { s.Version = "shadow-judgments.v0" },
		"other manifest":           func(s *JudgmentSet) { s.ManifestDigest = "another" },
		"unnamed rubric":           func(s *JudgmentSet) { s.Judge.RubricVersion = "" },
		"unknown task":             func(s *JudgmentSet) { s.Judgments[0].Task = "nobody-asked" },
		"task judged twice":        func(s *JudgmentSet) { s.Judgments = append(s.Judgments, s.Judgments[0]) },
		"arm the task never shown": func(s *JudgmentSet) { s.Judgments[0].Preferred = tasks.Tasks[1].First.Arm },
		"tie naming an arm":        func(s *JudgmentSet) { s.Judgments[0].Outcome = JudgmentTie },
		"malformed with no reason": func(s *JudgmentSet) { s.Judgments[0] = Judgment{Task: task.ID, Outcome: JudgmentMalformed} },
		"reason on a clean pick":   func(s *JudgmentSet) { s.Judgments[0].MalformedReason = "none" },
		"unknown outcome":          func(s *JudgmentSet) { s.Judgments[0].Outcome = "win" },
		"confidence above one":     func(s *JudgmentSet) { s.Judgments[0].Confidence = &outside },
		"negative cost":            func(s *JudgmentSet) { s.Judgments[0].CostUSD = -1 },
	} {
		t.Run(name, func(t *testing.T) {
			set := judgmentSet(tasks, pick(task, task.First.Arm))
			mutate(&set)
			if err := ValidateJudgments(tasks, set); err == nil {
				t.Fatal("ValidateJudgments accepted the set")
			}
		})
	}
}

// A judgment record has no field for reasoning, and decoding refuses one, so
// hidden reasoning cannot be carried into a published set by accident.
func TestDecodeJudgmentsRefusesAReasoningField(t *testing.T) {
	clean := `{"version":"shadow-judgments.v1","manifest_digest":"m","judge":{"model":"j","model_revision":"r","rubric_version":"v"},"judgments":[{"task":"t","outcome":"tie"}]}`
	if _, err := DecodeJudgments(strings.NewReader(clean)); err != nil {
		t.Fatalf("DecodeJudgments rejected a clean set: %v", err)
	}
	withReasoning := strings.Replace(clean, `"outcome":"tie"`, `"outcome":"tie","reasoning":"first is longer"`, 1)
	if _, err := DecodeJudgments(strings.NewReader(withReasoning)); err == nil {
		t.Fatal("DecodeJudgments accepted a judgment carrying reasoning")
	}
}
