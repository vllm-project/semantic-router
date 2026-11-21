package classification

import (
	"context"
	"crypto/sha256"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// evaluateTurnAfter evaluates the request stage of a turn whose earlier user
// turns were history.
func evaluateTurnAfter(classifier *Classifier, history []string, text string) *SignalResults {
	return classifier.evaluateAllSignalsWithContext(SignalEvaluationInput{
		Text: text, CurrentUserText: text, PriorUserMessages: history, RequestFacts: RequestFacts{Context: context.Background()},
	}, classifier.Config.Decisions, true)
}

// A tool loop sends the user turn it answers again on every request, so the
// guard reads that turn's windows on the first request and takes their
// answers on the next ones: the second request reads no window, and the
// guard still reports the turn read in full.
func TestAToolLoopReadsAnUnchangedLongTurnOnce(t *testing.T) {
	classifier, fake := refusingGuardClassifier(t, 20*time.Second, "", time.Millisecond)
	text := scanProse(30_000) + "\n\nSummarize the minutes above."

	first := evaluateTurn(classifier, text)
	read := len(guardStates(fake))
	if read < 10 || guarded(first) || first.ScanCoverage["jailbreak:prompt_attack"] != scanCoverageFull {
		t.Fatalf("first request: %d windows read, guarded %t, coverage %v", read, guarded(first), first.ScanCoverage)
	}
	second := evaluateTurn(classifier, text)
	if again := len(guardStates(fake)) - read; again != 0 {
		t.Fatalf("the second request read %d of the turn's windows again", again)
	}
	if guarded(second) || second.ScanCoverage["jailbreak:prompt_attack"] != scanCoverageFull ||
		second.ScanCoverage["safety:unsafe_request"] != scanCoverageFull {
		t.Fatalf("second request: guarded %t, coverage %v", guarded(second), second.ScanCoverage)
	}
	if second.SignalValues["jailbreak:prompt_attack"] != first.SignalValues["jailbreak:prompt_attack"] {
		t.Fatalf("the kept answers changed the rule's value: %v, then %v",
			first.SignalValues["jailbreak:prompt_attack"], second.SignalValues["jailbreak:prompt_attack"])
	}
}

// A rule that reads the earlier turns reads a conversation that grows at its
// end, so a new turn reads only the windows past what the turn before read
// whole: the last window of the earlier text, which now ends elsewhere, and
// the new text.
func TestAGrowingConversationReadsOnlyItsNewWindows(t *testing.T) {
	classifier, fake := refusingGuardClassifier(t, 20*time.Second, "          input: {history: true}\n", time.Millisecond)
	first := scanProse(30_000)
	evaluateTurnAfter(classifier, nil, first)
	before := guardStates(fake)

	results := evaluateTurnAfter(classifier, []string{first}, "And the minutes of the week after?")
	again := guardStates(fake)[len(before):]
	if len(again) == 0 || len(again) > 2 || guarded(results) || results.ScanCoverage["jailbreak:prompt_attack"] != scanCoverageFull {
		t.Fatalf("the second turn read %d windows (the first read %d), guarded %t, coverage %v",
			len(again), len(before), guarded(results), results.ScanCoverage)
	}
	for _, state := range again {
		if slices.Contains(before, state) {
			t.Fatalf("the second turn read a window the first turn read: %.80q", state)
		}
	}
	if !strings.HasSuffix(again[len(again)-1], "And the minutes of the week after?") && !strings.HasSuffix(again[0], "And the minutes of the week after?") {
		t.Fatal("the second turn did not read its new text")
	}
}

// A window whose kept answer found an attack still refuses the request,
// whatever order the next scan reads its windows in.
func TestAKeptWindowThatFoundAnAttackStillRefuses(t *testing.T) {
	classifier, fake := refusingGuardClassifier(t, 20*time.Second, "", 2*time.Millisecond)
	prose := scanProse(40_000)
	at := strings.IndexByte(prose[len(prose)/2:], ' ') + len(prose)/2 + 1
	text := prose[:at] + scanInjection + " " + prose[at:]

	for turn := range 3 {
		results := evaluateTurn(classifier, text)
		if !slices.Contains(results.MatchedJailbreakRules, "prompt_attack") {
			t.Fatalf("turn %d: the attack was not refused: coverage %v, errors %v", turn+1, results.ScanCoverage, results.SignalErrors)
		}
	}
	if read := len(guardStates(fake)); read == 0 {
		t.Fatal("no window was read")
	}
}

// A window's key holds the text before it: the same window after other text
// is a different key, the same text up to the window's end is the same key
// whatever follows, and other questions or another model are other keys.
func TestAWindowKeyHoldsTheTextBeforeIt(t *testing.T) {
	classifier, _ := refusingGuardClassifier(t, 20*time.Second, "", time.Millisecond)
	prose := scanProse(20_000)
	scanOf := func(text string) *guardScan {
		state := modelservice.Request{State: text}
		scan := newGuardScan("primary", state, classifier.scanWindowBudget("primary"))
		scan.rules = []askedRule{jailbreakAsked(t, classifier)}
		scan.verdicts = classifier.newScanVerdictKeys(scan)
		if scan.verdicts == nil {
			t.Fatal("a request-stage scan keeps no answers")
		}
		return scan
	}
	base := scanOf(prose + "\n\nfirst ending")
	window := base.windows[len(base.windows)/2]
	key := *base.verdictKey(window)

	if other := scanOf(prose + "\n\nanother ending"); *other.verdictKey(window) != key {
		t.Fatal("the same text up to a window's end gave another key")
	}
	if shifted := scanOf("X" + prose[1:] + "\n\nfirst ending"); *shifted.verdictKey(window) == key {
		t.Fatal("other text before a window gave the same key")
	}
	otherModel := scanOf(prose + "\n\nfirst ending")
	otherModel.deployment = "secondary"
	otherModel.verdicts = classifier.newScanVerdictKeys(otherModel)
	if otherModel.verdicts != nil {
		t.Fatal("an undeclared deployment must not retain window verdicts")
	}
	classifier.Config.ModelDeployments["secondary"] = classifier.Config.ModelDeployments["primary"]
	otherModel.verdicts = classifier.newScanVerdictKeys(otherModel)
	if otherModel.verdicts != nil {
		t.Fatal("an attached deployment without observed precision must not retain window verdicts")
	}
	// Observe exactly the same model under the second declaration.
	card, _ := classifier.models.runtime.CurrentDeploymentCard("primary")
	second := &Classifier{Config: classifier.Config, models: &classifierModelRuntime{runtime: serving.New(&cardServices{card: card}, nil)}}
	otherModel.verdicts = second.newScanVerdictKeys(otherModel)
	if otherModel.verdicts == nil || *otherModel.verdictKey(window) == key {
		t.Fatal("another observed deployment must have its own key")
	}
}

// Only an answer that proves the model read the whole window is kept, and
// the classifier keeps at most windowVerdictEntries windows, each until
// windowVerdictTTL after a request last read it.
func TestOnlyCompleteAnswersAreKeptAndTheyLeave(t *testing.T) {
	classifier, _ := refusingGuardClassifier(t, 20*time.Second, "", time.Millisecond)
	call := &systemOneCall{rules: []askedRule{jailbreakAsked(t, classifier)}}
	id := call.rules[0].questionID()
	complete := modelservice.Response{Answers: map[string]modelservice.Answer{id: {InputCoverage: "complete", Probabilities: map[string]float64{"jailbreak": 0.1}}}}
	for name, response := range map[string]modelservice.Response{
		"no answer":       {},
		"an error":        {Answers: map[string]modelservice.Answer{id: {Error: "timeout"}}},
		"coverage unread": {Answers: map[string]modelservice.Answer{id: {Probabilities: map[string]float64{"jailbreak": 0.1}}}},
	} {
		if keptAnswers(call, response) {
			t.Errorf("%s: kept", name)
		}
	}
	if !keptAnswers(call, complete) {
		t.Fatal("a complete answer is not kept")
	}

	var verdicts windowVerdicts
	keyOf := func(index int) windowVerdictKey { return sha256.Sum256([]byte(strconv.Itoa(index))) }
	for index := range windowVerdictEntries + 1 {
		verdicts.put(keyOf(index), complete)
	}
	if _, ok := verdicts.get(keyOf(0)); ok {
		t.Fatal("the least recently read window stayed past the bound")
	}
	last := keyOf(windowVerdictEntries)
	kept, ok := verdicts.get(last)
	if !ok || kept.Answers[id].Probabilities["jailbreak"] != 0.1 {
		t.Fatal("the newest window is not kept")
	}
	kept.Answers[id].Probabilities["jailbreak"] = 0.9
	if again, _ := verdicts.get(last); again.Answers[id].Probabilities["jailbreak"] != 0.1 {
		t.Fatal("a caller changed the kept answers")
	}
	verdicts.entries[last].Value.(*keptWindowEntry).expires = time.Now()
	if _, ok := verdicts.get(last); ok {
		t.Fatal("an expired window is still kept")
	}
}

// jailbreakAsked is the classifier's request-stage jailbreak rule.
func jailbreakAsked(t *testing.T, classifier *Classifier) askedRule {
	t.Helper()
	for _, rule := range classifier.applicationRules {
		if rule.Rule.Name == "prompt_attack" {
			return applicationAsked(rule)
		}
	}
	t.Fatal("no prompt_attack rule")
	return askedRule{}
}
