package testcases

import (
	"context"
	"fmt"
	"math"
	"slices"
	"strings"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("decision-runtime-set-span", pkgtestcases.TestCase{
		Description: "Set and span decision signals on an attached Vela 2.0 runtime publish and match exactly the runtime's own answers",
		Tags:        []string{"decision", "model-runtime", "routing", "vela2"},
		Fn:          testDecisionRuntimeSetSpan,
	})
}

// The questions the Router builds from e2e/profiles/model-runtime/values.yaml
// for attached-vela2; its labels are sorted there, as Criteria encode them.
var (
	zeroThreshold                = 0.0
	decisionRuntimeSetSpanPrompt = "Hi, I am Anna from Lisbon: please refund the duplicate charge and resend my parcel."
	decisionRuntimeVela2Question = map[string]modelruntime.Question{
		"topics": {
			Type: "set", Instructions: "Which topics does the request mention?", Threshold: &zeroThreshold,
			Criteria: map[string]string{"billing": "Payments, invoices or refunds", "shipping": "Deliveries, parcels or returns"},
		},
		"names": {
			Type: "span", Instructions: "Which spans name a city or a person?", Threshold: &zeroThreshold,
			Criteria: map[string]string{"city": "A city name", "person": "A person's name"},
		},
	}
)

// testDecisionRuntimeSetSpan compares the Router with the attached Vela 2.0
// fixture: the weights are random, so the expected answers are what the
// runtime returns for the identical questions about the identical text.
func testDecisionRuntimeSetSpan(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := openModelRuntimeSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	if err = session.waitReady(ctx, mrAttachedVela2); err != nil {
		return err
	}
	details := map[string]interface{}{}
	for _, prompt := range append([]string{decisionRuntimeSetSpanPrompt}, decisionRuntimePrompts...) {
		observed, err := checkDecisionRuntimeSetSpan(ctx, session, prompt)
		if err != nil {
			return fmt.Errorf("%q: %w", prompt, err)
		}
		details[prompt] = observed
	}
	if opts.SetDetails != nil {
		details["runtime"] = "random-weight Vela 2.0 encoder fixture on the attached runtime"
		opts.SetDetails(details)
	}
	return nil
}

func checkDecisionRuntimeSetSpan(ctx context.Context, session *modelRuntimeSession, prompt string) (string, error) {
	preview, err := session.preview(ctx, prompt)
	if err != nil {
		return "", err
	}
	if err = onlyOfflineSignalError(preview.SignalErrors); err != nil {
		return "", err
	}
	want, err := session.attachedRuntime().Decide(ctx, modelruntime.DecisionsRequest{Model: "vela2-a", State: prompt, Questions: decisionRuntimeVela2Question})
	if err != nil {
		return "", err
	}
	set, ok := want.Sets["topics"]
	if !ok {
		return "", fmt.Errorf("the runtime gave no set answer: %+v", want)
	}
	expected := map[string]bool{}
	for label, probability := range set.Probabilities {
		if got := preview.SignalValues["systemone:topics:"+label]; math.Abs(got-probability) > 1e-9 {
			return "", fmt.Errorf("set label %s is %v, the runtime answered %v", label, got, probability)
		}
	}
	for _, label := range set.Selected {
		expected["topics:"+label] = true
	}
	best := map[string]float64{}
	for _, span := range want.Spans["names"] {
		best[span.Label] = math.Max(best[span.Label], span.Probability)
		expected["names:"+span.Label] = true
	}
	highest := 0.0
	for _, label := range []string{"city", "person"} {
		if got := preview.SignalValues["systemone:names:"+label]; math.Abs(got-best[label]) > 1e-9 {
			return "", fmt.Errorf("span label %s is %v, the runtime's most probable span %v", label, got, best[label])
		}
		highest = math.Max(highest, best[label])
	}
	if got := preview.SignalValues["systemone:names"]; math.Abs(got-highest) > 1e-9 {
		return "", fmt.Errorf("span rule value %v, the runtime's most probable span of a label %v", got, highest)
	}
	response, err := session.chat(ctx, prompt)
	if err != nil {
		return "", err
	}
	// The preview lists the label references label conditions read; the
	// header lists a set or span rule by name.
	var previewMatches []string
	if preview.DecisionResult != nil {
		previewMatches = preview.DecisionResult.MatchedSignals.SystemOne
	}
	ruleMatched := map[string]bool{}
	for _, label := range []string{"topics:billing", "topics:shipping", "names:city", "names:person"} {
		if got := slices.Contains(previewMatches, label); got != expected[label] {
			return "", fmt.Errorf("routing preview systemone matches %v: %s matched=%v, the runtime's answer says %v", previewMatches, label, got, expected[label])
		}
		rule, _, _ := strings.Cut(label, ":")
		ruleMatched[rule] = ruleMatched[rule] || expected[label]
	}
	matched := headerItems(response.Headers, "x-vsr-matched-decision-model")
	for rule, want := range ruleMatched {
		if matched[rule] != want {
			return "", fmt.Errorf("x-vsr-matched-decision-model %v: %s matched=%v, the runtime's answer says %v", matched, rule, matched[rule], want)
		}
	}
	for item := range matched {
		if strings.HasPrefix(item, "topics:") || strings.HasPrefix(item, "names:") {
			return "", fmt.Errorf("x-vsr-matched-decision-model lists the label reference %s of a rule it names", item)
		}
	}
	var labels []string
	for label := range expected {
		labels = append(labels, label)
	}
	return strings.Join(labels, ","), nil
}
