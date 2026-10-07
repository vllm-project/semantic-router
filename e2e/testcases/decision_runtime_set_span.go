package testcases

import (
	"context"
	"fmt"
	"math"
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
		if got := preview.SignalValues["decision:topics:"+label]; math.Abs(got-probability) > 1e-9 {
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
	for _, label := range []string{"city", "person"} {
		if got := preview.SignalValues["decision:names:"+label]; math.Abs(got-best[label]) > 1e-9 {
			return "", fmt.Errorf("span label %s is %v, the runtime's most probable span %v", label, got, best[label])
		}
	}
	if noul := want.Answers["names"].Noul; noul == nil || math.Abs(preview.SignalValues["decision:names"]-*noul) > 1e-9 {
		return "", fmt.Errorf("span question value %v, the runtime answered %v", preview.SignalValues["decision:names"], noul)
	}
	response, err := session.chat(ctx, prompt)
	if err != nil {
		return "", err
	}
	matched := headerItems(response.Headers, "x-vsr-matched-decision-model")
	for _, label := range []string{"topics:billing", "topics:shipping", "names:city", "names:person"} {
		if matched[label] != expected[label] {
			return "", fmt.Errorf("x-vsr-matched-decision-model %v: %s matched=%v, the runtime's answer says %v", matched, label, matched[label], expected[label])
		}
	}
	var labels []string
	for label := range expected {
		labels = append(labels, label)
	}
	return strings.Join(labels, ","), nil
}
