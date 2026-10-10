package testcases

import (
	"context"
	"fmt"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("decision-runtime-routing", pkgtestcases.TestCase{
		Description: "Decision signals on a managed and an attached runtime, and the decision selector, follow the runtimes' own answers",
		Tags:        []string{"decision", "model-runtime", "routing", "selection"},
		Fn:          testDecisionRuntimeRouting,
	})
}

// The questions the Router builds from e2e/profiles/model-runtime/values.yaml.
var (
	decisionRuntimeManagedQuestions = map[string]modelruntime.Question{
		"answered": {Type: "noul", Instructions: "Is this request about software?"},
		"request_kind": {
			Type:         "choice",
			Instructions: "What kind of work does this request ask for?",
			Choices: []modelruntime.Choice{
				{Key: "code", Description: "Writing, reviewing or debugging code"},
				{Key: "math", Description: "Mathematics or quantitative reasoning"},
				{Key: "chat", Description: "Anything else"},
			},
		},
	}
	decisionRuntimeAttachedQuestions = map[string]modelruntime.Question{
		"attached_answered": {Type: "noul", Instructions: "Does this request ask a question?"},
	}
	decisionRuntimeSelectorQuestion = map[string]modelruntime.Question{
		"selector": {
			Type:         "choice",
			Instructions: "Which model should answer this request?",
			Choices: []modelruntime.Choice{
				{Key: "general-expert", Description: "General assistant for everyday requests"},
				{Key: "math-expert", Description: "Specialist for mathematics and quantitative reasoning"},
			},
		},
	}
	decisionRuntimePrompts = []string{
		"Write a Go function that merges two sorted slices.",
		"What is the integral of x squared from zero to three?",
		"Suggest a name for a friendly golden retriever.",
	}
)

// testDecisionRuntimeRouting compares the Router with the runtimes: the fixture
// weights are random, so the expected answers are what each runtime returns
// for the identical questions, and the Router must route on exactly those.
func testDecisionRuntimeRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := openModelRuntimeSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	if err = session.waitReady(ctx, mrDecisionDeployment, mrAttachedDecisions); err != nil {
		return err
	}
	managed, _, err := session.managed(ctx, mrDecisionDeployment)
	if err != nil {
		return err
	}
	details := map[string]interface{}{}
	for _, prompt := range decisionRuntimePrompts {
		observed, err := checkDecisionRuntimePrompt(ctx, session, managed, prompt)
		if err != nil {
			return fmt.Errorf("%q: %w", prompt, err)
		}
		details[prompt] = observed
	}
	if opts.SetDetails != nil {
		details["runtimes"] = "random-weight Decision 2.0 fixtures: one managed, one attached"
		opts.SetDetails(details)
	}
	return nil
}

func checkDecisionRuntimePrompt(
	ctx context.Context,
	session *modelRuntimeSession,
	managed *modelruntime.Client,
	prompt string,
) (string, error) {
	response, err := session.chat(ctx, prompt)
	if err != nil {
		return "", err
	}
	if decision := response.Headers.Get("x-vsr-selected-decision"); decision != "selector_route" {
		return "", fmt.Errorf("selected decision %q, want selector_route", decision)
	}
	signals, err := managed.Decide(ctx, modelruntime.DecisionsRequest{Model: mrDecisionDeployment, State: prompt, Questions: decisionRuntimeManagedQuestions})
	if err != nil {
		return "", err
	}
	if _, err = session.attachedRuntime().Decide(ctx, modelruntime.DecisionsRequest{Model: "decision-a", State: prompt, Questions: decisionRuntimeAttachedQuestions}); err != nil {
		return "", err
	}
	kind := signals.Answers["request_kind"].Choice
	matched := headerItems(response.Headers, "x-vsr-matched-decision-model")
	for _, want := range []string{"answered", "attached_answered", "request_kind:" + kind} {
		if !matched[want] {
			return "", fmt.Errorf("x-vsr-matched-decision-model %v lacks %q, the runtimes' own answer", matched, want)
		}
	}
	if matched["offline"] {
		return "", fmt.Errorf("the offline deployment's signal matched")
	}
	selector, err := managed.Decide(ctx, modelruntime.DecisionsRequest{Model: mrDecisionDeployment, State: prompt, Questions: decisionRuntimeSelectorQuestion})
	if err != nil {
		return "", err
	}
	want := selector.Answers["selector"].Choice
	if model := response.Headers.Get("x-vsr-selected-model"); model != want {
		return "", fmt.Errorf("selected model %q, the runtime chose %q (probabilities %v)", model, want, selector.Answers["selector"].Probabilities)
	}
	return fmt.Sprintf("request_kind=%s model=%s", kind, want), nil
}
