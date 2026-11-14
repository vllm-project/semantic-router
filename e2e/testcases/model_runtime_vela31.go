package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("model-runtime-vela31-described-noul", pkgtestcases.TestCase{
		Description: "A described Vela 3.1 Noul matches the deployed runtime's equivalent no/yes Choice, the Router's signal probability, and the gateway's selected decision and backend model",
		Tags:        []string{"model-runtime", "vela31", "decision", "routing"},
		Fn:          testModelRuntimeVela31DescribedNoul,
	})
}

const (
	vela31Signal       = "vela31_described"
	vela31Decision     = "vela31_described_route"
	vela31Model        = "math-expert"
	vela31Instructions = "Does this request need urgent action?"
	// Random weights establish request semantics, not classifier accuracy.
	vela31MinimumProbability = 0.0001
)

// These are the question in values.yaml and its identical two-option rendering.
var vela31Questions = map[string]modelruntime.Question{
	vela31Signal: {
		Type: "noul", Instructions: vela31Instructions,
		Criteria: map[string]string{"false": "can wait", "true": "needs immediate action"},
	},
	"equivalent": {
		Type: "choice", Instructions: vela31Instructions,
		Choices: []modelruntime.Choice{
			{Key: "no", Description: "can wait"},
			{Key: "yes", Description: "needs immediate action"},
		},
	},
}

func testModelRuntimeVela31DescribedNoul(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := openRouterRuntimeSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	if err = session.waitReady(ctx, mrVela31Deployment); err != nil {
		return err
	}
	runtime, _, err := session.managed(ctx, mrVela31Deployment)
	if err != nil {
		return err
	}
	prompt := freshPrompt("[vela31-described-noul] Please refund my duplicate charge.")
	// The Router asks one Noul per forward. Read each equivalent question
	// separately too: changing a CPU GEMM's batch shape changes FP32 rounding.
	answers := modelruntime.DecisionsResponse{Answers: map[string]modelruntime.Answer{}}
	for _, id := range []string{vela31Signal, "equivalent"} {
		response, referenceErr := runtime.Decide(ctx, modelruntime.DecisionsRequest{
			Model: mrVela31Deployment, State: prompt,
			Questions: map[string]modelruntime.Question{id: vela31Questions[id]},
		})
		if referenceErr != nil {
			return fmt.Errorf("single-question reference on the deployed Vela 3.1 fixture: %w", referenceErr)
		}
		answers.Answers[id] = response.Answers[id]
	}
	probability, err := checkVela31Answers(answers)
	if err != nil {
		return err
	}
	preview, err := session.preview(ctx, prompt)
	if err != nil {
		return err
	}
	if err = checkVela31Preview(preview, probability); err != nil {
		return err
	}
	response, err := session.chat(ctx, prompt)
	if err != nil {
		return err
	}
	if err = checkVela31Chat(response); err != nil {
		return err
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"noul_probability": probability, "choice_yes_probability": answers.Answers["equivalent"].Probabilities["yes"],
			"router_signal_probability": preview.SignalValues["systemone:"+vela31Signal],
			"minimum_probability":       vela31MinimumProbability, "decision": vela31Decision, "model": vela31Model,
		})
	}
	return nil
}

func checkVela31Answers(response modelruntime.DecisionsResponse) (float64, error) {
	noul, ok := response.Answers[vela31Signal]
	if !ok || noul.Error != "" || noul.Noul == nil {
		return 0, fmt.Errorf("described Noul has no probability: %+v", noul)
	}
	choice := response.Answers["equivalent"]
	yes, hasYes := choice.Probabilities["yes"]
	no, hasNo := choice.Probabilities["no"]
	probability := *noul.Noul
	if choice.Error != "" || !hasYes || !hasNo || len(choice.Probabilities) != 2 ||
		!validVela31Probability(probability) || !validVela31Probability(yes) || !validVela31Probability(no) ||
		math.Abs(yes+no-1) > 1e-9 || math.Abs(probability-yes) > 1e-9 {
		return 0, fmt.Errorf("described Noul %v differs from equivalent no/yes Choice %+v", probability, choice)
	}
	if probability < vela31MinimumProbability {
		return 0, fmt.Errorf("described Noul probability %v is below the route's nonzero predicate %v", probability, vela31MinimumProbability)
	}
	return probability, nil
}

func validVela31Probability(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0 && value <= 1
}

func checkVela31Preview(preview routingPreview, probability float64) error {
	if err := onlyOfflineSignalError(preview.SignalErrors); err != nil {
		return err
	}
	value, ok := preview.SignalValues["systemone:"+vela31Signal]
	if !ok || !validVela31Probability(value) || math.Abs(value-probability) > 1e-9 {
		return fmt.Errorf("router described-Noul signal %v (present %v), deployed runtime answered %v", value, ok, probability)
	}
	if preview.DecisionResult == nil || preview.DecisionResult.DecisionName != vela31Decision ||
		!slices.Contains(preview.DecisionResult.MatchedSignals.SystemOne, vela31Signal) {
		return fmt.Errorf("routing preview must select %s with the described-Noul signal: %+v", vela31Decision, preview.DecisionResult)
	}
	return nil
}

func checkVela31Chat(response *localChatCompletionResponse) error {
	if decision := response.Headers.Get("x-vsr-selected-decision"); decision != vela31Decision {
		return fmt.Errorf("gateway selected decision %q, want %s", decision, vela31Decision)
	}
	if model := response.Headers.Get("x-vsr-selected-model"); model != vela31Model {
		return fmt.Errorf("gateway selected model %q, want %s", model, vela31Model)
	}
	if !signalItems(response.Headers, "systemone")[vela31Signal] {
		return fmt.Errorf("gateway trace lacks the described-Noul signal")
	}
	var completion struct {
		Object  string `json:"object"`
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(response.Body, &completion); err != nil {
		return fmt.Errorf("decode backend completion: %w", err)
	}
	if completion.Model != vela31Model {
		return fmt.Errorf("backend completion model %q, want %s", completion.Model, vela31Model)
	}
	if completion.Object != "chat.completion" || len(completion.Choices) == 0 || strings.TrimSpace(completion.Choices[0].Message.Content) == "" {
		return fmt.Errorf("gateway returned no backend chat completion: %s", response.Body)
	}
	return nil
}
