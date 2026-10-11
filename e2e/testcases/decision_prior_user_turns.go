package testcases

import (
	"context"
	"fmt"
	"net/http"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("decision-prior-user-turns", pkgtestcases.TestCase{
		Description: "A decision signal with prior_user_turns routes a follow-up on the runtime's answer for the earlier user turn and the follow-up together",
		Tags:        []string{"decision", "model-runtime", "routing", "history"},
		Fn:          testDecisionPriorUserTurns,
	})
}

// The follow_up_kind question of e2e/profiles/model-runtime/values.yaml.
var decisionFollowUpQuestion = map[string]modelruntime.Question{
	"follow_up_kind": {
		Type:         "choice",
		Instructions: "What kind of work does the latest user turn ask for?",
		Choices: []modelruntime.Choice{
			{Key: "code", Description: "Writing, reviewing or debugging code"},
			{Key: "math", Description: "Mathematics or quantitative reasoning"},
			{Key: "writing", Description: "Prose, poems or other writing"},
			{Key: "chat", Description: "Anything else"},
		},
	},
}

const decisionFollowUp = "Now make it shorter."

var decisionFirstTurns = []string{
	"Write a Go function that merges two sorted slices.",
	"Prove that the square root of two is irrational.",
	"Write a short poem about rain on a tin roof.",
	"Explain how a hash map handles collisions.",
	"Draft an email asking my landlord to fix the heating.",
	"Solve x squared minus five x plus six equals zero.",
}

// testDecisionPriorUserTurns finds a first turn for which the fixture runtime
// answers the follow-up differently alone and after that turn, so the route
// shows which state the Router asked. The fixture weights are random, so the
// runtime's own answers are the expectation.
func testDecisionPriorUserTurns(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := openModelRuntimeSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	if err = session.waitReady(ctx, mrDecisionDeployment); err != nil {
		return err
	}
	managed, _, err := session.managed(ctx, mrDecisionDeployment)
	if err != nil {
		return err
	}
	for _, first := range decisionFirstTurns {
		alone, err := followUpChoice(ctx, managed, decisionFollowUp)
		if err != nil {
			return err
		}
		withPrior, err := followUpChoice(ctx, managed, first+"\n\n"+decisionFollowUp)
		if err != nil {
			return err
		}
		if alone == withPrior {
			continue
		}
		messages := []map[string]string{
			{"role": "user", "content": first},
			{"role": "assistant", "content": "Here it is."},
			{"role": "user", "content": decisionFollowUp},
		}
		response, err := sendLocalChatConversation(ctx, session.gatewayPort, "vllm-sr/auto", messages, mrRequestTimeout)
		if err != nil {
			return err
		}
		if response.StatusCode != http.StatusOK {
			return fmt.Errorf("follow-up chat: %s", formatUnexpectedChatCompletionStatus(response))
		}
		matched := headerItems(response.Headers, "x-vsr-matched-decision-model")
		if !matched["follow_up_kind:"+withPrior] || matched["follow_up_kind:"+alone] {
			return fmt.Errorf("x-vsr-matched-decision-model %v: want follow_up_kind:%s (the runtime's answer after %q), not follow_up_kind:%s (the follow-up alone)",
				matched, withPrior, first, alone)
		}
		if opts.SetDetails != nil {
			opts.SetDetails(map[string]interface{}{"first_turn": first, "alone": alone, "with_prior_turn": withPrior})
		}
		return nil
	}
	return fmt.Errorf("the fixture answers %q the same alone and after each of %d first turns, so no route can show which state was asked", decisionFollowUp, len(decisionFirstTurns))
}

func followUpChoice(ctx context.Context, runtime *modelruntime.Client, state string) (string, error) {
	answers, err := runtime.Decide(ctx, modelruntime.DecisionsRequest{Model: mrDecisionDeployment, State: state, Questions: decisionFollowUpQuestion})
	if err != nil {
		return "", err
	}
	return answers.Answers["follow_up_kind"].Choice, nil
}
