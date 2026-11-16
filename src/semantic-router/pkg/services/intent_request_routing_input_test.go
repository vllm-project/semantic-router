package services

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routinginput"
)

const (
	previewQuestion = "Compare the three supplier contracts and draft a summary."
	previewMemories = "<relevant-memories>- prefers short answers</relevant-memories>"
	previewContext  = "OpenClaw runtime context:\nConversation info: {\"chat_id\": \"chat-example\"}\nEnd OpenClaw runtime context."
)

func previewSelector(t *testing.T) *routinginput.Selector {
	t.Helper()
	selector, err := routinginput.Compile(config.RoutingInputConfig{
		SkipUserMessagesMatching: []string{`(?s)^\s*OpenClaw runtime context:.*End OpenClaw runtime context\.\s*$`},
		StripBlocks:              []config.RoutingInputStripBlock{{Start: "<relevant-memories>", End: "</relevant-memories>"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return selector
}

func previewMessage(t *testing.T, role string, content any) IntentMessage {
	t.Helper()
	encoded, err := json.Marshal(content)
	if err != nil {
		t.Fatal(err)
	}
	return IntentMessage{Role: role, Content: encoded}
}

// The routing preview must score the text live routing scores.
func TestResolveSignalInputAppliesRoutingInput(t *testing.T) {
	req := IntentRequest{Messages: []IntentMessage{
		previewMessage(t, "system", "You are a helpful agent."),
		previewMessage(t, "user", previewMemories+"\n\n"+previewQuestion),
		previewMessage(t, "user", previewContext),
	}}
	plain, err := req.resolveSignalInput()
	if err != nil || plain.evaluationText != previewContext {
		t.Fatalf("without routing_input the preview scores %q (%v), want the runtime-context turn", plain.evaluationText, err)
	}

	input, err := req.resolveSignalInputWith(previewSelector(t))
	if err != nil {
		t.Fatal(err)
	}
	if input.evaluationText != previewQuestion || input.currentUserText != previewQuestion || len(input.priorUserMessages) != 0 {
		t.Fatalf("preview routing text = %q, current = %q, prior = %q", input.evaluationText, input.currentUserText, input.priorUserMessages)
	}
	if facts := input.conversationFacts; facts.UserMessageCount != 1 || facts.LastMessageRole != "user" {
		t.Fatalf("conversation facts = %+v, want the skipped turn uncounted", facts)
	}
	routing := input.requestFacts.RoutingInput
	if routing.SkippedMessages != 1 || routing.StrippedBlocks != 1 ||
		!slices.Equal(routing.ExcludedCurrent, []string{previewMemories, previewContext}) {
		t.Fatalf("safety signals would not see the removed text: %+v", routing)
	}
	if input.requestFacts.ContextTokenFloor != plain.requestFacts.ContextTokenFloor {
		t.Fatal("the context estimate must keep the request as sent")
	}
}

// A conversation whose every user message matches routing_input is scored
// on its last user message, as live routing routes it, not on the assistant's
// text or nothing.
func TestResolveSignalInputScoresTheLastUserMessageWhenEveryOneMatches(t *testing.T) {
	req := IntentRequest{Messages: []IntentMessage{
		previewMessage(t, "user", previewContext),
		previewMessage(t, "assistant", "Noted."),
		previewMessage(t, "user", previewMemories),
	}}
	input, err := req.resolveSignalInputWith(previewSelector(t))
	if err != nil {
		t.Fatal(err)
	}
	if input.evaluationText != previewMemories || input.currentUserText != previewMemories {
		t.Fatalf("preview routing text = %q, current = %q, want the last user message", input.evaluationText, input.currentUserText)
	}
	if facts := input.conversationFacts; facts.UserMessageCount != 1 {
		t.Fatalf("conversation facts = %+v, want the routed turn counted", facts)
	}
}

func TestResolveSignalInputNeverSkipsMessagesWithMedia(t *testing.T) {
	req := IntentRequest{Messages: []IntentMessage{previewMessage(t, "user", []map[string]any{
		{"type": "text", "text": previewContext},
		{"type": "image_url", "image_url": map[string]string{"url": "data:image/png;base64,iVBORw0KGgo="}},
	})}}
	input, err := req.resolveSignalInputWith(previewSelector(t))
	if err != nil {
		t.Fatal(err)
	}
	if input.requestFacts.RoutingInput.SkippedMessages != 0 || input.evaluationText != previewContext {
		t.Fatalf("a message with an image was skipped: %+v", input.requestFacts.RoutingInput)
	}
}
