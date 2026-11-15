package codexreview

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// The approval review Codex CLI 0.166.0 sent for one escalated shell command,
// captured at the gateway. The judge policy is shortened, one tool is kept,
// IDs are placeholders and the workspace path is neutral.
const reviewFixture = "testdata/codex-cli-0.166.0-approval-review.json"

func loadReview(t *testing.T) ([]byte, llmprotocol.Request) {
	t.Helper()
	body, err := os.ReadFile(reviewFixture)
	if err != nil {
		t.Fatal(err)
	}
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	return body, request
}

func TestCapturedReviewIsObservedWithItsIDsAndAction(t *testing.T) {
	body, request := loadReview(t)
	if !HasEnvelope(&request) {
		t.Fatal("the captured review carries the envelope")
	}
	review := Observe(&request, body, store.ReviewIdentifiedByContent)
	want := store.Review{
		SchemaVersion: 1, IdentificationSource: "content", Status: "pending",
		ReviewedSessionID: "00000000-0000-4000-8000-000000000005",
		ThreadID:          "00000000-0000-4000-8000-000000000007",
		TurnID:            "00000000-0000-4000-8000-000000000008",
		ParentTurnID:      "00000000-0000-4000-8000-000000000010",
	}
	action := review.Action
	review.Action = nil
	if *review != want {
		t.Fatalf("review = %+v, want %+v", *review, want)
	}
	if action == nil || action.Tool != "exec_command" || action.Summary != "printf 'hi' > hello.txt" || action.Cwd != "/work/project" {
		t.Fatalf("action = %+v", action)
	}
	var arguments map[string]any
	if err := json.Unmarshal([]byte(action.Arguments), &arguments); err != nil || arguments["sandbox_permissions"] != "require_escalated" || action.ArgumentsTruncated {
		t.Fatalf("arguments = %q (%v)", action.Arguments, err)
	}
}

// Each part of the envelope is required, so a request that only talks about
// reviews, or carries one part, is not a review.
func TestRequestsWithoutTheWholeEnvelopeAreNotReviews(t *testing.T) {
	_, captured := loadReview(t)
	block := ">>> APPROVAL REQUEST START\nPlanned action JSON:\n{\"tool\":\"exec_command\"}\n>>> APPROVAL REQUEST END"
	text := func(value string) []llmprotocol.Content {
		return []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: value}}
	}
	for name, mutate := range map[string]func(*llmprotocol.Request){
		"unrelated request about reviews": func(request *llmprotocol.Request) {
			*request = llmprotocol.Request{Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: text(
				"Please review this planned action and tell me allow or deny. You are judging one planned coding-agent action. " + block)}}}
		},
		"no output schema": func(request *llmprotocol.Request) { request.OutputFormat = llmprotocol.OutputFormat{} },
		"schema without a verdict": func(request *llmprotocol.Request) {
			request.OutputFormat.Schema = json.RawMessage(`{"properties":{"answer":{}}}`)
		},
		"other instructions": func(request *llmprotocol.Request) { request.Instructions = request.Instructions[1:] },
		"preamble in a user message": func(request *llmprotocol.Request) {
			request.Messages = append([]llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: request.Instructions[0].Content}}, request.Messages...)
			request.Instructions = request.Instructions[1:]
		},
		"no approval block": func(request *llmprotocol.Request) {
			request.Messages = request.Messages[:len(request.Messages)-1]
		},
	} {
		t.Run(name, func(t *testing.T) {
			request := captured
			request.Instructions = append([]llmprotocol.InstructionBlock(nil), captured.Instructions...)
			request.Messages = append([]llmprotocol.Message(nil), captured.Messages...)
			mutate(&request)
			if HasEnvelope(&request) {
				t.Fatal("recognised as a review")
			}
		})
	}
	if HasEnvelope(nil) {
		t.Fatal("a nil request is not a review")
	}
}

func TestObserveWithoutEnvelopeKeepsOnlyTheSource(t *testing.T) {
	request := llmprotocol.Request{Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser}}}
	review := Observe(&request, []byte(`{"client_metadata":{"x-codex-turn-metadata":"{\"session_id\":\"s\",\"turn_id\":\"t\"}"}}`), store.ReviewIdentifiedByEntrypoint)
	if review.IdentificationSource != "entrypoint" || review.Action != nil || review.ReviewedSessionID != "s" || review.TurnID != "t" {
		t.Fatalf("review = %+v", review)
	}
	if long := Observe(&request, []byte(`{"client_metadata":{"thread_id":"`+strings.Repeat("x", 300)+`"}}`), "entrypoint"); len(long.ThreadID) != maxIDBytes {
		t.Fatalf("thread id kept %d bytes", len(long.ThreadID))
	}
}

func TestPlannedActionSummaries(t *testing.T) {
	for name, test := range map[string]struct{ action, summary string }{
		"shell script":    {`{"tool":"shell","command":["/bin/zsh","-c","git status\n  --short"]}`, "git status --short"},
		"plain command":   {`{"tool":"exec_command","command":["git","push","fork"]}`, "git push fork"},
		"command string":  {`{"command":"make test"}`, "make test"},
		"patch":           {`{"tool":"apply_patch","files":["a.go","b.go"],"patch":"..."}`, "a.go, b.go"},
		"network":         {`{"tool":"network_access","target":"https://example.com"}`, "https://example.com"},
		"mcp":             {`{"tool":"mcp_tool_call","server":"docs","tool_name":"search"}`, "docs.search"},
		"nothing to show": {`{"tool":"other"}`, ""},
	} {
		t.Run(name, func(t *testing.T) {
			action := plannedAction(plannedActionLabel + "\n" + test.action + "\n")
			if action == nil || action.Summary != test.summary {
				t.Fatalf("action = %+v, want summary %q", action, test.summary)
			}
		})
	}
	long := plannedAction(plannedActionLabel + `{"command":"` + strings.Repeat("é", 400) + `","cwd":"/w"}`)
	if long == nil || len([]rune(long.Summary)) != maxSummaryRunes || !strings.HasSuffix(long.Summary, "…") {
		t.Fatalf("long summary = %q", long.Summary)
	}
	if plannedAction(plannedActionLabel+" not json") != nil || plannedAction("no label") != nil {
		t.Fatal("an unreadable action is left out")
	}
}
