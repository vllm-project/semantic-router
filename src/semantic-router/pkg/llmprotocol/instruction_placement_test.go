package llmprotocol

import (
	"reflect"
	"testing"
)

func placementCall(id string) Message {
	return Message{Role: RoleAssistant, Content: []Content{{Kind: ContentToolCall, ToolCall: &ToolCall{ID: id, Name: "lookup", Arguments: "{}"}}}}
}

func placementResult(id string) Message {
	return Message{Role: RoleTool, Content: []Content{{Kind: ContentToolResult, ToolResult: &ToolResult{CallID: id, Content: systemProjectionText(id)}}}}
}

func placementText(role Role, text string) Message {
	return Message{Role: role, Content: systemProjectionText(text)}
}

// Responses accepts a developer item between a call and its output, and
// between parallel outputs. It moves after the last of those outputs, and
// nothing else changes place.
func TestInstructionsAfterToolResultsMovesOnlyThoseBetweenACallAndItsResults(t *testing.T) {
	for _, check := range []struct {
		name  string
		in    []Message
		want  []Message
		moved int
	}{
		{
			"between call and output",
			[]Message{placementCall("a"), placementText(RoleDeveloper, "brief"), placementResult("a"), placementText(RoleUser, "next")},
			[]Message{placementCall("a"), placementResult("a"), placementText(RoleDeveloper, "brief"), placementText(RoleUser, "next")},
			1,
		},
		{
			"between parallel calls and outputs",
			[]Message{
				placementCall("a"), placementText(RoleSystem, "one"), placementCall("b"), placementResult("a"),
				placementText(RoleDeveloper, "two"), placementResult("b"),
			},
			[]Message{
				placementCall("a"), placementCall("b"), placementResult("a"), placementResult("b"),
				placementText(RoleSystem, "one"), placementText(RoleDeveloper, "two"),
			},
			2,
		},
		{
			"a call without output ends at the next user turn",
			[]Message{placementCall("a"), placementText(RoleDeveloper, "brief"), placementText(RoleUser, "next")},
			[]Message{placementCall("a"), placementText(RoleDeveloper, "brief"), placementText(RoleUser, "next")},
			0,
		},
		{
			"only the first held instruction crosses content",
			[]Message{placementCall("a"), placementText(RoleSystem, "one"), placementResult("other"), placementText(RoleDeveloper, "two")},
			[]Message{placementCall("a"), placementResult("other"), placementText(RoleSystem, "one"), placementText(RoleDeveloper, "two")},
			1,
		},
		{
			"a text assistant turn ends an unanswered span",
			[]Message{placementCall("a"), placementText(RoleDeveloper, "brief"), placementText(RoleAssistant, "never mind"), placementText(RoleUser, "next")},
			[]Message{placementCall("a"), placementText(RoleDeveloper, "brief"), placementText(RoleAssistant, "never mind"), placementText(RoleUser, "next")},
			0,
		},
		{
			"outside a call stays in place",
			[]Message{
				placementText(RoleUser, "hi"), placementText(RoleSystem, "French"), placementCall("a"), placementResult("a"),
				placementText(RoleDeveloper, "brief"), placementText(RoleUser, "next"),
			},
			[]Message{
				placementText(RoleUser, "hi"), placementText(RoleSystem, "French"), placementCall("a"), placementResult("a"),
				placementText(RoleDeveloper, "brief"), placementText(RoleUser, "next"),
			},
			0,
		},
	} {
		t.Run(check.name, func(t *testing.T) {
			placed, moved := InstructionsAfterToolResults(check.in)
			if moved != check.moved || !reflect.DeepEqual(placed, check.want) {
				t.Fatalf("placed %d: %+v", moved, placed)
			}
		})
	}
}
