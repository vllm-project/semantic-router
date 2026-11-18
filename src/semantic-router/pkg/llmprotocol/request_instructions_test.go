package llmprotocol

import (
	"reflect"
	"testing"
)

func instructionPositions(instructions []InstructionBlock) []int {
	positions := make([]int, 0, len(instructions))
	for _, instruction := range instructions {
		positions = append(positions, instruction.Position)
	}
	return positions
}

// A later instruction stays after the messages the client sent before it when
// the Router inserts messages before them, and in front of messages inserted
// at its own place.
func TestInsertMessagesMovesLaterInstructionsWithTheirMessages(t *testing.T) {
	text := systemProjectionText
	shared := []InstructionBlock{
		{Role: RoleSystem, Content: text("leading")},
		{Role: RoleSystem, Content: text("after the question"), Position: 1},
		{Role: RoleSystem, Content: text("after the answer"), Position: 2},
	}
	request := Request{Instructions: shared, Messages: []Message{
		{Role: RoleUser, Content: text("question")},
		{Role: RoleAssistant, Content: text("answer")},
	}}
	InsertMessages(&request, 1, Message{Role: RoleAssistant, Content: text("call")}, Message{Role: RoleTool, Content: text("result")})
	if got := instructionPositions(request.Instructions); !reflect.DeepEqual(got, []int{0, 1, 4}) {
		t.Fatalf("positions = %v", got)
	}
	if len(request.Messages) != 4 || request.Messages[1].Content[0].Text != "call" || request.Messages[3].Content[0].Text != "answer" {
		t.Fatalf("conversation = %+v", request.Messages)
	}
	if shared[2].Position != 2 {
		t.Fatal("the insert moved an instruction another request shares")
	}
}

// Removing messages moves each later instruction back past the removed ones
// before it, and keeps the instructions of a request that shares them.
func TestStripToolsKeepsLaterInstructionsAfterTheirMessages(t *testing.T) {
	text := systemProjectionText
	shared := []InstructionBlock{
		{Role: RoleSystem, Content: text("leading")},
		{Role: RoleSystem, Content: text("after the tool result"), Position: 3},
		{Role: RoleSystem, Content: text("after the second question"), Position: 4},
	}
	request := Request{Instructions: shared, Messages: []Message{
		{Role: RoleUser, Content: text("question")},
		{Role: RoleAssistant, Content: []Content{{Kind: ContentToolCall, ToolCall: &ToolCall{ID: "call_1", Name: "read"}}}},
		{Role: RoleTool, Content: []Content{{Kind: ContentToolResult, ToolResult: &ToolResult{CallID: "call_1"}}}},
		{Role: RoleUser, Content: text("second question")},
	}}
	if changed, _ := StripTools(&request, true); !changed || len(request.Messages) != 2 {
		t.Fatalf("changed %t, conversation %+v", changed, request.Messages)
	}
	if got := instructionPositions(request.Instructions); !reflect.DeepEqual(got, []int{0, 1, 2}) {
		t.Fatalf("positions = %v", got)
	}
	if shared[1].Position != 3 || shared[2].Position != 4 {
		t.Fatal("the removal moved instructions another request shares")
	}
	unchanged := []InstructionBlock{{Role: RoleSystem, Content: text("leading")}}
	if kept := KeepInstructionPositions(unchanged, []bool{false, true}); &kept[0] != &unchanged[0] {
		t.Fatal("leading instructions were copied although none moved")
	}
}
