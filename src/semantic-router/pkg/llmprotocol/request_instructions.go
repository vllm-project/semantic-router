package llmprotocol

import "slices"

// SetSystemInstruction applies a deterministic system instruction policy to the
// neutral request. insert prepends to the first system block; other modes
// replace every system and developer block the caller sent with one system
// block, matching the router's existing system_prompt contract.
func SetSystemInstruction(request *Request, text, mode string) bool {
	if request == nil || text == "" {
		return false
	}
	content := Content{Kind: ContentText, Text: text}
	if mode != "insert" {
		request.Instructions = []InstructionBlock{{Role: RoleSystem, Content: []Content{content}}}
		return true
	}
	for i := range request.Instructions {
		instruction := &request.Instructions[i]
		if instruction.Role != RoleSystem {
			continue
		}
		instruction.Content = append([]Content{content}, instruction.Content...)
		return true
	}
	request.Instructions = append([]InstructionBlock{{Role: RoleSystem, Content: []Content{content}}}, request.Instructions...)
	return true
}

// InsertMessages inserts messages into the conversation before the message at
// index at. An instruction the client sent after that message moves with it.
func InsertMessages(request *Request, at int, messages ...Message) {
	request.Messages = slices.Insert(request.Messages, at, messages...)
	request.Instructions = moveInstructions(request.Instructions, func(position int) int {
		if position > at {
			return position + len(messages)
		}
		return position
	})
}

// KeepInstructionPositions returns instructions placed among the
// conversation messages kept marks, one entry per message their positions
// count: each instruction stays after the kept messages that came before it.
func KeepInstructionPositions(instructions []InstructionBlock, kept []bool) []InstructionBlock {
	if !slices.ContainsFunc(instructions, func(instruction InstructionBlock) bool { return instruction.Position > 0 }) {
		return instructions
	}
	before := make([]int, len(kept)+1)
	for index, keep := range kept {
		before[index+1] = before[index]
		if keep {
			before[index+1]++
		}
	}
	return moveInstructions(instructions, func(position int) int {
		return before[min(position, len(kept))]
	})
}

// moveInstructions copies instructions only when a position changes, so a
// request that shares them keeps its own.
func moveInstructions(instructions []InstructionBlock, move func(int) int) []InstructionBlock {
	var moved []InstructionBlock
	for index, instruction := range instructions {
		position := move(instruction.Position)
		if position == instruction.Position {
			continue
		}
		if moved == nil {
			moved = slices.Clone(instructions)
		}
		moved[index].Position = position
	}
	if moved == nil {
		return instructions
	}
	return moved
}
