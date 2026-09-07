package contextcompression

import (
	"encoding/json"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type MessageIR struct {
	Index       int
	Role        string
	Raw         map[string]interface{}
	ToolCallID  string
	ToolName    string
	Protected   bool
	Blocks      []*TextBlockIR
	ToolCallIDs []string
}

type TextBlockIR struct {
	MessageIndex int
	BlockIndex   int
	Role         string
	ToolCallID   string
	Source       TargetKind
	Text         string
	JSON         bool

	parent  map[string]interface{}
	field   string
	setText func(string)
}

func (block *TextBlockIR) SetText(text string) {
	if block == nil {
		return
	}
	block.Text = text
	if block.setText != nil {
		block.setText(text)
		return
	}
	if block.parent == nil || block.field == "" {
		return
	}
	block.parent[block.field] = text
}

type RequestIR struct {
	Raw         map[string]interface{}
	Semantic    *llmprotocol.Request
	Messages    []*MessageIR
	ToolIntents map[string]string
	LastUser    string
}

// ParseSemanticRequest builds the compression view directly over neutral
// request content. Text mutations are applied to the original semantic blocks;
// no protocol wire representation is manufactured inside the Router.
//
//nolint:gocognit // Conversion walks the complete neutral message and tool-result hierarchy.
func ParseSemanticRequest(request *llmprotocol.Request, provenance Provenance) *RequestIR {
	ir := &RequestIR{Semantic: request, ToolIntents: make(map[string]string)}
	if request == nil {
		return ir
	}
	lastUser, lastAssistant := -1, -1
	for index := range request.Messages {
		message := &request.Messages[index]
		entry := &MessageIR{Index: index, Role: string(message.Role)}
		for contentIndex := range message.Content {
			content := &message.Content[contentIndex]
			appendSemanticContent(ir, entry, contentIndex, content, provenance)
		}
		if _, memory := provenance.MemoryMessageIndexes[index]; memory {
			for _, block := range entry.Blocks {
				block.Source = TargetMemory
			}
		}
		if message.Role == llmprotocol.RoleUser {
			lastUser = index
			for _, block := range entry.Blocks {
				if strings.TrimSpace(block.Text) != "" {
					ir.LastUser = block.Text
				}
			}
		}
		if message.Role == llmprotocol.RoleAssistant {
			lastAssistant = index
		}
		ir.Messages = append(ir.Messages, entry)
	}
	for _, message := range ir.Messages {
		message.Protected = message.Index == lastUser || message.Index == lastAssistant
	}
	protectAtomicToolExchanges(ir.Messages)
	return ir
}

func appendSemanticContent(
	ir *RequestIR,
	message *MessageIR,
	contentIndex int,
	content *llmprotocol.Content,
	provenance Provenance,
) {
	switch content.Kind {
	case llmprotocol.ContentText, llmprotocol.ContentRefusal, llmprotocol.ContentReasoning:
		message.Blocks = append(message.Blocks, semanticTextBlock(message, contentIndex, "", TargetHistory, content))
	case llmprotocol.ContentToolCall:
		if content.ToolCall == nil {
			return
		}
		call := content.ToolCall
		message.ToolCallIDs = append(message.ToolCallIDs, call.ID)
		ir.ToolIntents[call.ID] = strings.TrimSpace(call.Name + " " + call.Arguments)
	case llmprotocol.ContentToolResult:
		appendSemanticToolResult(message, contentIndex, content.ToolResult, provenance)
	}
}

func appendSemanticToolResult(
	message *MessageIR,
	contentIndex int,
	result *llmprotocol.ToolResult,
	provenance Provenance,
) {
	if result == nil {
		return
	}
	message.ToolCallID = result.CallID
	source := TargetToolOutput
	if isRAGToolMessage(result.CallID, provenance) {
		source = TargetRAG
	}
	for nestedIndex := range result.Content {
		nested := &result.Content[nestedIndex]
		if nested.Kind == llmprotocol.ContentText || nested.Kind == llmprotocol.ContentRefusal {
			message.Blocks = append(message.Blocks, semanticTextBlock(
				message, contentIndex*1000+nestedIndex, result.CallID, source, nested,
			))
		}
	}
}

func semanticTextBlock(
	message *MessageIR,
	blockIndex int,
	toolCallID string,
	source TargetKind,
	content *llmprotocol.Content,
) *TextBlockIR {
	block := &TextBlockIR{
		MessageIndex: message.Index,
		BlockIndex:   blockIndex,
		Role:         message.Role,
		ToolCallID:   toolCallID,
		Source:       source,
		Text:         content.Text,
		JSON:         isJSONObjectOrArray(content.Text),
	}
	block.setText = func(text string) { content.Text = text }
	return block
}

func ParseRequestIR(
	request map[string]interface{},
	provenance Provenance,
) *RequestIR {
	ir := &RequestIR{
		Raw:         request,
		ToolIntents: make(map[string]string),
	}
	rawMessages, _ := request["messages"].([]interface{})
	lastUser := -1
	lastAssistant := -1
	for index, rawMessage := range rawMessages {
		message, ok := rawMessage.(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := message["role"].(string)
		entry := &MessageIR{
			Index: index,
			Role:  role,
			Raw:   message,
		}
		entry.ToolCallID, _ = message["tool_call_id"].(string)
		entry.ToolName, _ = message["name"].(string)
		ir.Messages = append(ir.Messages, entry)
		if role == "user" {
			lastUser = index
			if text := messageText(message["content"]); strings.TrimSpace(text) != "" {
				ir.LastUser = text
			}
		}
		if role == "assistant" {
			lastAssistant = index
			collectToolIntents(ir.ToolIntents, message)
		}
	}
	for _, message := range ir.Messages {
		message.Protected = message.Role == "system" ||
			message.Index == lastUser ||
			message.Index == lastAssistant
		source := TargetToolOutput
		if _, ok := provenance.MemoryMessageIndexes[message.Index]; ok {
			source = TargetMemory
		}
		if isRAGToolMessage(message.ToolCallID, provenance) {
			source = TargetRAG
		}
		message.Blocks = parseMessageBlocks(message, source, provenance)
	}
	protectAtomicToolExchanges(ir.Messages)
	return ir
}

func protectAtomicToolExchanges(messages []*MessageIR) {
	owners := make(map[string]*MessageIR)
	results := make(map[string]*MessageIR)
	for _, message := range messages {
		indexToolExchange(message, owners, results)
	}
	for id, owner := range owners {
		protectToolExchange(owner, results[id])
	}
}

func indexToolExchange(
	message *MessageIR,
	owners map[string]*MessageIR,
	results map[string]*MessageIR,
) {
	if message.Role == "assistant" {
		for _, id := range messageToolCallIDs(message) {
			owners[id] = message
		}
	}
	if message.ToolCallID != "" {
		results[message.ToolCallID] = message
	}
	for _, block := range message.Blocks {
		if block.ToolCallID != "" && block.Source != TargetHistory {
			results[block.ToolCallID] = message
		}
	}
}

func messageToolCallIDs(message *MessageIR) []string {
	if len(message.ToolCallIDs) > 0 {
		return message.ToolCallIDs
	}
	return assistantToolCallIDs(message.Raw)
}

func protectToolExchange(owner, result *MessageIR) {
	if result == nil || (!owner.Protected && !result.Protected) {
		return
	}
	owner.Protected = true
	result.Protected = true
}

func assistantToolCallIDs(message map[string]interface{}) []string {
	ids := make([]string, 0)
	toolCalls, _ := message["tool_calls"].([]interface{})
	for _, rawCall := range toolCalls {
		call, _ := rawCall.(map[string]interface{})
		id, _ := call["id"].(string)
		if id != "" {
			ids = append(ids, id)
		}
	}
	content, _ := message["content"].([]interface{})
	for _, rawBlock := range content {
		block, _ := rawBlock.(map[string]interface{})
		if block["type"] != "tool_use" {
			continue
		}
		id, _ := block["id"].(string)
		if id != "" {
			ids = append(ids, id)
		}
	}
	return ids
}

func (request *RequestIR) QueryFor(block *TextBlockIR) string {
	if request == nil || block == nil {
		return ""
	}
	if intent := request.ToolIntents[block.ToolCallID]; strings.TrimSpace(intent) != "" {
		return intent
	}
	return request.LastUser
}

func (request *RequestIR) Marshal() ([]byte, error) {
	return json.Marshal(request.Raw)
}

func collectToolIntents(intents map[string]string, message map[string]interface{}) {
	toolCalls, _ := message["tool_calls"].([]interface{})
	for _, rawCall := range toolCalls {
		call, ok := rawCall.(map[string]interface{})
		if !ok {
			continue
		}
		id, _ := call["id"].(string)
		function, _ := call["function"].(map[string]interface{})
		intent := renderToolIntent(function)
		if id != "" && intent != "" {
			intents[id] = intent
		}
	}
	functionCall, _ := message["function_call"].(map[string]interface{})
	if intent := renderToolIntent(functionCall); intent != "" {
		name, _ := functionCall["name"].(string)
		if name != "" {
			intents["function:"+name] = intent
		}
	}
	content, _ := message["content"].([]interface{})
	for _, rawBlock := range content {
		block, ok := rawBlock.(map[string]interface{})
		if !ok || block["type"] != "tool_use" {
			continue
		}
		id, _ := block["id"].(string)
		if id != "" {
			intents[id] = renderAnthropicToolIntent(block)
		}
	}
}

func renderToolIntent(function map[string]interface{}) string {
	if function == nil {
		return ""
	}
	name, _ := function["name"].(string)
	arguments := renderJSONValue(function["arguments"])
	return strings.TrimSpace(strings.TrimSpace(name) + " " + arguments)
}

func renderAnthropicToolIntent(block map[string]interface{}) string {
	name, _ := block["name"].(string)
	return strings.TrimSpace(strings.TrimSpace(name) + " " + renderJSONValue(block["input"]))
}

func renderJSONValue(value interface{}) string {
	switch typed := value.(type) {
	case string:
		return typed
	case nil:
		return ""
	default:
		encoded, err := json.Marshal(typed)
		if err != nil {
			return ""
		}
		return string(encoded)
	}
}

func parseMessageBlocks(
	message *MessageIR,
	source TargetKind,
	provenance Provenance,
) []*TextBlockIR {
	switch content := message.Raw["content"].(type) {
	case string:
		toolID := message.ToolCallID
		if message.Role == "function" && toolID == "" {
			toolID = "function:" + message.ToolName
		}
		kind := TargetHistory
		if source == TargetMemory ||
			source == TargetRAG ||
			message.Role == "tool" ||
			message.Role == "function" {
			kind = source
		}
		return []*TextBlockIR{newTextBlock(message, -1, toolID, kind, content, message.Raw, "content")}
	case []interface{}:
		return parseArrayBlocks(message, content, source, provenance)
	default:
		return nil
	}
}

func parseArrayBlocks(
	message *MessageIR,
	blocks []interface{},
	source TargetKind,
	provenance Provenance,
) []*TextBlockIR {
	result := make([]*TextBlockIR, 0, len(blocks))
	for index, rawBlock := range blocks {
		block, ok := rawBlock.(map[string]interface{})
		if !ok {
			continue
		}
		blockType, _ := block["type"].(string)
		switch blockType {
		case "text", "input_text":
			text, _ := block["text"].(string)
			blockSource := TargetHistory
			if message.Role == "tool" || message.Role == "function" {
				blockSource = source
			}
			result = append(result, newTextBlock(
				message,
				index,
				message.ToolCallID,
				blockSource,
				text,
				block,
				"text",
			))
		case "tool_result":
			toolUseID, _ := block["tool_use_id"].(string)
			blockSource := source
			if isRAGToolMessage(toolUseID, provenance) {
				blockSource = TargetRAG
			}
			result = append(result, parseToolResultBlocks(
				message,
				index,
				toolUseID,
				blockSource,
				block,
			)...)
		}
	}
	return result
}

func parseToolResultBlocks(
	message *MessageIR,
	blockIndex int,
	toolUseID string,
	source TargetKind,
	block map[string]interface{},
) []*TextBlockIR {
	switch content := block["content"].(type) {
	case string:
		return []*TextBlockIR{newTextBlock(
			message,
			blockIndex,
			toolUseID,
			source,
			content,
			block,
			"content",
		)}
	case []interface{}:
		result := make([]*TextBlockIR, 0, len(content))
		for nestedIndex, rawNested := range content {
			nested, ok := rawNested.(map[string]interface{})
			if !ok || nested["type"] != "text" {
				continue
			}
			text, _ := nested["text"].(string)
			result = append(result, newTextBlock(
				message,
				blockIndex*1000+nestedIndex,
				toolUseID,
				source,
				text,
				nested,
				"text",
			))
		}
		return result
	default:
		return nil
	}
}

func newTextBlock(
	message *MessageIR,
	blockIndex int,
	toolCallID string,
	source TargetKind,
	text string,
	parent map[string]interface{},
	field string,
) *TextBlockIR {
	return &TextBlockIR{
		MessageIndex: message.Index,
		BlockIndex:   blockIndex,
		Role:         message.Role,
		ToolCallID:   toolCallID,
		Source:       source,
		Text:         text,
		JSON:         isJSONObjectOrArray(text),
		parent:       parent,
		field:        field,
	}
}

func isRAGToolMessage(toolCallID string, provenance Provenance) bool {
	if _, ok := provenance.RAGToolCallIDs[toolCallID]; ok {
		return true
	}
	return strings.HasPrefix(toolCallID, "rag_")
}

func messageText(content interface{}) string {
	switch typed := content.(type) {
	case string:
		return typed
	case []interface{}:
		parts := make([]string, 0, len(typed))
		for _, rawPart := range typed {
			part, ok := rawPart.(map[string]interface{})
			if !ok {
				continue
			}
			if text, ok := part["text"].(string); ok && text != "" {
				parts = append(parts, text)
			}
			if part["type"] == "tool_result" {
				if contentText, ok := part["content"].(string); ok {
					parts = append(parts, contentText)
				}
			}
		}
		return strings.Join(parts, "\n")
	default:
		return ""
	}
}
