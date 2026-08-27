package llmprotocol

import (
	"fmt"
	"strings"
)

type Capability uint64

const (
	CapabilityText Capability = 1 << iota
	CapabilityImageInput
	CapabilityImageOutput
	CapabilityAudioInput
	CapabilityAudioOutput
	CapabilityVideoInput
	CapabilityVideoOutput
	CapabilityFileInput
	CapabilityFileOutput
	CapabilityTools
	CapabilityParallelTools
	CapabilityReasoning
	CapabilityStructuredJSON
	CapabilityStrictJSONSchema
	CapabilityStrictToolSchema
	CapabilityStreaming
	CapabilityCacheAccounting
	CapabilityReasoningAccounting
	CapabilityAuthoritativeUsage
	CapabilityMultipleCandidates
)

// CapabilitySet is an immutable value bitset.
type CapabilitySet struct{ bits Capability }

func Capabilities(values ...Capability) CapabilitySet {
	var bits Capability
	for _, value := range values {
		bits |= value
	}
	return CapabilitySet{bits: bits}
}

func (set CapabilitySet) Supports(value Capability) bool { return set.bits&value == value }
func (set CapabilitySet) Contains(required CapabilitySet) bool {
	return set.bits&required.bits == required.bits
}

func (set CapabilitySet) Intersect(other CapabilitySet) CapabilitySet {
	return CapabilitySet{bits: set.bits & other.bits}
}
func (set CapabilitySet) Empty() bool { return set.bits == 0 }

func (set CapabilitySet) Names() []string {
	known := []struct {
		capability Capability
		name       string
	}{
		{CapabilityText, "text"},
		{CapabilityImageInput, "image_input"},
		{CapabilityImageOutput, "image_output"},
		{CapabilityAudioInput, "audio_input"},
		{CapabilityAudioOutput, "audio_output"},
		{CapabilityVideoInput, "video_input"},
		{CapabilityVideoOutput, "video_output"},
		{CapabilityFileInput, "file_input"},
		{CapabilityFileOutput, "file_output"},
		{CapabilityTools, "tools"},
		{CapabilityParallelTools, "parallel_tools"},
		{CapabilityReasoning, "reasoning"},
		{CapabilityStructuredJSON, "structured_json"},
		{CapabilityStrictJSONSchema, "strict_json_schema"},
		{CapabilityStrictToolSchema, "strict_tool_schema"},
		{CapabilityStreaming, "streaming"},
		{CapabilityCacheAccounting, "cache_accounting"},
		{CapabilityReasoningAccounting, "reasoning_accounting"},
		{CapabilityAuthoritativeUsage, "authoritative_usage"},
		{CapabilityMultipleCandidates, "multiple_candidates"},
	}
	names := make([]string, 0, len(known))
	for _, item := range known {
		if set.Supports(item.capability) {
			names = append(names, item.name)
		}
	}
	return names
}

func RequiredCapabilities(request Request) CapabilitySet {
	required := requestOptionCapabilities(request)
	required.bits |= toolCapabilities(request.Tools)
	required.bits |= instructionCapabilities(request.Instructions)
	required.bits |= messageCapabilities(request.Messages)
	return required
}

func requestOptionCapabilities(request Request) CapabilitySet {
	required := CapabilitySet{}
	if request.Stream {
		required.bits |= CapabilityStreaming
	}
	if len(request.Tools) > 0 || request.ToolChoice.Mode != "" && request.ToolChoice.Mode != ToolChoiceNone {
		required.bits |= CapabilityTools
	}
	if request.ParallelToolCalls != nil && *request.ParallelToolCalls {
		required.bits |= CapabilityParallelTools
	}
	if request.CandidateCount != nil && *request.CandidateCount > 1 {
		required.bits |= CapabilityMultipleCandidates
	}
	required.bits |= outputOptionCapabilities(request)
	return required
}

func outputOptionCapabilities(request Request) Capability {
	var required Capability
	if request.OutputFormat.Kind == OutputJSONObject {
		required |= CapabilityStructuredJSON
	}
	if request.OutputFormat.Kind == OutputJSONSchema {
		required |= CapabilityStrictJSONSchema
	}
	if request.ReasoningEffort != "" || request.ReasoningBudgetTokens != nil {
		required |= CapabilityReasoning
	}
	return required
}

func toolCapabilities(tools []Tool) Capability {
	var required Capability
	for _, tool := range tools {
		if tool.Strict != nil && *tool.Strict {
			required |= CapabilityStrictToolSchema
		}
	}
	return required
}

func instructionCapabilities(instructions []InstructionBlock) Capability {
	var required Capability
	for _, instruction := range instructions {
		for _, block := range instruction.Content {
			required |= capabilityForRequestContent(block)
		}
	}
	return required
}

func messageCapabilities(messages []Message) Capability {
	var required Capability
	for _, message := range messages {
		for _, block := range message.Content {
			required |= capabilityForRequestContent(block)
		}
	}
	return required
}

func RequiredResponseCapabilities(response Response) CapabilitySet {
	var required CapabilitySet
	sequences := append([][]OutputItem{response.Output}, response.Alternatives...)
	for _, sequence := range sequences {
		for _, item := range sequence {
			for _, block := range item.Content {
				required.bits |= capabilityForResponseContent(block)
			}
		}
	}
	return required
}

func capabilityForRequestContent(content Content) Capability {
	switch content.Kind {
	case ContentText, ContentRefusal:
		return CapabilityText
	case ContentImage:
		return CapabilityImageInput
	case ContentAudio:
		return CapabilityAudioInput
	case ContentVideo:
		return CapabilityVideoInput
	case ContentFile:
		return CapabilityFileInput
	case ContentToolCall:
		return CapabilityTools
	case ContentToolResult:
		capabilities := CapabilityTools
		if content.ToolResult != nil {
			for _, nested := range content.ToolResult.Content {
				capabilities |= capabilityForRequestContent(nested)
			}
		}
		return capabilities
	case ContentReasoning:
		return CapabilityText | CapabilityReasoning
	default:
		return 0
	}
}

func capabilityForResponseContent(content Content) Capability {
	switch content.Kind {
	case ContentText, ContentRefusal:
		return CapabilityText
	case ContentImage:
		return CapabilityImageOutput
	case ContentAudio:
		return CapabilityAudioOutput
	case ContentVideo:
		return CapabilityVideoOutput
	case ContentFile:
		return CapabilityFileOutput
	case ContentToolCall:
		return CapabilityTools
	case ContentToolResult:
		capabilities := CapabilityTools
		if content.ToolResult != nil {
			for _, nested := range content.ToolResult.Content {
				capabilities |= capabilityForResponseContent(nested)
			}
		}
		return capabilities
	case ContentReasoning:
		return CapabilityText | CapabilityReasoning
	default:
		return 0
	}
}

func ParseCapabilities(names []string) (CapabilitySet, error) {
	lookup := map[string]Capability{
		"text": CapabilityText, "chat": CapabilityText, "image_input": CapabilityImageInput,
		"image_output": CapabilityImageOutput, "audio_input": CapabilityAudioInput,
		"audio_output": CapabilityAudioOutput, "video_input": CapabilityVideoInput,
		"video_output": CapabilityVideoOutput, "file_input": CapabilityFileInput,
		"file_output": CapabilityFileOutput, "tools": CapabilityTools,
		"parallel_tools": CapabilityParallelTools, "reasoning": CapabilityReasoning,
		"structured_json": CapabilityStructuredJSON, "strict_json_schema": CapabilityStrictJSONSchema,
		"strict_tool_schema": CapabilityStrictToolSchema,
		"streaming":          CapabilityStreaming, "cache_accounting": CapabilityCacheAccounting,
		"reasoning_accounting": CapabilityReasoningAccounting,
		"authoritative_usage":  CapabilityAuthoritativeUsage,
		"multiple_candidates":  CapabilityMultipleCandidates,
	}
	var set CapabilitySet
	for _, name := range names {
		canonical := strings.ToLower(strings.TrimSpace(name))
		capability, found := lookup[canonical]
		if !found {
			return CapabilitySet{}, fmt.Errorf("unknown protocol capability %q", name)
		}
		set.bits |= capability
	}
	return set, nil
}
