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
	CapabilityCacheDirectives
	CapabilityReasoningDisable
	CapabilityReasoningEffort
	CapabilityReasoningBudget
	CapabilitySamplingTopK
	CapabilitySamplingSeed
	CapabilitySamplingPenalties
	CapabilityStopSequences
	CapabilityRequestMetadata
	CapabilityRequestStorage
	CapabilityAutomaticStorage
	CapabilityConversationState
	CapabilityReasoningAdaptive
	CapabilityReasoningSignature
	CapabilityReasoningDisplay
	// CapabilityMatchedStopSequence is response fidelity, distinct from the
	// request-side ability to send stop sequences. Only a wire contract that
	// exposes the exact sequence which ended generation can preserve it.
	CapabilityMatchedStopSequence
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
		{CapabilityCacheDirectives, "cache_directives"},
		{CapabilityReasoningDisable, "reasoning_disable"},
		{CapabilityReasoningEffort, "reasoning_effort"},
		{CapabilityReasoningBudget, "reasoning_budget"},
		{CapabilitySamplingTopK, "sampling_top_k"},
		{CapabilitySamplingSeed, "sampling_seed"},
		{CapabilitySamplingPenalties, "sampling_penalties"},
		{CapabilityStopSequences, "stop_sequences"},
		{CapabilityRequestMetadata, "request_metadata"},
		{CapabilityRequestStorage, "request_storage"},
		{CapabilityAutomaticStorage, "automatic_storage"},
		{CapabilityConversationState, "conversation_state"},
		{CapabilityReasoningAdaptive, "reasoning_adaptive"},
		{CapabilityReasoningSignature, "reasoning_signature"},
		{CapabilityReasoningDisplay, "reasoning_display"},
		{CapabilityMatchedStopSequence, "matched_stop_sequence"},
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
	if request.Sampling.TopK != nil {
		required.bits |= CapabilitySamplingTopK
	}
	if request.Sampling.Seed != nil {
		required.bits |= CapabilitySamplingSeed
	}
	if request.Sampling.FrequencyPenalty != nil || request.Sampling.PresencePenalty != nil {
		required.bits |= CapabilitySamplingPenalties
	}
	if len(request.Sampling.Stop) > 0 {
		required.bits |= CapabilityStopSequences
	}
	if len(request.Metadata) > 0 {
		required.bits |= CapabilityRequestMetadata
	}
	if request.Store != nil {
		required.bits |= CapabilityRequestStorage
	}
	if request.AutoStore != nil {
		required.bits |= CapabilityAutomaticStorage
	}
	if request.PreviousResponseID != "" || request.ConversationID != "" || request.Truncation != "" {
		required.bits |= CapabilityConversationState
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
	if request.ReasoningEffort != "" {
		required |= CapabilityReasoning | CapabilityReasoningEffort
	}
	if request.ReasoningBudgetTokens != nil {
		required |= CapabilityReasoning | CapabilityReasoningBudget
	}
	if request.ReasoningDisplay != "" {
		required |= CapabilityReasoning | CapabilityReasoningDisplay
	}
	if request.ReasoningMode == ReasoningModeDisabled {
		required |= CapabilityReasoningDisable
	}
	if request.ReasoningMode == ReasoningModeAdaptive {
		required |= CapabilityReasoning | CapabilityReasoningAdaptive
	}
	return required
}

func toolCapabilities(tools []Tool) Capability {
	var required Capability
	for _, tool := range tools {
		if tool.Strict != nil && *tool.Strict {
			required |= CapabilityStrictToolSchema
		}
		if tool.Cache != nil {
			required |= CapabilityCacheDirectives
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
	if len(response.Alternatives) > 0 {
		required.bits |= CapabilityMultipleCandidates
	}
	if response.MatchedStopSequence != "" {
		required.bits |= CapabilityMatchedStopSequence
	}
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

// RequiredEventCapabilities describes fidelity that must survive at the
// streaming boundary. Most event semantics are already closed by the stream
// encoder contract. This method records response-only fields whose presence
// has no approximation in another public wire format.
func RequiredEventCapabilities(event Event) CapabilitySet {
	var required CapabilitySet
	if event.MatchedStopSequence != "" {
		required.bits |= CapabilityMatchedStopSequence
	}
	return required
}

func capabilityForRequestContent(content Content) Capability {
	var cache Capability
	if content.Cache != nil {
		cache = CapabilityCacheDirectives
	}
	switch content.Kind {
	case ContentText, ContentRefusal:
		return CapabilityText | cache
	case ContentImage:
		return CapabilityImageInput | cache
	case ContentAudio:
		return CapabilityAudioInput | cache
	case ContentVideo:
		return CapabilityVideoInput | cache
	case ContentFile:
		return CapabilityFileInput | cache
	case ContentToolCall:
		return CapabilityTools | cache
	case ContentToolResult:
		capabilities := CapabilityTools | cache
		if content.ToolResult != nil {
			for _, nested := range content.ToolResult.Content {
				capabilities |= capabilityForRequestContent(nested)
			}
		}
		return capabilities
	case ContentReasoning:
		capabilities := CapabilityText | CapabilityReasoning | cache
		if content.Signature != "" {
			capabilities |= CapabilityReasoningSignature
		}
		return capabilities
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
		capabilities := CapabilityText | CapabilityReasoning
		if content.Signature != "" {
			capabilities |= CapabilityReasoningSignature
		}
		return capabilities
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
		"reasoning_accounting":  CapabilityReasoningAccounting,
		"authoritative_usage":   CapabilityAuthoritativeUsage,
		"multiple_candidates":   CapabilityMultipleCandidates,
		"cache_directives":      CapabilityCacheDirectives,
		"reasoning_disable":     CapabilityReasoningDisable,
		"reasoning_effort":      CapabilityReasoningEffort,
		"reasoning_budget":      CapabilityReasoningBudget,
		"sampling_top_k":        CapabilitySamplingTopK,
		"sampling_seed":         CapabilitySamplingSeed,
		"sampling_penalties":    CapabilitySamplingPenalties,
		"stop_sequences":        CapabilityStopSequences,
		"request_metadata":      CapabilityRequestMetadata,
		"request_storage":       CapabilityRequestStorage,
		"automatic_storage":     CapabilityAutomaticStorage,
		"conversation_state":    CapabilityConversationState,
		"reasoning_adaptive":    CapabilityReasoningAdaptive,
		"reasoning_signature":   CapabilityReasoningSignature,
		"reasoning_display":     CapabilityReasoningDisplay,
		"matched_stop_sequence": CapabilityMatchedStopSequence,
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
