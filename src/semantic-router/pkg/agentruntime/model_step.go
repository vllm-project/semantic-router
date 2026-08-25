package agentruntime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const (
	maximumToolArgumentsBytes  = 1 << 20
	maximumModelStepTextBytes  = 1 << 20
	maximumAssistantEventBytes = 60 << 10
)

type streamedToolCall struct {
	itemIndex    int
	invocationID string
	name         string
	arguments    strings.Builder
}

// modelStepCollector is the only boundary that projects provider-neutral
// inference events into the durable Agent transcript. It never persists
// reasoning or provider errors, and it assigns Router-owned invocation IDs so
// a provider cannot choose an idempotency identity.
type modelStepCollector struct {
	ctx         context.Context
	worker      *Worker
	lease       agentmanagement.TurnLease
	registry    *agentmanagement.ToolRegistry
	policy      agentmanagement.ToolPolicy
	modelStepID string
	step        int

	started     bool
	terminal    bool
	stop        llmprotocol.StopReason
	failed      error
	toolCalls   map[int]*streamedToolCall
	text        strings.Builder
	liveOrdinal int
	observation PublicInferenceObservation
	observed    bool
	usage       *agentmanagement.ModelStepUsage
}

type modelStepOutput struct {
	Events     []agentmanagement.EventAppend
	StopReason llmprotocol.StopReason
}

func newModelStepCollector(
	ctx context.Context,
	worker *Worker,
	lease agentmanagement.TurnLease,
	registry *agentmanagement.ToolRegistry,
	policy agentmanagement.ToolPolicy,
	modelStepID string,
	completedToolSteps int,
) *modelStepCollector {
	return &modelStepCollector{
		ctx: ctx, worker: worker, lease: lease, registry: registry, policy: policy,
		modelStepID: modelStepID, step: completedToolSteps + 1,
		toolCalls: make(map[int]*streamedToolCall),
	}
}

func (collector *modelStepCollector) consume(event llmprotocol.Event) error {
	if collector.terminal {
		return fmt.Errorf("%w: inference emitted data after its terminal event", agentmanagement.ErrConflict)
	}
	if event.Usage != nil {
		collector.captureAuthoritativeUsage(event.Usage)
	}
	switch event.Type {
	case llmprotocol.EventResponseStarted:
		if collector.started {
			return fmt.Errorf("%w: inference response started twice", agentmanagement.ErrConflict)
		}
		collector.started = true
	case llmprotocol.EventOutputItemStarted:
		if event.ToolCall != nil {
			call, err := collector.toolCall(event.ItemIndex)
			if err != nil {
				return err
			}
			if err := mergeToolIdentity(call, event.ToolCall); err != nil {
				return err
			}
			if event.ToolCall.Arguments != "" {
				if err := collector.appendToolArguments(call, event.ToolCall.Arguments); err != nil {
					return err
				}
			}
		}
	case llmprotocol.EventOutputTextDelta:
		if event.Delta == "" {
			return nil
		}
		if !collector.started || !utf8.ValidString(event.Delta) {
			return fmt.Errorf("%w: inference emitted an invalid text delta", agentmanagement.ErrConflict)
		}
		if collector.text.Len()+len(event.Delta) > maximumModelStepTextBytes {
			return fmt.Errorf("%w: inference output exceeds its durable bound", agentmanagement.ErrInvalid)
		}
		collector.text.WriteString(event.Delta)
		collector.publishLiveText(event.Delta)
		return nil
	case llmprotocol.EventToolCallDelta:
		if event.ToolCall == nil {
			return fmt.Errorf("%w: inference tool-call delta is empty", agentmanagement.ErrConflict)
		}
		call, err := collector.toolCall(event.ItemIndex)
		if err != nil {
			return err
		}
		if err := mergeToolIdentity(call, event.ToolCall); err != nil {
			return err
		}
		if event.ToolCall.Arguments != "" {
			return collector.appendToolArguments(call, event.ToolCall.Arguments)
		}
	case llmprotocol.EventResponseFailed:
		collector.terminal = true
		collector.stop = llmprotocol.StopError
		collector.failed = errors.New("public inference failed")
	case llmprotocol.EventResponseCompleted:
		collector.terminal = true
		collector.stop = event.StopReason
	case llmprotocol.EventReasoningDelta, llmprotocol.EventUsageUpdated,
		llmprotocol.EventOutputItemCompleted, llmprotocol.EventProviderOpaque:
		// Reasoning and provider-opaque data are deliberately not durable Agent
		// transcript fields. Only authoritative usage is retained through the
		// closed model_step_summary projection.
	default:
		return fmt.Errorf("%w: unsupported inference event %q", agentmanagement.ErrConflict, event.Type)
	}
	return nil
}

func (collector *modelStepCollector) observe(observation PublicInferenceObservation) error {
	if collector.observed || strings.TrimSpace(observation.RequestID) == "" {
		return fmt.Errorf("%w: public inference observation is invalid", agentmanagement.ErrConflict)
	}
	collector.observation = observation
	collector.observed = true
	return nil
}

func (collector *modelStepCollector) captureAuthoritativeUsage(usage *llmprotocol.Usage) {
	if usage == nil || usage.State != llmprotocol.UsageAvailable ||
		!authoritativeTokenCount(usage.InputTotal) ||
		!authoritativeTokenCount(usage.OutputTotal) ||
		!authoritativeTokenCount(usage.Total) {
		return
	}
	input := *usage.InputTotal.Value
	output := *usage.OutputTotal.Value
	total := *usage.Total.Value
	if input < 0 || output < 0 || total < 0 || input > int64(1<<53-1)-output || total != input+output {
		return
	}
	collector.usage = &agentmanagement.ModelStepUsage{
		InputTokens:           input,
		OutputTokens:          output,
		TotalTokens:           total,
		InputUncachedTokens:   authoritativeTokenPointer(usage.InputUncached),
		InputCacheReadTokens:  authoritativeTokenPointer(usage.InputCacheRead),
		InputCacheWriteTokens: authoritativeTokenPointer(usage.InputCacheWrite),
		OutputReasoningTokens: authoritativeTokenPointer(usage.OutputReasoning),
		OutputOtherTokens:     authoritativeTokenPointer(usage.OutputOther),
	}
}

func authoritativeTokenCount(value llmprotocol.TokenCount) bool {
	return value.Value != nil && value.Provenance == llmprotocol.UsageAuthoritative
}

func authoritativeTokenPointer(value llmprotocol.TokenCount) *int64 {
	if !authoritativeTokenCount(value) || *value.Value < 0 || *value.Value > int64(1<<53-1) {
		return nil
	}
	result := *value.Value
	return &result
}

func (collector *modelStepCollector) toolCall(itemIndex int) (*streamedToolCall, error) {
	if itemIndex < 0 || itemIndex > 1024 {
		return nil, fmt.Errorf("%w: inference tool item index is invalid", agentmanagement.ErrConflict)
	}
	if call := collector.toolCalls[itemIndex]; call != nil {
		return call, nil
	}
	if len(collector.toolCalls) >= 64 {
		return nil, fmt.Errorf("%w: inference returned too many tool calls", agentmanagement.ErrConflict)
	}
	call := &streamedToolCall{
		itemIndex:    itemIndex,
		invocationID: deterministicInvocationID(collector.lease.TurnID, collector.step, itemIndex),
	}
	collector.toolCalls[itemIndex] = call
	return call, nil
}

func mergeToolIdentity(target *streamedToolCall, delta *llmprotocol.ToolCall) error {
	if delta.Name != "" {
		if target.name != "" && target.name != delta.Name {
			return fmt.Errorf("%w: inference changed a tool name mid-stream", agentmanagement.ErrConflict)
		}
		target.name = delta.Name
	}
	return nil
}

func (collector *modelStepCollector) appendToolArguments(call *streamedToolCall, fragment string) error {
	if call.arguments.Len()+len(fragment) > maximumToolArgumentsBytes {
		return fmt.Errorf("%w: inference tool arguments exceed their bound", agentmanagement.ErrInvalid)
	}
	call.arguments.WriteString(fragment)
	// Streaming argument fragments can split a credential at arbitrary byte
	// boundaries and therefore cannot be safely redacted. Only the complete,
	// closed-schema ToolRequest is normalized and persisted in finish.
	return nil
}

func (collector *modelStepCollector) finish() (modelStepOutput, error) {
	if collector.failed != nil {
		return modelStepOutput{}, collector.failed
	}
	if !collector.started || !collector.terminal {
		return modelStepOutput{}, fmt.Errorf("%w: inference stream ended without a complete response", agentmanagement.ErrConflict)
	}
	if collector.stop == "" || collector.stop == llmprotocol.StopUnknown {
		return modelStepOutput{}, fmt.Errorf("%w: inference response has no stable stop reason", agentmanagement.ErrConflict)
	}
	if !collector.observed {
		return modelStepOutput{}, fmt.Errorf("%w: inference response has no Router observation", agentmanagement.ErrConflict)
	}
	output := modelStepOutput{StopReason: collector.stop}
	if !utf8.ValidString(collector.text.String()) {
		return modelStepOutput{}, fmt.Errorf("%w: inference returned invalid text", agentmanagement.ErrInvalid)
	}
	for chunkIndex, chunk := range splitAssistantText(collector.text.String()) {
		payload, err := json.Marshal(agentmanagement.AssistantDeltaEvent{
			ModelStepID: collector.modelStepID, ChunkIndex: chunkIndex,
			Delta: agentmanagement.AssistantDelta{
				Kind: agentmanagement.AssistantTextDelta, Text: chunk,
			},
		})
		if err != nil {
			return modelStepOutput{}, err
		}
		output.Events = append(output.Events, collector.workerEvent(agentmanagement.EventAssistantDelta, payload))
	}
	summaryPayload, err := json.Marshal(agentmanagement.ModelStepSummaryEvent{
		ModelStepID:         collector.modelStepID,
		RequestID:           collector.observation.RequestID,
		SelectedRecipe:      collector.observation.SelectedRecipe,
		SelectedDecision:    collector.observation.SelectedDecision,
		SelectedModel:       collector.observation.SelectedModel,
		SelectedAlgorithm:   collector.observation.SelectedAlgorithm,
		ResponsePath:        collector.observation.ResponsePath,
		LatencyMilliseconds: collector.observation.LatencyMilliseconds,
		TTFTMilliseconds:    collector.observation.TTFTMilliseconds,
		Usage:               collector.usage,
	})
	if err != nil {
		return modelStepOutput{}, err
	}
	output.Events = append(
		output.Events,
		collector.workerEvent(agentmanagement.EventModelStepSummary, summaryPayload),
	)
	if len(collector.toolCalls) == 0 {
		if collector.stop == llmprotocol.StopToolCall {
			return modelStepOutput{}, fmt.Errorf("%w: inference ended for a missing tool call", agentmanagement.ErrConflict)
		}
		return output, nil
	}
	if collector.stop != llmprotocol.StopToolCall && collector.stop != llmprotocol.StopUnknown {
		return modelStepOutput{}, fmt.Errorf("%w: inference returned tool calls with an invalid stop reason", agentmanagement.ErrConflict)
	}
	indices := make([]int, 0, len(collector.toolCalls))
	for index := range collector.toolCalls {
		indices = append(indices, index)
	}
	sort.Ints(indices)
	if len(indices) > 1 {
		for _, index := range indices {
			if collector.toolCalls[index].name == "router.publish.prepare" {
				return modelStepOutput{}, fmt.Errorf("%w: publish preparation must be the only tool call in a model step", agentmanagement.ErrConflict)
			}
		}
	}
	for _, index := range indices {
		call := collector.toolCalls[index]
		definition, _, found := collector.registry.Definition(call.name, collector.policy)
		if !found {
			return modelStepOutput{}, agentmanagement.ErrToolUnavailable
		}
		arguments := call.arguments.String()
		if arguments == "" {
			arguments = "{}"
		}
		cleanArguments, finishErr := collector.registry.ScrubInvocationInput(
			collector.ctx,
			collector.lease.RegistryRevision,
			collector.policy,
			call.name,
			json.RawMessage(arguments),
		)
		if finishErr != nil {
			return modelStepOutput{}, finishErr
		}
		var object map[string]json.RawMessage
		if err := json.Unmarshal(cleanArguments, &object); err != nil || object == nil {
			return modelStepOutput{}, fmt.Errorf("%w: inference returned invalid tool arguments", agentmanagement.ErrInvalid)
		}
		payload := agentmanagement.ToolRequestEvent{
			InvocationID: call.invocationID, ToolName: definition.Name,
			Arguments: cleanArguments, Class: definition.Class,
		}
		encoded, finishErr := json.Marshal(payload)
		if finishErr != nil {
			return modelStepOutput{}, finishErr
		}
		output.Events = append(output.Events, collector.workerEvent(agentmanagement.EventToolRequest, encoded))
	}
	return output, nil
}

func (collector *modelStepCollector) publishLiveText(value string) {
	if collector.worker == nil || collector.worker.liveEvents == nil {
		return
	}
	for _, chunk := range splitAssistantText(value) {
		collector.liveOrdinal++
		delta := agentmanagement.AssistantDelta{
			Kind: agentmanagement.AssistantTextDelta, Text: chunk,
		}
		_ = collector.worker.liveEvents.PublishLiveModelStep(
			collector.ctx, collector.lease.NamespaceID, agentmanagement.LiveModelStepEvent{
				SessionID: collector.lease.SessionID, TurnID: collector.lease.TurnID,
				ModelStepID: collector.modelStepID, Phase: agentmanagement.LiveModelStepDelta,
				Ordinal: collector.liveOrdinal, Delta: &delta, CreatedAt: collector.worker.now().UTC(),
			},
		)
	}
}

func (collector *modelStepCollector) publishLiveTerminal(phase agentmanagement.LiveModelStepPhase) {
	if collector.worker == nil || collector.worker.liveEvents == nil {
		return
	}
	_ = collector.worker.liveEvents.PublishLiveModelStep(
		context.WithoutCancel(collector.ctx), collector.lease.NamespaceID,
		agentmanagement.LiveModelStepEvent{
			SessionID: collector.lease.SessionID, TurnID: collector.lease.TurnID,
			ModelStepID: collector.modelStepID, Phase: phase, Ordinal: collector.liveOrdinal,
			CreatedAt: collector.worker.now().UTC(),
		},
	)
}

func (collector *modelStepCollector) workerEvent(
	eventType agentmanagement.EventType, payload json.RawMessage,
) agentmanagement.EventAppend {
	fence := collector.lease.Fence
	return agentmanagement.EventAppend{
		NamespaceID: collector.lease.NamespaceID,
		SessionID:   collector.lease.SessionID,
		TurnID:      collector.lease.TurnID,
		Origin:      "worker",
		Fence:       &fence,
		Type:        eventType,
		Payload:     payload,
	}
}

func splitAssistantText(value string) []string {
	if value == "" {
		return nil
	}
	result := make([]string, 0, (len(value)/maximumAssistantEventBytes)+1)
	for len(value) > maximumAssistantEventBytes {
		end := maximumAssistantEventBytes
		for end > 0 && !utf8.RuneStart(value[end]) {
			end--
		}
		if end == 0 {
			end = maximumAssistantEventBytes
		}
		result = append(result, value[:end])
		value = value[end:]
	}
	if value != "" {
		result = append(result, value)
	}
	return result
}
