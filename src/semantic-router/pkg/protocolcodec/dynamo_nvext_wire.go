package protocolcodec

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type dynamoRequestNVExtWire struct {
	GreedSampling      *bool                         `json:"greed_sampling,omitempty"`
	UseRawPrompt       *bool                         `json:"use_raw_prompt,omitempty"`
	Annotations        []string                      `json:"annotations,omitempty"`
	BackendInstanceID  *uint64                       `json:"backend_instance_id,omitempty"`
	TokenData          []uint32                      `json:"token_data,omitempty"`
	MaxThinkingTokens  *uint32                       `json:"max_thinking_tokens,omitempty"`
	CacheSalt          string                        `json:"cache_salt,omitempty"`
	ExtraFields        []string                      `json:"extra_fields,omitempty"`
	MetadataUpload     *dynamoMetadataUploadWire     `json:"metadata_upload,omitempty"`
	PrefillWorkerID    *uint64                       `json:"prefill_worker_id,omitempty"`
	DecodeWorkerID     *uint64                       `json:"decode_worker_id,omitempty"`
	DPRank             *uint32                       `json:"dp_rank,omitempty"`
	PrefillDPRank      *uint32                       `json:"prefill_dp_rank,omitempty"`
	AgentHints         *dynamoAgentHintsWire         `json:"agent_hints,omitempty"`
	RequestTimestampMS *float64                      `json:"request_timestamp_ms,omitempty"`
	RoutingConstraints *dynamoRoutingConstraintsWire `json:"routing_constraints,omitempty"`
	Router             *dynamoRouterParamsWire       `json:"router,omitempty"`
}

type dynamoMetadataUploadWire struct {
	URL string `json:"url"`
}

type dynamoAgentHintsWire struct {
	Priority           *int32   `json:"priority,omitempty"`
	StrictPriority     *uint32  `json:"strict_priority,omitempty"`
	OSL                *uint32  `json:"osl,omitempty"`
	SpeculativePrefill *bool    `json:"speculative_prefill,omitempty"`
	LatencySensitivity *float64 `json:"latency_sensitivity,omitempty"`
}

type dynamoRoutingConstraintsWire struct {
	RequiredTaints  []string           `json:"required_taints,omitempty"`
	PreferredTaints map[string]float32 `json:"preferred_taints,omitempty"`
}

type dynamoRouterParamsWire struct {
	TTFTTarget *float64 `json:"ttft_target,omitempty"`
	ITLTarget  *float64 `json:"itl_target,omitempty"`
}

type dynamoResponseNVExtWire struct {
	WorkerID           *dynamoWorkerInfoWire                     `json:"worker_id,omitempty"`
	Timing             *dynamoTimingInfoWire                     `json:"timing,omitempty"`
	RoutedExperts      json.RawMessage                           `json:"routed_experts,omitempty"`
	EngineData         json.RawMessage                           `json:"engine_data,omitempty"`
	StopReason         json.RawMessage                           `json:"stop_reason,omitempty"`
	PromptTokenIDs     []uint32                                  `json:"prompt_token_ids,omitempty"`
	CompletionTokenIDs []uint32                                  `json:"completion_token_ids,omitempty"`
	PromptLogprobs     []map[uint32]dynamoPromptLogprobEntryWire `json:"prompt_logprobs,omitempty"`
	TokenIDs           []uint32                                  `json:"token_ids,omitempty"`
}

type dynamoTimingInfoWire struct {
	RequestReceivedMS            *uint64  `json:"request_received_ms"`
	PrefillWaitTimeMS            *float64 `json:"prefill_wait_time_ms,omitempty"`
	PrefillTimeMS                *float64 `json:"prefill_time_ms,omitempty"`
	TTFTMS                       *float64 `json:"ttft_ms,omitempty"`
	TotalTimeMS                  *float64 `json:"total_time_ms,omitempty"`
	KVHitRate                    *float64 `json:"kv_hit_rate,omitempty"`
	RouterQueueDepth             *uint64  `json:"router_queue_depth,omitempty"`
	KVTransferEstimatedLatencyMS *float64 `json:"kv_transfer_estimated_latency_ms,omitempty"`
}

type dynamoPromptLogprobEntryWire struct {
	Logprob      *float32 `json:"logprob"`
	Rank         *uint32  `json:"rank,omitempty"`
	DecodedToken *string  `json:"decoded_token,omitempty"`
}

type dynamoWorkerInfoWire struct {
	PrefillWorkerID *uint64 `json:"prefill_worker_id,omitempty"`
	PrefillDPRank   *uint32 `json:"prefill_dp_rank,omitempty"`
	DecodeWorkerID  *uint64 `json:"decode_worker_id,omitempty"`
	DecodeDPRank    *uint32 `json:"decode_dp_rank,omitempty"`
}

func decodeDynamoRequestNVExt(raw json.RawMessage, policy llmprotocol.Policy) (*llmprotocol.DynamoRequestNVExt, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	if policy.Limits.DynamoNVExtBytes > 0 && len(raw) > policy.Limits.DynamoNVExtBytes {
		return nil, llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"dynamo_nvext_size_limit",
			"Dynamo nvext request exceeds the configured limit",
			nil,
		)
	}
	var wire dynamoRequestNVExtWire
	if err := decodeWire(raw, &wire, policy); err != nil {
		return nil, err
	}
	extension := decodeDynamoRequestNVExtWire(wire)
	if err := llmprotocol.ValidateDynamoRequestNVExt(extension, policy.Limits); err != nil {
		return nil, err
	}
	return extension, nil
}

func decodeDynamoRequestNVExtWire(wire dynamoRequestNVExtWire) *llmprotocol.DynamoRequestNVExt {
	extension := &llmprotocol.DynamoRequestNVExt{
		GreedSampling: wire.GreedSampling, UseRawPrompt: wire.UseRawPrompt,
		Annotations:       append([]string(nil), wire.Annotations...),
		BackendInstanceID: wire.BackendInstanceID,
		TokenData:         append([]uint32(nil), wire.TokenData...),
		MaxThinkingTokens: wire.MaxThinkingTokens, CacheSalt: wire.CacheSalt,
		ExtraFields:     append([]string(nil), wire.ExtraFields...),
		PrefillWorkerID: wire.PrefillWorkerID, DecodeWorkerID: wire.DecodeWorkerID,
		DPRank: wire.DPRank, PrefillDPRank: wire.PrefillDPRank,
		RequestTimestampMS: wire.RequestTimestampMS,
	}
	if wire.MetadataUpload != nil {
		extension.MetadataUpload = &llmprotocol.DynamoMetadataUpload{URL: wire.MetadataUpload.URL}
	}
	if wire.AgentHints != nil {
		extension.AgentHints = &llmprotocol.DynamoAgentHints{
			Priority: wire.AgentHints.Priority, StrictPriority: wire.AgentHints.StrictPriority,
			OSL: wire.AgentHints.OSL, SpeculativePrefill: wire.AgentHints.SpeculativePrefill,
			LatencySensitivity: wire.AgentHints.LatencySensitivity,
		}
	}
	if wire.RoutingConstraints != nil {
		extension.RoutingConstraints = &llmprotocol.DynamoRoutingConstraints{
			RequiredTaints:  append([]string(nil), wire.RoutingConstraints.RequiredTaints...),
			PreferredTaints: cloneDynamoPreferredTaints(wire.RoutingConstraints.PreferredTaints),
		}
	}
	if wire.Router != nil {
		extension.Router = &llmprotocol.DynamoRouterParams{TTFTTarget: wire.Router.TTFTTarget, ITLTarget: wire.Router.ITLTarget}
	}
	return extension
}

func encodeDynamoRequestNVExt(extension *llmprotocol.DynamoRequestNVExt, policy llmprotocol.Policy) (json.RawMessage, error) {
	if err := llmprotocol.ValidateDynamoRequestNVExt(extension, policy.Limits); err != nil {
		return nil, err
	}
	wire := dynamoRequestNVExtWire{
		GreedSampling: extension.GreedSampling, UseRawPrompt: extension.UseRawPrompt,
		Annotations:       append([]string(nil), extension.Annotations...),
		BackendInstanceID: extension.BackendInstanceID,
		TokenData:         append([]uint32(nil), extension.TokenData...),
		MaxThinkingTokens: extension.MaxThinkingTokens, CacheSalt: extension.CacheSalt,
		ExtraFields:     append([]string(nil), extension.ExtraFields...),
		PrefillWorkerID: extension.PrefillWorkerID, DecodeWorkerID: extension.DecodeWorkerID,
		DPRank: extension.DPRank, PrefillDPRank: extension.PrefillDPRank,
		RequestTimestampMS: extension.RequestTimestampMS,
	}
	if extension.MetadataUpload != nil {
		wire.MetadataUpload = &dynamoMetadataUploadWire{URL: extension.MetadataUpload.URL}
	}
	if extension.AgentHints != nil {
		wire.AgentHints = &dynamoAgentHintsWire{
			Priority: extension.AgentHints.Priority, StrictPriority: extension.AgentHints.StrictPriority,
			OSL: extension.AgentHints.OSL, SpeculativePrefill: extension.AgentHints.SpeculativePrefill,
			LatencySensitivity: extension.AgentHints.LatencySensitivity,
		}
	}
	if extension.RoutingConstraints != nil {
		wire.RoutingConstraints = &dynamoRoutingConstraintsWire{
			RequiredTaints:  append([]string(nil), extension.RoutingConstraints.RequiredTaints...),
			PreferredTaints: cloneDynamoPreferredTaints(extension.RoutingConstraints.PreferredTaints),
		}
	}
	if extension.Router != nil {
		wire.Router = &dynamoRouterParamsWire{TTFTTarget: extension.Router.TTFTTarget, ITLTarget: extension.Router.ITLTarget}
	}
	body, err := marshalWire(wire)
	if err != nil {
		return nil, err
	}
	if policy.Limits.DynamoNVExtBytes > 0 && len(body) > policy.Limits.DynamoNVExtBytes {
		return nil, llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"dynamo_nvext_size_limit",
			"Dynamo nvext request exceeds the configured limit",
			nil,
		)
	}
	return json.RawMessage(body), nil
}

func validateDynamoRequestEnvelope(envelope llmprotocol.Envelope, target llmprotocol.WireFormat, policy llmprotocol.Policy) error {
	if envelope.Dynamo == nil ||
		(envelope.Dynamo.RequestNVExt == nil && envelope.Dynamo.RequestTopLevelCacheSalt == nil) {
		return nil
	}
	if envelope.Format != llmprotocol.OpenAIChatV1 || target != llmprotocol.OpenAIChatV1 {
		return llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_dynamo_nvext_translation",
			"Dynamo nvext requests cannot be translated across wire formats",
			nil,
		)
	}
	if envelope.Dynamo.RequestTopLevelCacheSalt != nil && policy.Limits.DynamoNVExtStringBytes > 0 &&
		len(*envelope.Dynamo.RequestTopLevelCacheSalt) > policy.Limits.DynamoNVExtStringBytes {
		return llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest, "dynamo_nvext_string_limit",
			"Dynamo top-level cache_salt exceeds the configured limit", nil,
		)
	}
	return llmprotocol.ValidateDynamoRequestNVExt(envelope.Dynamo.RequestNVExt, policy.Limits)
}

func decodeDynamoResponseNVExt(raw json.RawMessage, policy llmprotocol.Policy) (*llmprotocol.DynamoResponseNVExt, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	if policy.Limits.DynamoNVExtBytes > 0 && len(raw) > policy.Limits.DynamoNVExtBytes {
		return nil, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"dynamo_nvext_size_limit",
			"upstream Dynamo nvext exceeds the configured limit",
			nil,
		)
	}
	var wire dynamoResponseNVExtWire
	if err := decodeProviderWire(raw, &wire, policy); err != nil {
		return nil, err
	}
	extension := &llmprotocol.DynamoResponseNVExt{
		RoutedExperts:      append(json.RawMessage(nil), wire.RoutedExperts...),
		EngineData:         append(json.RawMessage(nil), wire.EngineData...),
		StopReason:         append(json.RawMessage(nil), wire.StopReason...),
		PromptTokenIDs:     append([]uint32(nil), wire.PromptTokenIDs...),
		CompletionTokenIDs: append([]uint32(nil), wire.CompletionTokenIDs...),
		TokenIDs:           append([]uint32(nil), wire.TokenIDs...),
	}
	if wire.Timing != nil {
		if wire.Timing.RequestReceivedMS == nil {
			return nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_dynamo_nvext", "upstream Dynamo timing is missing request_received_ms", nil)
		}
		extension.Timing = decodeDynamoTimingInfoWire(*wire.Timing)
	}
	for _, position := range wire.PromptLogprobs {
		for _, entry := range position {
			if entry.Logprob == nil {
				return nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_dynamo_nvext", "upstream Dynamo prompt logprob entry is missing logprob", nil)
			}
		}
	}
	extension.PromptLogprobs = decodeDynamoPromptLogprobsWire(wire.PromptLogprobs)
	if wire.WorkerID != nil {
		extension.WorkerID = &llmprotocol.DynamoWorkerInfo{
			PrefillWorkerID: wire.WorkerID.PrefillWorkerID,
			PrefillDPRank:   wire.WorkerID.PrefillDPRank,
			DecodeWorkerID:  wire.WorkerID.DecodeWorkerID,
			DecodeDPRank:    wire.WorkerID.DecodeDPRank,
		}
	}
	if err := llmprotocol.ValidateDynamoResponseNVExt(extension, policy.Limits); err != nil {
		return nil, err
	}
	return extension, nil
}

func encodeDynamoResponseNVExt(extension *llmprotocol.DynamoResponseNVExt, policy llmprotocol.Policy) (json.RawMessage, error) {
	if err := llmprotocol.ValidateDynamoResponseNVExt(extension, policy.Limits); err != nil {
		return nil, err
	}
	wire := dynamoResponseNVExtWire{
		RoutedExperts:      append(json.RawMessage(nil), extension.RoutedExperts...),
		EngineData:         append(json.RawMessage(nil), extension.EngineData...),
		StopReason:         append(json.RawMessage(nil), extension.StopReason...),
		PromptTokenIDs:     append([]uint32(nil), extension.PromptTokenIDs...),
		CompletionTokenIDs: append([]uint32(nil), extension.CompletionTokenIDs...),
		TokenIDs:           append([]uint32(nil), extension.TokenIDs...),
	}
	if extension.Timing != nil {
		wire.Timing = encodeDynamoTimingInfoWire(extension.Timing)
	}
	wire.PromptLogprobs = encodeDynamoPromptLogprobsWire(extension.PromptLogprobs)
	if extension.WorkerID != nil {
		wire.WorkerID = &dynamoWorkerInfoWire{
			PrefillWorkerID: extension.WorkerID.PrefillWorkerID,
			PrefillDPRank:   extension.WorkerID.PrefillDPRank,
			DecodeWorkerID:  extension.WorkerID.DecodeWorkerID,
			DecodeDPRank:    extension.WorkerID.DecodeDPRank,
		}
	}
	body, err := marshalWire(wire)
	if err != nil {
		return nil, err
	}
	if policy.Limits.DynamoNVExtBytes > 0 && len(body) > policy.Limits.DynamoNVExtBytes {
		return nil, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"dynamo_nvext_size_limit",
			"upstream Dynamo nvext exceeds the configured limit",
			nil,
		)
	}
	return json.RawMessage(body), nil
}

func cloneDynamoPreferredTaints(source map[string]float32) map[string]float32 {
	if source == nil {
		return nil
	}
	result := make(map[string]float32, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}

func decodeDynamoTimingInfoWire(wire dynamoTimingInfoWire) *llmprotocol.DynamoTimingInfo {
	return &llmprotocol.DynamoTimingInfo{
		RequestReceivedMS: *wire.RequestReceivedMS, PrefillWaitTimeMS: wire.PrefillWaitTimeMS,
		PrefillTimeMS: wire.PrefillTimeMS, TTFTMS: wire.TTFTMS, TotalTimeMS: wire.TotalTimeMS,
		KVHitRate: wire.KVHitRate, RouterQueueDepth: wire.RouterQueueDepth,
		KVTransferEstimatedLatencyMS: wire.KVTransferEstimatedLatencyMS,
	}
}

func encodeDynamoTimingInfoWire(timing *llmprotocol.DynamoTimingInfo) *dynamoTimingInfoWire {
	requestReceived := timing.RequestReceivedMS
	return &dynamoTimingInfoWire{
		RequestReceivedMS: &requestReceived, PrefillWaitTimeMS: timing.PrefillWaitTimeMS,
		PrefillTimeMS: timing.PrefillTimeMS, TTFTMS: timing.TTFTMS, TotalTimeMS: timing.TotalTimeMS,
		KVHitRate: timing.KVHitRate, RouterQueueDepth: timing.RouterQueueDepth,
		KVTransferEstimatedLatencyMS: timing.KVTransferEstimatedLatencyMS,
	}
}

func decodeDynamoPromptLogprobsWire(source []map[uint32]dynamoPromptLogprobEntryWire) []map[uint32]llmprotocol.DynamoPromptLogprobEntry {
	if source == nil {
		return nil
	}
	result := make([]map[uint32]llmprotocol.DynamoPromptLogprobEntry, len(source))
	for index, position := range source {
		if position == nil {
			continue
		}
		result[index] = make(map[uint32]llmprotocol.DynamoPromptLogprobEntry, len(position))
		for tokenID, entry := range position {
			result[index][tokenID] = llmprotocol.DynamoPromptLogprobEntry{Logprob: *entry.Logprob, Rank: entry.Rank, DecodedToken: entry.DecodedToken}
		}
	}
	return result
}

func encodeDynamoPromptLogprobsWire(source []map[uint32]llmprotocol.DynamoPromptLogprobEntry) []map[uint32]dynamoPromptLogprobEntryWire {
	if source == nil {
		return nil
	}
	result := make([]map[uint32]dynamoPromptLogprobEntryWire, len(source))
	for index, position := range source {
		if position == nil {
			continue
		}
		result[index] = make(map[uint32]dynamoPromptLogprobEntryWire, len(position))
		for tokenID, entry := range position {
			logprob := entry.Logprob
			result[index][tokenID] = dynamoPromptLogprobEntryWire{Logprob: &logprob, Rank: entry.Rank, DecodedToken: entry.DecodedToken}
		}
	}
	return result
}

func validateDynamoResponseEnvelope(envelope llmprotocol.Envelope, target llmprotocol.WireFormat, policy llmprotocol.Policy) error {
	if envelope.Dynamo == nil || envelope.Dynamo.ResponseNVExt == nil {
		return nil
	}
	if envelope.Format != llmprotocol.OpenAIChatV1 || target != llmprotocol.OpenAIChatV1 {
		return llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_dynamo_nvext_translation",
			"Dynamo nvext responses cannot be translated across wire formats",
			nil,
		)
	}
	return llmprotocol.ValidateDynamoResponseNVExt(envelope.Dynamo.ResponseNVExt, policy.Limits)
}
