package llmprotocol

import (
	"encoding/json"
	"fmt"
	"strings"
)

// DynamoEnvelope carries structured NVIDIA Dynamo wire extensions across a
// decode-mutate-encode lifecycle. RequestNVExt and ResponseNVExt are populated
// by their respective codec directions and are not neutral model semantics.
type DynamoEnvelope struct {
	RequestNVExt             *DynamoRequestNVExt
	RequestTopLevelCacheSalt *string
	ResponseNVExt            *DynamoResponseNVExt
}

// DynamoRequestNVExt models the documented nvext request object accepted by
// NVIDIA Dynamo. Values in this object are untrusted scheduling and routing
// hints; in particular, CacheSalt is not authenticated tenant identity.
type DynamoRequestNVExt struct {
	GreedSampling      *bool
	UseRawPrompt       *bool
	Annotations        []string
	BackendInstanceID  *uint64
	TokenData          []uint32
	MaxThinkingTokens  *uint32
	CacheSalt          string
	ExtraFields        []string
	MetadataUpload     *DynamoMetadataUpload
	PrefillWorkerID    *uint64
	DecodeWorkerID     *uint64
	DPRank             *uint32
	PrefillDPRank      *uint32
	AgentHints         *DynamoAgentHints
	RequestTimestampMS *float64
	RoutingConstraints *DynamoRoutingConstraints
	Router             *DynamoRouterParams
}

type DynamoMetadataUpload struct {
	URL string
}

type DynamoAgentHints struct {
	Priority           *int32
	StrictPriority     *uint32
	OSL                *uint32
	SpeculativePrefill *bool
	LatencySensitivity *float64
}

type DynamoRoutingConstraints struct {
	RequiredTaints  []string
	PreferredTaints map[string]float32
}

type DynamoRouterParams struct {
	TTFTTarget *float64
	ITLTarget  *float64
}

// DynamoResponseNVExt models the bounded nvext metadata Dynamo can attach to
// buffered responses and streaming chunks. Evolving backend-owned payloads
// remain raw JSON but are validated for size, syntax, duplicates, and depth.
type DynamoResponseNVExt struct {
	WorkerID           *DynamoWorkerInfo
	Timing             *DynamoTimingInfo
	RoutedExperts      json.RawMessage
	EngineData         json.RawMessage
	StopReason         json.RawMessage
	PromptTokenIDs     []uint32
	CompletionTokenIDs []uint32
	PromptLogprobs     []map[uint32]DynamoPromptLogprobEntry
	TokenIDs           []uint32
}

type DynamoTimingInfo struct {
	RequestReceivedMS            uint64
	PrefillWaitTimeMS            *float64
	PrefillTimeMS                *float64
	TTFTMS                       *float64
	TotalTimeMS                  *float64
	KVHitRate                    *float64
	RouterQueueDepth             *uint64
	KVTransferEstimatedLatencyMS *float64
}

type DynamoPromptLogprobEntry struct {
	Logprob      float32
	Rank         *uint32
	DecodedToken *string
}

type DynamoWorkerInfo struct {
	PrefillWorkerID *uint64
	PrefillDPRank   *uint32
	DecodeWorkerID  *uint64
	DecodeDPRank    *uint32
}

var supportedDynamoExtraFields = map[string]struct{}{
	"worker_id": {}, "timing": {}, "routed_experts": {}, "engine_data": {},
	"stop_reason": {}, "prompt_token_ids": {}, "completion_token_ids": {}, "prompt_logprobs": {},
}

// ValidateDynamoRequestNVExt enforces the bounded request-side nvext contract
// before a codec stores it in or reads it from an Envelope.
func ValidateDynamoRequestNVExt(extension *DynamoRequestNVExt, limits Limits) error {
	if extension == nil {
		return nil
	}
	if err := validateDynamoRequestCounts(extension, limits); err != nil {
		return err
	}
	if err := validateDynamoRequestNumbers(extension); err != nil {
		return err
	}
	if exceedsDynamoString(extension.CacheSalt, limits) {
		return NewError(ErrorInvalidRequest, "dynamo_nvext_string_limit", "Dynamo nvext request string exceeds the configured limit", nil)
	}
	bytes := len(extension.CacheSalt)
	annotationBytes, err := validateDynamoAnnotations(extension.Annotations, limits)
	if err != nil {
		return err
	}
	bytes += annotationBytes
	extraFieldBytes, err := validateDynamoExtraFields(extension.ExtraFields)
	if err != nil {
		return err
	}
	bytes += extraFieldBytes
	metadataBytes, err := validateDynamoMetadataUpload(extension.MetadataUpload, limits)
	if err != nil {
		return err
	}
	bytes += metadataBytes
	routingBytes, err := validateDynamoRoutingConstraints(extension.RoutingConstraints, limits)
	if err != nil {
		return err
	}
	bytes += routingBytes + len(extension.TokenData)*4
	if limits.DynamoNVExtBytes > 0 && bytes > limits.DynamoNVExtBytes {
		return NewError(ErrorInvalidRequest, "dynamo_nvext_size_limit", "Dynamo nvext request exceeds the configured limit", nil)
	}
	return nil
}

func validateDynamoRequestCounts(extension *DynamoRequestNVExt, limits Limits) error {
	tooManyRoutingItems := extension.RoutingConstraints != nil &&
		(exceedsDynamoItems(len(extension.RoutingConstraints.RequiredTaints), limits) || exceedsDynamoItems(len(extension.RoutingConstraints.PreferredTaints), limits))
	if exceedsDynamoItems(len(extension.Annotations), limits) || exceedsDynamoItems(len(extension.ExtraFields), limits) || tooManyRoutingItems {
		return NewError(ErrorInvalidRequest, "dynamo_nvext_items_limit", "Dynamo nvext request item limit exceeded", nil)
	}
	if exceedsDynamoTokenIDs(len(extension.TokenData), limits) {
		return NewError(ErrorInvalidRequest, "dynamo_nvext_token_limit", "Dynamo nvext request token ID limit exceeded", nil)
	}
	return nil
}

func validateDynamoRequestNumbers(extension *DynamoRequestNVExt) error {
	invalidAgentHint := extension.AgentHints != nil && !validOptionalDynamoFloat(extension.AgentHints.LatencySensitivity)
	invalidRouter := extension.Router != nil && (!validOptionalDynamoFloat(extension.Router.TTFTTarget) || !validOptionalDynamoFloat(extension.Router.ITLTarget))
	if !validOptionalDynamoFloat(extension.RequestTimestampMS) || invalidAgentHint || invalidRouter {
		return NewError(ErrorInvalidRequest, "invalid_dynamo_nvext_number", "Dynamo nvext numeric value must be finite", nil)
	}
	return nil
}

func validateDynamoAnnotations(annotations []string, limits Limits) (int, error) {
	bytes := 0
	for _, annotation := range annotations {
		if exceedsDynamoString(annotation, limits) {
			return 0, NewError(ErrorInvalidRequest, "dynamo_nvext_string_limit", "Dynamo nvext annotation exceeds the configured limit", nil)
		}
		bytes += len(annotation)
	}
	return bytes, nil
}

func validateDynamoExtraFields(fields []string) (int, error) {
	bytes := 0
	seen := make(map[string]struct{}, len(fields))
	for _, field := range fields {
		if _, supported := supportedDynamoExtraFields[field]; !supported {
			return 0, NewError(ErrorInvalidRequest, "unsupported_dynamo_nvext_extra_field", fmt.Sprintf("Dynamo nvext response field %q is unsupported", field), nil)
		}
		if _, duplicate := seen[field]; duplicate {
			return 0, NewError(ErrorInvalidRequest, "duplicate_dynamo_nvext_extra_field", fmt.Sprintf("Dynamo nvext response field %q is duplicated", field), nil)
		}
		seen[field] = struct{}{}
		bytes += len(field)
	}
	return bytes, nil
}

func validateDynamoMetadataUpload(upload *DynamoMetadataUpload, limits Limits) (int, error) {
	if upload == nil {
		return 0, nil
	}
	if strings.TrimSpace(upload.URL) == "" {
		return 0, NewError(ErrorInvalidRequest, "invalid_dynamo_nvext_metadata_upload", "Dynamo nvext metadata upload URL is required", nil)
	}
	if exceedsDynamoString(upload.URL, limits) {
		return 0, NewError(ErrorInvalidRequest, "dynamo_nvext_string_limit", "Dynamo nvext metadata upload URL exceeds the configured limit", nil)
	}
	return len(upload.URL), nil
}

func validateDynamoRoutingConstraints(constraints *DynamoRoutingConstraints, limits Limits) (int, error) {
	if constraints == nil {
		return 0, nil
	}
	bytes := 0
	for _, taint := range constraints.RequiredTaints {
		if exceedsDynamoString(taint, limits) {
			return 0, NewError(ErrorInvalidRequest, "dynamo_nvext_string_limit", "Dynamo nvext taint exceeds the configured limit", nil)
		}
		bytes += len(taint)
	}
	for taint, preference := range constraints.PreferredTaints {
		if exceedsDynamoString(taint, limits) || !finiteFloat(float64(preference)) {
			return 0, NewError(ErrorInvalidRequest, "invalid_dynamo_nvext_routing_constraint", "Dynamo nvext preferred taint is invalid", nil)
		}
		bytes += len(taint) + 4
	}
	return bytes, nil
}

// ValidateDynamoResponseNVExt enforces the bounded response-side nvext
// contract before a codec stores it in or reads it from an Envelope.
func ValidateDynamoResponseNVExt(extension *DynamoResponseNVExt, limits Limits) error {
	if extension == nil {
		return nil
	}
	if exceedsDynamoTokenIDs(len(extension.PromptTokenIDs), limits) ||
		exceedsDynamoTokenIDs(len(extension.CompletionTokenIDs), limits) ||
		exceedsDynamoTokenIDs(len(extension.TokenIDs), limits) {
		return NewError(ErrorUpstreamUnavailable, "dynamo_nvext_token_limit", "upstream Dynamo nvext token ID limit exceeded", nil)
	}
	totalBytes := 4 * (len(extension.PromptTokenIDs) + len(extension.CompletionTokenIDs) + len(extension.TokenIDs))
	if err := validateDynamoTiming(extension.Timing); err != nil {
		return err
	}
	if err := validateDynamoPromptLogprobs(extension.PromptLogprobs, limits); err != nil {
		return err
	}
	fields := []struct {
		name string
		raw  json.RawMessage
	}{
		{"routed_experts", extension.RoutedExperts},
		{"engine_data", extension.EngineData}, {"stop_reason", extension.StopReason},
	}
	for _, field := range fields {
		if len(field.raw) == 0 {
			continue
		}
		if err := validateDynamoRawJSON(field.raw, limits.JSONDepth); err != nil {
			return NewError(ErrorUpstreamUnavailable, "invalid_dynamo_nvext", fmt.Sprintf("upstream Dynamo nvext %s is invalid", field.name), err)
		}
		totalBytes += len(field.raw)
	}
	if limits.DynamoNVExtBytes > 0 && totalBytes > limits.DynamoNVExtBytes {
		return NewError(ErrorUpstreamUnavailable, "dynamo_nvext_size_limit", "upstream Dynamo nvext exceeds the configured limit", nil)
	}
	return nil
}

func validateDynamoPromptLogprobs(positions []map[uint32]DynamoPromptLogprobEntry, limits Limits) error {
	if exceedsDynamoItems(len(positions), limits) {
		return NewError(ErrorUpstreamUnavailable, "dynamo_nvext_items_limit", "upstream Dynamo prompt logprobs position limit exceeded", nil)
	}
	for _, position := range positions {
		if exceedsDynamoItems(len(position), limits) {
			return NewError(ErrorUpstreamUnavailable, "dynamo_nvext_items_limit", "upstream Dynamo prompt logprobs item limit exceeded", nil)
		}
		for _, entry := range position {
			if !finiteFloat(float64(entry.Logprob)) || (entry.DecodedToken != nil && exceedsDynamoString(*entry.DecodedToken, limits)) {
				return NewError(ErrorUpstreamUnavailable, "invalid_dynamo_nvext", "upstream Dynamo prompt logprobs are invalid", nil)
			}
		}
	}
	return nil
}

func validateDynamoTiming(timing *DynamoTimingInfo) error {
	if timing == nil {
		return nil
	}
	values := []*float64{timing.PrefillWaitTimeMS, timing.PrefillTimeMS, timing.TTFTMS, timing.TotalTimeMS, timing.KVHitRate, timing.KVTransferEstimatedLatencyMS}
	for _, value := range values {
		if !validOptionalDynamoFloat(value) {
			return NewError(ErrorUpstreamUnavailable, "invalid_dynamo_nvext", "upstream Dynamo timing value must be finite", nil)
		}
	}
	return nil
}

func validOptionalDynamoFloat(value *float64) bool {
	return value == nil || finiteFloat(*value)
}

func validateDynamoRawJSON(raw json.RawMessage, maximumDepth int) error {
	wrapped := make([]byte, 0, len(raw)+11)
	wrapped = append(wrapped, `{"value":`...)
	wrapped = append(wrapped, raw...)
	wrapped = append(wrapped, '}')
	if maximumDepth > 0 {
		maximumDepth++ // The synthetic wrapper must not reduce the nvext depth allowance.
	}
	return ValidateJSONObject(wrapped, maximumDepth)
}

func exceedsDynamoItems(count int, limits Limits) bool {
	return limits.DynamoNVExtItems > 0 && count > limits.DynamoNVExtItems
}

func exceedsDynamoTokenIDs(count int, limits Limits) bool {
	return limits.DynamoNVExtTokenIDs > 0 && count > limits.DynamoNVExtTokenIDs
}

func exceedsDynamoString(value string, limits Limits) bool {
	return limits.DynamoNVExtStringBytes > 0 && len(value) > limits.DynamoNVExtStringBytes
}
