package llmprotocol

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"strings"
)

func ValidateResponse(response Response, limits Limits) error {
	if err := validateResponseEnvelope(response, limits); err != nil {
		return err
	}
	if response.Error != nil {
		return validateErrorResponse(response, limits)
	}
	if err := validateSuccessfulResponseEnvelope(response, limits); err != nil {
		return err
	}
	blocks := 0
	outputItems := 0
	itemIDs := make(map[string]struct{})
	for _, sequence := range append([][]OutputItem{response.Output}, response.Alternatives...) {
		if err := validateOutputSequence(sequence, limits, &outputItems, &blocks, itemIDs); err != nil {
			return err
		}
	}
	if limits.ContentBlocks > 0 && blocks > limits.ContentBlocks {
		return NewError(ErrorUpstreamUnavailable, "content_limit", "upstream content block limit exceeded", nil)
	}
	if err := ValidateUsage(response.Usage); err != nil {
		return err
	}
	return nil
}

func validateResponseEnvelope(response Response, limits Limits) error {
	if response.Generation == 0 {
		return NewError(ErrorInternal, "generation_required", "semantic generation is required", nil)
	}
	if exceeds(response.ID, limits.IdentifierBytes) || exceeds(response.ProviderRequestID, limits.IdentifierBytes) ||
		exceeds(response.Model, limits.ModelBytes) || exceeds(response.SourceStopReason, limits.IdentifierBytes) ||
		exceeds(response.MatchedStopSequence, limits.StopBytes) {
		return NewError(ErrorUpstreamUnavailable, "response_field_limit", "upstream response identity or model exceeds the configured limit", nil)
	}
	return nil
}

func validateErrorResponse(response Response, limits Limits) error {
	if len(response.Output) > 0 || len(response.Alternatives) > 0 || response.StopReason != StopError || response.MatchedStopSequence != "" {
		return NewError(ErrorUpstreamUnavailable, "invalid_error_response", "an error response cannot contain output", nil)
	}
	if err := validateProtocolError(response.Error); err != nil {
		return err
	}
	if exceeds(response.Error.Code, limits.IdentifierBytes) || exceeds(response.Error.Parameter, limits.IdentifierBytes) ||
		exceeds(response.Error.Message, limits.TextBytes) {
		return NewError(ErrorUpstreamUnavailable, "response_error_limit", "upstream response error exceeds the configured limit", nil)
	}
	// Failed requests commonly have no token evidence at all. Keep the zero
	// value distinct from an explicit "unknown" usage report, while still
	// validating every provider-supplied count when usage is present.
	if response.Usage.State == "" {
		return nil
	}
	return ValidateUsage(response.Usage)
}

func validateSuccessfulResponseEnvelope(response Response, limits Limits) error {
	if strings.TrimSpace(response.ID) == "" {
		return NewError(ErrorUpstreamUnavailable, "response_id_required", "upstream response ID is required", nil)
	}
	if err := validateResponseStopReason(response); err != nil {
		return err
	}
	return validateResponseOutputCardinality(response, limits)
}

func validateResponseStopReason(response Response) error {
	if !validStopReason(response.StopReason) {
		return NewError(ErrorUpstreamUnavailable, "invalid_stop_reason", "upstream stop reason is invalid", nil)
	}
	if response.StopReason == StopSequence && strings.TrimSpace(response.MatchedStopSequence) == "" {
		return NewError(ErrorUpstreamUnavailable, "matched_stop_sequence_required", "upstream stop_sequence reason requires the matched sequence", nil)
	}
	if response.StopReason != StopSequence && response.MatchedStopSequence != "" {
		return NewError(ErrorUpstreamUnavailable, "matched_stop_sequence_reason", "upstream matched stop sequence requires stop_sequence reason", nil)
	}
	return nil
}

func validateResponseOutputCardinality(response Response, limits Limits) error {
	if len(response.Output) == 0 && response.StopReason != StopContentFilter {
		return NewError(ErrorUpstreamUnavailable, "empty_response_output", "successful upstream response has no primary output", nil)
	}
	if limits.Alternatives > 0 && len(response.Alternatives) > limits.Alternatives {
		return NewError(ErrorUpstreamUnavailable, "alternatives_limit", "upstream response alternative limit exceeded", nil)
	}
	for _, alternative := range response.Alternatives {
		if len(alternative) == 0 {
			return NewError(ErrorUpstreamUnavailable, "empty_response_alternative", "upstream response contains an empty alternative", nil)
		}
	}
	return nil
}

func validateOutputSequence(
	sequence []OutputItem,
	limits Limits,
	outputItems *int,
	blocks *int,
	itemIDs map[string]struct{},
) error {
	*outputItems += len(sequence)
	if limits.OutputItems > 0 && *outputItems > limits.OutputItems {
		return NewError(ErrorUpstreamUnavailable, "output_items_limit", "upstream output item limit exceeded", nil)
	}
	for index, item := range sequence {
		if err := validateOutputItem(item, index, limits, blocks, itemIDs); err != nil {
			return err
		}
	}
	return nil
}

func validateOutputItem(
	item OutputItem,
	index int,
	limits Limits,
	blocks *int,
	itemIDs map[string]struct{},
) error {
	if strings.TrimSpace(item.ID) == "" {
		return NewError(ErrorUpstreamUnavailable, "output_id_required", "upstream output item ID is required", nil)
	}
	if exceeds(item.ID, limits.IdentifierBytes) {
		return NewError(ErrorUpstreamUnavailable, "output_id_limit", "upstream output item ID exceeds the configured limit", nil)
	}
	if _, duplicate := itemIDs[item.ID]; duplicate {
		return NewError(ErrorUpstreamUnavailable, "duplicate_output_id", "upstream output item IDs must be unique", nil)
	}
	itemIDs[item.ID] = struct{}{}
	if item.Role != "" && item.Role != RoleAssistant && item.Role != RoleTool {
		return NewError(ErrorUpstreamUnavailable, "invalid_output_role", "upstream output role is invalid", nil)
	}
	if len(item.Content) == 0 {
		return NewError(ErrorUpstreamUnavailable, "empty_output_item", fmt.Sprintf("upstream output item %d is empty", index), nil)
	}
	for _, content := range item.Content {
		if err := validateOutputContent(item.Role, content, blocks, limits); err != nil {
			return err
		}
	}
	return nil
}

func validateOutputContent(role Role, content Content, blocks *int, limits Limits) error {
	if role == RoleAssistant && content.Kind == ContentToolResult ||
		role == RoleTool && content.Kind != ContentToolResult {
		return NewError(ErrorUpstreamUnavailable, "invalid_output_role_content", "upstream output role and content do not match", nil)
	}
	(*blocks)++
	if err := validateContent(content, blocks, limits, 0); err != nil {
		return err
	}
	if content.Kind == ContentGeneratedImage && content.GeneratedImage.Status != ImageGenerationCompleted &&
		content.GeneratedImage.Status != ImageGenerationFailed {
		return NewError(
			ErrorUpstreamUnavailable,
			"nonterminal_image_generation_output",
			"upstream buffered image generation output is not terminal",
			nil,
		)
	}
	return nil
}

// ValidateTransportError enforces the bounded neutral contract before and
// after transport-error mutation. The public sanitizer may impose tighter
// limits before terminal evidence is persisted.
func ValidateTransportError(transportError TransportError, limits Limits) error {
	if transportError.Error == nil {
		return NewError(ErrorUpstreamUnavailable, "transport_error_required", "upstream transport error is missing", nil)
	}
	if err := validateProtocolError(transportError.Error); err != nil {
		return err
	}
	if exceeds(transportError.ProviderRequestID, limits.IdentifierBytes) ||
		exceeds(transportError.Error.Code, limits.IdentifierBytes) ||
		exceeds(transportError.Error.Parameter, limits.IdentifierBytes) ||
		exceeds(transportError.Error.Message, limits.TextBytes) {
		return NewError(ErrorUpstreamUnavailable, "transport_error_limit", "upstream transport error exceeds the configured limit", nil)
	}
	return nil
}

func validateProtocolError(protocolError *ProtocolError) error {
	if protocolError == nil {
		return NewError(ErrorUpstreamUnavailable, "protocol_error_required", "protocol error is missing", nil)
	}
	if !validErrorCategory(protocolError.Category) {
		return NewError(ErrorUpstreamUnavailable, "invalid_error_category", "protocol error category is invalid", nil)
	}
	if strings.TrimSpace(protocolError.Message) == "" {
		return NewError(ErrorUpstreamUnavailable, "error_message_required", "protocol error message is required", nil)
	}
	return nil
}

func validErrorCategory(category ErrorCategory) bool {
	switch category {
	case ErrorInvalidRequest, ErrorAuthentication, ErrorPermission, ErrorNotFound,
		ErrorConflict, ErrorUnsupportedFeature, ErrorRateLimited,
		ErrorUpstreamUnavailable, ErrorUpstreamTimeout, ErrorInternal:
		return true
	default:
		return false
	}
}

func ValidateUsage(usage Usage) error {
	if usage.State != UsageAvailable && usage.State != UsageUnavailable {
		return NewError(ErrorUpstreamUnavailable, "usage_state", "usage state is required", nil)
	}
	counts := []TokenCount{
		usage.InputUncached, usage.InputCacheRead, usage.InputCacheWrite,
		usage.OutputReasoning, usage.OutputOther, usage.InputTotal, usage.OutputTotal, usage.Total,
	}
	hasValue := false
	for _, count := range counts {
		if err := validateTokenCount(count); err != nil {
			return err
		}
		hasValue = hasValue || count.Value != nil
	}
	if usage.State == UsageUnavailable && hasValue {
		return NewError(ErrorUpstreamUnavailable, "usage_state", "unknown usage cannot carry token counts", nil)
	}
	if usage.State == UsageAvailable && !hasValue {
		return NewError(ErrorUpstreamUnavailable, "usage_state", "available usage requires at least one token count", nil)
	}
	return validateUsageTotals(usage)
}

func validateTokenCount(count TokenCount) error {
	if count.Value != nil && *count.Value < 0 {
		return NewError(ErrorUpstreamUnavailable, "negative_usage", "upstream usage cannot be negative", nil)
	}
	if count.Value == nil && count.Provenance != "" && count.Provenance != UsageUnknown {
		return NewError(ErrorUpstreamUnavailable, "usage_provenance", "usage provenance requires a value", nil)
	}
	if count.Value != nil && count.Provenance == "" {
		return NewError(ErrorUpstreamUnavailable, "usage_provenance", "usage value requires provenance", nil)
	}
	if !validUsageProvenance(count.Provenance) {
		return NewError(ErrorUpstreamUnavailable, "usage_provenance", "usage provenance is invalid", nil)
	}
	return nil
}

func validUsageProvenance(provenance UsageProvenance) bool {
	return provenance == "" || provenance == UsageAuthoritative || provenance == UsageDerived ||
		provenance == UsageEstimated || provenance == UsageUnknown
}

func validateUsageTotals(usage Usage) error {
	checks := [][2]bool{
		usageSumStatus(usage.InputTotal, usage.InputUncached, usage.InputCacheRead, usage.InputCacheWrite),
		usageSumStatus(usage.OutputTotal, usage.OutputReasoning, usage.OutputOther),
		usageSumStatus(usage.Total, usage.InputTotal, usage.OutputTotal),
	}
	for _, check := range checks {
		if check[1] {
			return NewError(ErrorUpstreamUnavailable, "usage_overflow", "upstream usage totals overflow", nil)
		}
		if !check[0] {
			return NewError(ErrorUpstreamUnavailable, "usage_total_mismatch", "upstream usage totals are inconsistent", nil)
		}
	}
	return nil
}

func usageSumStatus(total TokenCount, parts ...TokenCount) [2]bool {
	matches, overflow := countEqualsSum(total, parts...)
	return [2]bool{matches, overflow}
}

func countEqualsSum(total TokenCount, parts ...TokenCount) (bool, bool) {
	var sum int64
	found := false
	for _, part := range parts {
		if part.Value == nil {
			continue
		}
		found = true
		if *part.Value > math.MaxInt64-sum {
			return false, true
		}
		sum += *part.Value
	}
	if total.Value == nil {
		return true, false
	}
	return !found || *total.Value == sum, false
}

func exceeds(value string, limit int) bool { return limit > 0 && len(value) > limit }

func validStopReason(reason StopReason) bool {
	switch reason {
	case "", StopEndTurn, StopMaxTokens, StopSequence, StopToolCall, StopContentFilter,
		StopPaused, StopContextWindow, StopCanceled, StopError, StopUnknown:
		return true
	default:
		return false
	}
}

func RequireCapabilities(format WireFormat, available, required CapabilitySet) error {
	if available.Contains(required) {
		return nil
	}
	missing := make([]string, 0)
	for _, name := range required.Names() {
		capability, err := ParseCapabilities([]string{name})
		if err != nil {
			return NewError(ErrorInternal, "capability_registry_invalid", "capability registry is invalid", err)
		}
		if !available.Contains(capability) {
			missing = append(missing, name)
		}
	}
	return NewError(
		ErrorUnsupportedFeature, "unsupported_capability",
		fmt.Sprintf("wire format %q does not support: %s", format, strings.Join(missing, ", ")), nil,
	)
}

func StableID(parts ...string) string {
	hash := sha256.New()
	var length [8]byte
	for _, part := range parts {
		binary.BigEndian.PutUint64(length[:], uint64(len(part)))
		_, _ = hash.Write(length[:])
		_, _ = hash.Write([]byte(part))
	}
	return "item_" + hex.EncodeToString(hash.Sum(nil)[:12])
}
