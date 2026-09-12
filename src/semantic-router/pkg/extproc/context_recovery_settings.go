package extproc

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Context recovery is one request-level facility shared by every context
// action that can remove content. Compression and history reset configure it
// independently, so their settings are merged into a single contract here: one
// store, one budget, one reserved retrieval tool, one key set.

// resolveContextRecoverySettings merges the selected decision's compression and
// history-reset recovery settings. Incompatible stores are rejected rather than
// silently resolved in one plugin's favour, and every bound resolves to the
// stricter of the configured values so neither plugin can widen the other's.
func resolveContextRecoverySettings(
	decision *config.Decision,
) (*config.ContextCompressionRecoveryConfig, error) {
	if decision == nil {
		return nil, nil
	}
	var active []*config.ContextCompressionRecoveryConfig
	if compression := decision.GetContextCompressionConfig(); compression != nil &&
		compression.Enabled && compression.Recovery != nil && compression.Recovery.Enabled {
		active = append(active, compression.Recovery)
	}
	if reset := decision.GetHistoryResetConfig(); reset.RequiresRecovery() {
		active = append(active, reset.Recovery)
	}
	switch len(active) {
	case 0:
		return nil, nil
	case 1:
		merged := *active[0]
		return &merged, nil
	}
	return mergeContextRecoverySettings(active[0], active[1])
}

func mergeContextRecoverySettings(
	first *config.ContextCompressionRecoveryConfig,
	second *config.ContextCompressionRecoveryConfig,
) (*config.ContextCompressionRecoveryConfig, error) {
	if !strings.EqualFold(strings.TrimSpace(first.Store), strings.TrimSpace(second.Store)) {
		return nil, fmt.Errorf(
			"context recovery stores disagree: %q and %q",
			strings.TrimSpace(first.Store),
			strings.TrimSpace(second.Store),
		)
	}
	merged := *first
	merged.TTLSeconds = stricterRecoveryBound(first.TTLSeconds, second.TTLSeconds)
	merged.MaxBytesPerRequest = stricterRecoveryBound(first.MaxBytesPerRequest, second.MaxBytesPerRequest)
	merged.MaxTotalBytes = stricterRecoveryBound(first.MaxTotalBytes, second.MaxTotalBytes)
	merged.MaxRetrievals = stricterRecoveryBound(first.MaxRetrievals, second.MaxRetrievals)
	return &merged, nil
}

// stricterRecoveryBound treats zero as "unbounded by this plugin", so a
// configured bound always wins over an omitted one.
func stricterRecoveryBound(first int, second int) int {
	switch {
	case first <= 0:
		return second
	case second <= 0:
		return first
	case second < first:
		return second
	}
	return first
}

// contextRecoverySettingsForRequest resolves the merged settings for the
// request's selected decision. A configuration conflict is reported once, as a
// failure to establish recovery, rather than by guessing a store.
func contextRecoverySettingsForRequest(
	ctx *RequestContext,
) (*config.ContextCompressionRecoveryConfig, error) {
	if ctx == nil {
		return nil, nil
	}
	return resolveContextRecoverySettings(ctx.VSRSelectedDecision)
}
