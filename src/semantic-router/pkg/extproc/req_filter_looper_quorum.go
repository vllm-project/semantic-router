/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// finalizeLooperQuorumOutcome settles a Fusion quorum outcome: it emits the one
// metric sample and log line, and stages the bounded projection on the request
// context. The success and failure recorders create and finalize the Replay
// record itself, afterwards.
//
// All three finalizing paths converge here, and they are mutually exclusive, so
// each outcome is counted once. A recovered fallback arrives carrying
// FusionQuorumFallbackReady, which only this layer can promote because the
// looper cannot know whether protocol translation will succeed. The
// encode-failure and execution-failure paths arrive already terminal.
//
// Recording fallback_served before translation would let metrics claim a served
// fallback while an error immediate response was returned to Envoy instead.
func finalizeLooperQuorumOutcome(
	reqCtx *RequestContext,
	outcome *looper.FusionQuorumOutcome,
	decision *config.Decision,
	responsePrepared bool,
) {
	if reqCtx == nil || outcome == nil {
		return
	}
	if !outcome.Disposition.IsTerminal() {
		outcome.Disposition = looper.FusionQuorumFallbackServed
		if !responsePrepared {
			outcome.Disposition = looper.FusionQuorumResponseEncodeFailed
		}
	}

	metrics.RecordFusionQuorumFailure(metrics.FusionQuorumOutcome{
		Decision:       decisionName(decision),
		Policy:         outcome.SelectedPolicy,
		Disposition:    string(outcome.Disposition),
		FallbackTarget: outcome.FallbackTarget,
		RequiredCount:  outcome.RequiredCount,
		UsableCount:    outcome.UsableCount,
		AttemptStates:  outcome.AttemptStates(),
	})
	logging.ComponentEvent("extproc", "fusion_quorum_terminal_outcome", map[string]interface{}{
		"request_id":      reqCtx.RequestID,
		"decision":        decisionName(decision),
		"required_count":  outcome.RequiredCount,
		"usable_count":    outcome.UsableCount,
		"selected_policy": outcome.SelectedPolicy,
		"fallback_target": outcome.FallbackTarget,
		"disposition":     string(outcome.Disposition),
	})
	reqCtx.VSRFusionQuorum = looperQuorumOutcomeDiagnostics(outcome)
}

func decisionName(decision *config.Decision) string {
	if decision == nil {
		return ""
	}
	return decision.Name
}
