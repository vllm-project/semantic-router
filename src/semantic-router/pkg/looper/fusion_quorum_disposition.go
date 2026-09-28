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

package looper

// FusionQuorumDisposition is the closed set of outcomes a below-quorum Fusion
// panel can reach: the shared vocabulary between the algorithm layer, which
// decides every value, and the ExtProc boundary, which alone knows whether
// protocol encoding succeeded and so is where all of them are counted.
//
// The values are content-free and stable so operators can alert on them without
// parsing logs.
type FusionQuorumDisposition string

const (
	FusionQuorumFailed         FusionQuorumDisposition = "quorum_failed"
	FusionQuorumFallbackFailed FusionQuorumDisposition = "fallback_failed"
	FusionQuorumCancelled      FusionQuorumDisposition = "cancelled"

	// FusionQuorumBudgetExhausted means no budget remained for a recovery call.
	FusionQuorumBudgetExhausted FusionQuorumDisposition = "budget_exhausted"

	// FusionQuorumFallbackResponseFailed means the fallback answered but its
	// response could not be built, so nothing was served despite the spend.
	FusionQuorumFallbackResponseFailed FusionQuorumDisposition = "fallback_response_failed"

	// FusionQuorumFallbackReady is the only non-terminal value: the fallback
	// answered and formatted, but protocol translation has not run yet.
	FusionQuorumFallbackReady FusionQuorumDisposition = "fallback_ready"

	// FusionQuorumFallbackServed means encoding succeeded and ExtProc returned
	// the immediate response to Envoy. ExtProc sees no delivery
	// acknowledgement, so this says nothing about what the client received.
	FusionQuorumFallbackServed FusionQuorumDisposition = "fallback_served"

	// FusionQuorumResponseEncodeFailed means an error response was returned to
	// Envoy even though the fallback tokens were spent.
	FusionQuorumResponseEncodeFailed FusionQuorumDisposition = "response_encode_failed"
)

// IsTerminal reports whether the disposition already describes a settled
// outcome. Only FusionQuorumFallbackReady is not.
func (d FusionQuorumDisposition) IsTerminal() bool {
	return d != FusionQuorumFallbackReady
}

// AttemptStates projects per-attempt states onto the closed-enum label set used
// by metrics and logs. It is the only attempt projection that crosses the
// telemetry boundary, so no provider text can follow it.
func (o *FusionQuorumOutcome) AttemptStates() []string {
	if o == nil {
		return nil
	}
	states := make([]string, 0, len(o.Attempts))
	for _, attempt := range o.Attempts {
		states = append(states, string(attempt.State))
	}
	return states
}

// FusionQuorumOutcomeFromError recovers the bounded operator outcome from a
// terminal below-quorum error. Failures reach ExtProc as an error rather than a
// response, so this is the failure path's equivalent of Response.QuorumOutcome;
// handing both paths the same type is what lets one recorder own every sample.
func FusionQuorumOutcomeFromError(err error) (*FusionQuorumOutcome, bool) {
	evidence, ok := FusionQuorumEvidenceFromError(err)
	if !ok {
		return nil, false
	}
	return newFusionQuorumOutcome(
		evidence,
		evidence.SelectedPolicy,
		evidence.FallbackTarget,
		evidence.Disposition,
	), true
}
