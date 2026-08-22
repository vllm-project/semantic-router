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

// RoutingEvidence carries the routing decision that led to this Looper
// execution, so algorithms can make bounded, explainable decisions without
// re-deriving why they were invoked. It never carries prompt content:
// MatchedSignals holds config-defined rule/category identifiers (e.g.
// "domain:coding", "keyword:refund"), not the request text that matched
// them.
type RoutingEvidence struct {
	// DecisionName is the name of the decision that selected this algorithm.
	DecisionName string

	// Confidence is the DecisionEngine's confidence score for DecisionName,
	// in [0, 1].
	Confidence float64

	// MatchedSignals lists the config-defined rule/category identifiers that
	// matched for DecisionName (e.g. "domain:coding"). Values come from
	// decision.DecisionResult.MatchedRules and are bounded to identifiers
	// declared in config, never raw request text.
	MatchedSignals []string
}
