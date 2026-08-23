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

import (
	"fmt"
)

func newConfidencePartialExecutionError(
	cause error,
	responses []*ModelResponse,
	modelsUsed []string,
	iterations int,
) error {
	return newPartialExecutionError(
		cause,
		executionEvidenceFromResponses(responses, modelsUsed, iterations),
	)
}

// formatConfidenceJSONResponse creates response with confidence algorithm type.
func (l *ConfidenceLooper) formatConfidenceJSONResponse(
	agg *AggregatedResponse,
	modelsUsed []string,
	iterations int,
) (*Response, error) {
	resp, err := l.formatJSONResponse(agg, modelsUsed, iterations)
	if err != nil {
		return nil, err
	}
	resp.AlgorithmType = "confidence"
	return resp, nil
}

// formatConfidenceStreamingResponse publishes only the candidate selected by
// the confidence cascade while retaining usage from every attempted model.
func (l *ConfidenceLooper) formatConfidenceStreamingResponse(
	agg *AggregatedResponse,
	modelsUsed []string,
	iterations int,
	includeUsage bool,
) (*Response, error) {
	if len(agg.Responses) == 0 {
		return nil, fmt.Errorf("confidence produced no model responses")
	}
	selected := *agg
	selected.Responses = []*ModelResponse{agg.Responses[len(agg.Responses)-1]}
	resp, err := l.formatStreamingResponse(&selected, modelsUsed, iterations, includeUsage)
	if err != nil {
		return nil, err
	}
	resp.AlgorithmType = "confidence"
	resp.Usage = SumUsage(aggregatedUsageResponses(agg)...)
	return resp, nil
}
