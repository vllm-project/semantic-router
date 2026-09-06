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
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// ConfidenceExecutionEvidence is the content-free accounting trace retained
// when a confidence cascade fails after one or more backend calls completed.
// It deliberately excludes ModelResponse and response bodies so transporting
// the error through extproc cannot disclose candidate or verifier text.
type ConfidenceExecutionEvidence struct {
	ModelsUsed []string
	Iterations int
	Usage      TokenUsage
}

// ConfidencePartialExecutionError preserves paid execution evidence while
// keeping the ordinary error message safe to return through the API.
type ConfidencePartialExecutionError struct {
	cause    error
	evidence ConfidenceExecutionEvidence
}

func (e *ConfidencePartialExecutionError) Error() string {
	if e == nil || e.cause == nil {
		return "confidence execution failed"
	}
	return e.cause.Error()
}

func (e *ConfidencePartialExecutionError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.cause
}

// Evidence returns a defensive copy of the content-free execution trace.
func (e *ConfidencePartialExecutionError) Evidence() ConfidenceExecutionEvidence {
	if e == nil {
		return ConfidenceExecutionEvidence{}
	}
	evidence := e.evidence
	evidence.ModelsUsed = append([]string(nil), evidence.ModelsUsed...)
	return evidence
}

// ConfidenceEvidenceFromError extracts partial confidence accounting without
// exposing the concrete error implementation to callers.
func ConfidenceEvidenceFromError(err error) (ConfidenceExecutionEvidence, bool) {
	var partial *ConfidencePartialExecutionError
	if !errors.As(err, &partial) {
		return ConfidenceExecutionEvidence{}, false
	}
	return partial.Evidence(), true
}

func newConfidencePartialExecutionError(
	cause error,
	responses []*ModelResponse,
	modelsUsed []string,
	iterations int,
) error {
	return &ConfidencePartialExecutionError{
		cause: cause,
		evidence: ConfidenceExecutionEvidence{
			ModelsUsed: append([]string(nil), modelsUsed...),
			Iterations: iterations,
			Usage:      SumUsage(responses...),
		},
	}
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
	resp, err := l.formatStreamingResponse(&selected, modelsUsed, iterations)
	if err != nil {
		return nil, err
	}
	resp.AlgorithmType = "confidence"
	resp.Usage = SumUsage(aggregatedUsageResponses(agg)...)
	resp.Body = normalizeConfidenceSSEUsage(resp.Body, resp.Model, resp.Usage, includeUsage)
	return resp, nil
}

// normalizeConfidenceSSEUsage suppresses any usage emitted by the selected
// upstream stream. When the client requested stream_options.include_usage, it
// emits exactly one aggregate usage chunk before DONE; otherwise it emits none.
// Confidence may pay for rejected candidates and self-verification calls, so
// forwarding the selected model's usage would under-report the cascade.
func normalizeConfidenceSSEUsage(body []byte, model string, usage TokenUsage, includeUsage bool) []byte {
	normalized := bytes.ReplaceAll(body, []byte("\r\n"), []byte("\n"))
	state := confidenceSSEUsageState{
		out:     make([]byte, 0, len(body)+192),
		id:      fmt.Sprintf("chatcmpl-looper-%d", time.Now().UnixNano()),
		created: time.Now().Unix(),
	}
	for _, line := range bytes.Split(normalized, []byte("\n")) {
		state.consumeLine(line)
	}

	if includeUsage {
		usagePayload := map[string]interface{}{
			"id":      state.id,
			"object":  "chat.completion.chunk",
			"created": state.created,
			"model":   model,
			"choices": []interface{}{},
			"usage":   usage.Map(),
		}
		state.out = appendSSEDataLine(state.out, usagePayload)
	}
	return appendSSEDone(state.out)
}

type confidenceSSEUsageState struct {
	out     []byte
	id      string
	created int64
}

func (state *confidenceSSEUsageState) consumeLine(line []byte) {
	line = bytes.TrimSuffix(line, []byte("\r"))
	if len(bytes.TrimSpace(line)) == 0 {
		return
	}
	data, ok := sseDataPayload(line)
	if !ok {
		state.out = append(state.out, line...)
		state.out = append(state.out, '\n')
		return
	}
	data = bytes.TrimSpace(data)
	if bytes.Equal(data, []byte("[DONE]")) {
		return
	}

	payload, ok := decodeConfidenceSSEPayload(data)
	if !ok {
		state.appendEvent(line)
		return
	}
	state.captureEnvelope(payload)
	if _, hasUsage := payload["usage"]; hasUsage {
		state.appendUsageStrippedPayload(payload)
		return
	}
	state.appendEvent(line)
}

func decodeConfidenceSSEPayload(data []byte) (map[string]interface{}, bool) {
	var payload map[string]interface{}
	if len(data) == 0 || json.Unmarshal(data, &payload) != nil {
		return nil, false
	}
	return payload, true
}

func (state *confidenceSSEUsageState) captureEnvelope(payload map[string]interface{}) {
	if value, ok := payload["id"].(string); ok && value != "" {
		state.id = value
	}
	if value, ok := payload["created"].(float64); ok {
		state.created = int64(value)
	}
}

func (state *confidenceSSEUsageState) appendUsageStrippedPayload(payload map[string]interface{}) {
	delete(payload, "usage")
	choices, hasChoices := payload["choices"].([]interface{})
	if !hasChoices || len(choices) == 0 {
		return
	}
	rewritten, err := json.Marshal(payload)
	if err != nil {
		return
	}
	state.out = append(state.out, []byte("data: ")...)
	state.appendEvent(rewritten)
}

func (state *confidenceSSEUsageState) appendEvent(line []byte) {
	state.out = append(state.out, line...)
	state.out = append(state.out, '\n', '\n')
}
