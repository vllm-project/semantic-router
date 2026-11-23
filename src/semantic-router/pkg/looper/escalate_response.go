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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// EscalateExecutionEvidence is the content-free accounting trace retained
// when an escalation fails after one or more backend calls completed.
// It deliberately excludes ModelResponse and response bodies so transporting
// the error through extproc cannot disclose candidate or verifier text.
type EscalateExecutionEvidence struct {
	ModelsUsed     []string
	Iterations     int
	Usage          TokenUsage
	ExecutionTrace ExecutionTrace
}

// EscalatePartialExecutionError preserves paid execution evidence while
// keeping the ordinary error message safe to return through the API.
type EscalatePartialExecutionError struct {
	cause    error
	evidence EscalateExecutionEvidence
}

func (e *EscalatePartialExecutionError) Error() string {
	if e == nil || e.cause == nil {
		return "escalate execution failed"
	}
	return e.cause.Error()
}

func (e *EscalatePartialExecutionError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.cause
}

// Evidence returns a defensive copy of the content-free execution trace.
func (e *EscalatePartialExecutionError) Evidence() EscalateExecutionEvidence {
	if e == nil {
		return EscalateExecutionEvidence{}
	}
	evidence := e.evidence
	evidence.ModelsUsed = append([]string(nil), evidence.ModelsUsed...)
	evidence.ExecutionTrace.Attempts = append([]AttemptTrace(nil), evidence.ExecutionTrace.Attempts...)
	return evidence
}

// EscalateEvidenceFromError extracts partial escalate accounting without
// exposing the concrete error implementation to callers.
func EscalateEvidenceFromError(err error) (EscalateExecutionEvidence, bool) {
	var partial *EscalatePartialExecutionError
	if !errors.As(err, &partial) {
		return EscalateExecutionEvidence{}, false
	}
	return partial.Evidence(), true
}

func newEscalatePartialExecutionError(
	cause error,
	responses []*ModelResponse,
	modelsUsed []string,
	iterations int,
	trace ExecutionTrace,
) error {
	return &EscalatePartialExecutionError{
		cause: cause,
		evidence: EscalateExecutionEvidence{
			ModelsUsed:     append([]string(nil), modelsUsed...),
			Iterations:     iterations,
			Usage:          SumUsage(responses...),
			ExecutionTrace: trace,
		},
	}
}

// formatEscalateJSONResponse creates response with the escalate algorithm type.
func (l *EscalateLooper) formatEscalateJSONResponse(
	agg *AggregatedResponse,
	modelsUsed []string,
	iterations int,
) (*Response, error) {
	resp, err := l.formatJSONResponse(agg, modelsUsed, iterations)
	if err != nil {
		return nil, err
	}
	resp.AlgorithmType = config.DecisionAlgorithmEscalate
	return resp, nil
}

// formatEscalateStreamingResponse publishes only the candidate selected by
// the escalation while retaining usage from every attempted model.
func (l *EscalateLooper) formatEscalateStreamingResponse(
	agg *AggregatedResponse,
	modelsUsed []string,
	iterations int,
	includeUsage bool,
) (*Response, error) {
	if len(agg.Responses) == 0 {
		return nil, fmt.Errorf("escalate produced no model responses")
	}
	selected := *agg
	selected.Responses = []*ModelResponse{agg.Responses[len(agg.Responses)-1]}
	resp, err := l.formatStreamingResponse(&selected, modelsUsed, iterations)
	if err != nil {
		return nil, err
	}
	resp.AlgorithmType = config.DecisionAlgorithmEscalate
	resp.Usage = SumUsage(aggregatedUsageResponses(agg)...)
	resp.Body = normalizeEscalateSSEUsage(resp.Body, resp.Model, resp.Usage, includeUsage)
	return resp, nil
}

// StreamWithUsage replaces the usage chunks of a Looper Chat stream with one
// chunk carrying usage, before DONE. A Looper stream carries usage only when a
// Chat client asks for it, so the Router adds it for the client's codec.
func StreamWithUsage(body []byte, model string, usage TokenUsage) []byte {
	return normalizeEscalateSSEUsage(body, model, usage, true)
}

// normalizeEscalateSSEUsage suppresses any usage emitted by the selected
// upstream stream. When the client requested stream_options.include_usage, it
// emits exactly one aggregate usage chunk before DONE; otherwise it emits none.
// Escalate may pay for rejected candidates and self-verification calls, so
// forwarding the selected model's usage would under-report the escalation.
// The usage chunk names the stream's own model, which a backend may report
// differently from the requested name; model names a stream that reports none.
func normalizeEscalateSSEUsage(body []byte, model string, usage TokenUsage, includeUsage bool) []byte {
	normalized := bytes.ReplaceAll(body, []byte("\r\n"), []byte("\n"))
	state := escalateSSEUsageState{
		out:     make([]byte, 0, len(body)+192),
		id:      fmt.Sprintf("chatcmpl-looper-%d", time.Now().UnixNano()),
		created: time.Now().Unix(),
		model:   model,
	}
	for _, line := range bytes.Split(normalized, []byte("\n")) {
		state.consumeLine(line)
	}

	if includeUsage {
		usagePayload := map[string]interface{}{
			"id":      state.id,
			"object":  "chat.completion.chunk",
			"created": state.created,
			"model":   state.model,
			"choices": []interface{}{},
			"usage":   usage.Map(),
		}
		state.out = appendSSEDataLine(state.out, usagePayload)
	}
	return appendSSEDone(state.out)
}

type escalateSSEUsageState struct {
	out     []byte
	id      string
	created int64
	model   string
}

func (state *escalateSSEUsageState) consumeLine(line []byte) {
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

	payload, ok := decodeEscalateSSEPayload(data)
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

func decodeEscalateSSEPayload(data []byte) (map[string]interface{}, bool) {
	var payload map[string]interface{}
	if len(data) == 0 || json.Unmarshal(data, &payload) != nil {
		return nil, false
	}
	return payload, true
}

func (state *escalateSSEUsageState) captureEnvelope(payload map[string]interface{}) {
	if value, ok := payload["id"].(string); ok && value != "" {
		state.id = value
	}
	if value, ok := payload["created"].(float64); ok {
		state.created = int64(value)
	}
	if value, ok := payload["model"].(string); ok && value != "" {
		state.model = value
	}
}

func (state *escalateSSEUsageState) appendUsageStrippedPayload(payload map[string]interface{}) {
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

func (state *escalateSSEUsageState) appendEvent(line []byte) {
	state.out = append(state.out, line...)
	state.out = append(state.out, '\n', '\n')
}
