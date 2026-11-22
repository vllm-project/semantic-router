package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/google/uuid"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("shadow-dispatch-multi-model-completes", pkgtestcases.TestCase{
		Description: "Candidate groups complete with concurrency one, including a failed candidate, while HTTP and SSE primaries match no-shadow baselines",
		Tags:        []string{"router-replay", "shadow-dispatch", "functional", "streaming", "failure-isolation"},
		Fn:          testShadowDispatchMultiModel,
	})
}

func testShadowDispatchMultiModel(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) (result error) {
	run, err := openShadowDispatchRun(ctx, client, opts)
	if err != nil {
		return err
	}
	defer run.Close()
	defer run.releaseOnExit(ctx, "malformed", &result)
	if resetErr := run.control(ctx, "malformed", "reset", "malformed"); resetErr != nil {
		return resetErr
	}
	for _, scenario := range []struct {
		entrypoint string
		outcomes   map[string]string
	}{
		{"vllm-sr/shadow-multi", map[string]string{
			"openai/shadow-candidate": "completed", "openai/shadow-second": "completed",
		}},
		{"vllm-sr/shadow-multi-failure", map[string]string{
			"openai/shadow-malformed": "malformed_response", "openai/shadow-candidate": "completed",
		}},
	} {
		session, primaryErr := run.primary(ctx, scenario.entrypoint, shadowPrimaryCeiling)
		if primaryErr != nil {
			return primaryErr
		}
		record, replayErr := run.replay(ctx, session, len(scenario.outcomes))
		if replayErr != nil {
			return replayErr
		}
		if outcomeErr := requireShadowMultiOutcomes(record, scenario.outcomes); outcomeErr != nil {
			return outcomeErr
		}
		if streamErr := runShadowMultiStreaming(ctx, run, scenario.entrypoint, scenario.outcomes); streamErr != nil {
			return streamErr
		}
	}
	// One failed candidate in each HTTP/SSE request, with no retries.
	_, err = run.waitProvider(ctx, "malformed", 2, 0)
	return err
}

func runShadowMultiStreaming(ctx context.Context, run *shadowDispatchRun, entrypoint string, expected map[string]string) error {
	var baseline []byte
	for _, model := range []string{"vllm-sr/auto", entrypoint} {
		sessionID := "e2e_shadow_stream_" + uuid.NewString()
		requestCtx, cancel := context.WithTimeout(ctx, shadowPrimaryCeiling)
		response, err := sendProtocolMatrixRaw(requestCtx, run.gateway, "/v1/chat/completions", map[string]any{
			"model": model, "stream": true, "user": "e2e-shadow-user",
			"messages": []fixtures.ChatMessage{{Role: "user", Content: "Please say hello."}},
		}, true, map[string]string{"x-authz-user-id": "e2e-shadow-user", "x-session-id": sessionID})
		cancel()
		run.details["primary_sse_"+model] = map[string]any{"session_id": sessionID, "response": response}
		if err != nil {
			return fmt.Errorf("primary SSE %s: %w", model, err)
		}
		if response.StatusCode != http.StatusOK {
			return fmt.Errorf("primary SSE %s status=%d: %s", model, response.StatusCode, response.Body)
		}
		if streamErr := validateStreamingCacheStream(response.Headers, response.Body, shadowPrimaryContent); streamErr != nil {
			return fmt.Errorf("primary SSE %s: %w", model, streamErr)
		}
		// Compare all mock-provider fields except creation time; the shared
		// validator above checks exact text, finish_reason=stop and final [DONE].
		var frames []map[string]any
		for _, data := range protocolSSEDataFrames(response.Body) {
			if data == "[DONE]" {
				continue
			}
			var fields map[string]any
			if decodeErr := json.Unmarshal([]byte(data), &fields); decodeErr != nil {
				return decodeErr
			}
			if fields["model"] != shadowPrimaryModel {
				return fmt.Errorf("unexpected primary SSE model: %s", data)
			}
			delete(fields, "created")
			frames = append(frames, fields)
		}
		normalized, err := json.Marshal(frames)
		if err != nil {
			return err
		}
		if baseline == nil {
			baseline = normalized
			continue
		}
		if !bytes.Equal(normalized, baseline) {
			return fmt.Errorf("primary SSE differs from no-shadow baseline: got %s, want %s", normalized, baseline)
		}
		// Streaming replay reconstructs a completion, so compare SSE at the
		// client boundary and check the replay's primary model/content separately.
		record, err := waitForShadowReplay(ctx, run.api, sessionID, len(expected))
		if err != nil {
			return err
		}
		run.details["stream_replay_"+model] = record
		if record.SessionID != sessionID || record.SelectedModel != shadowPrimaryModel {
			return fmt.Errorf("unexpected streaming replay identity: %+v", record)
		}
		if _, err := normalizeShadowPrimary([]byte(record.ResponseBody)); err != nil {
			return err
		}
		if err := requireShadowMultiOutcomes(record, expected); err != nil {
			return err
		}
	}
	return nil
}

func requireShadowMultiOutcomes(record *shadowReplayRecord, expected map[string]string) error {
	if len(record.Outcomes) != len(expected) {
		return fmt.Errorf("replay %s has %d outcomes, want %d", record.ID, len(record.Outcomes), len(expected))
	}
	seenModels, seenRequests := map[string]bool{}, map[string]bool{}
	for _, outcome := range record.Outcomes {
		reason, found := expected[outcome.TargetRef]
		if !found || seenModels[outcome.TargetRef] {
			return fmt.Errorf("unexpected candidate outcome: %+v", outcome)
		}
		requestID := outcome.Metadata["shadow_request_id"]
		if seenRequests[requestID] {
			return fmt.Errorf("replay %s candidates share a shadow request ID", record.ID)
		}
		verdict := "completed"
		if reason != "completed" {
			verdict = "failed"
		}
		// The mocker emits SSE for stream=true, so completed JSON shadows also
		// guard nonstream dispatch while the primary streams.
		single := *record
		single.Outcomes = []shadowReplayOutcome{outcome}
		if err := requireShadowOutcome(&single, outcome.TargetRef, verdict, reason); err != nil {
			return err
		}
		if reason == "malformed_response" && outcome.Metadata["status_code"] != "200" {
			return fmt.Errorf("malformed candidate must reach the decoder after HTTP 200: %v", outcome.Metadata)
		}
		seenModels[outcome.TargetRef], seenRequests[requestID] = true, true
	}
	return nil
}
