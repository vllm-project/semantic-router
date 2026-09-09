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

package testcases

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const looperConfidenceTraceProbe = "__LOOPER_CONFIDENCE_TRACE_PROBE__"

func init() {
	pkgtestcases.Register("looper-confidence-telemetry", pkgtestcases.TestCase{
		Description: "Verify Confidence attempt diagnostics in Router Replay and bounded Prometheus metrics",
		Tags:        []string{"kubernetes", "routing", "looper", "observability"},
		Fn:          testLooperConfidenceTelemetry,
	})
}

func testLooperConfidenceTelemetry(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	response, err := sendLocalChatCompletion(ctx, localPort, "MoM", looperConfidenceTraceProbe, 30*time.Second)
	if err != nil {
		return fmt.Errorf("confidence telemetry request: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("confidence telemetry request: %s", formatUnexpectedChatCompletionStatus(response))
	}
	management, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer management.Close()
	replayID, err := fetchFirstReplayRecordID(
		management,
		"/v1/router_replay?decision=looper_confidence_trace_decision&limit=1",
		false,
	)
	if err != nil {
		return fmt.Errorf("list confidence replay: %w", err)
	}

	replayResponse, err := fixtures.DoGETRequest(ctx, management.HTTPClient(15*time.Second), management.URL("/v1/router_replay/"+replayID))
	if err != nil {
		return fmt.Errorf("fetch confidence replay: %w", err)
	}
	if replayResponse.StatusCode != http.StatusOK {
		return fmt.Errorf("fetch confidence replay: status %d: %s", replayResponse.StatusCode, replayResponse.Body)
	}
	var replay struct {
		RouteDiagnostics struct {
			Looper struct {
				Attempts            []struct{} `json:"attempts"`
				FinalAttemptOrdinal int        `json:"final_attempt_ordinal"`
			} `json:"looper"`
		} `json:"route_diagnostics"`
	}
	if decodeErr := replayResponse.DecodeJSON(&replay); decodeErr != nil {
		return fmt.Errorf("decode confidence replay: %w", decodeErr)
	}
	if len(replay.RouteDiagnostics.Looper.Attempts) != 2 || replay.RouteDiagnostics.Looper.FinalAttemptOrdinal != 2 {
		return fmt.Errorf("confidence replay attempts=%d final=%d, want 2 and 2", len(replay.RouteDiagnostics.Looper.Attempts), replay.RouteDiagnostics.Looper.FinalAttemptOrdinal)
	}
	if strings.Contains(string(replayResponse.Body), "private-low-confidence-candidate") {
		return fmt.Errorf("confidence replay leaked candidate content")
	}

	metricsSession, err := fixtures.OpenSemanticRouterMetricsSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer metricsSession.Close()
	metricsBody, err := fetchMetrics(ctx, metricsSession)
	if err != nil {
		return err
	}
	for _, metric := range []string{"llm_looper_attempts_total", "llm_looper_execution_duration_seconds"} {
		if !strings.Contains(metricsBody, metric) || !strings.Contains(metricsBody, `algorithm="confidence"`) {
			return fmt.Errorf("metrics output missing Confidence series for %s", metric)
		}
	}
	return nil
}
