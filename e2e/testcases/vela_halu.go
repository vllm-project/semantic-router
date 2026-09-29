package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("vela-halu-grounding", pkgtestcases.TestCase{
		Description: "Published Halu pair grounding: supported and contradictory answers, Unicode spans, and the 8192-token task limit",
		Tags:        []string{"kubernetes", "hallucination", "real-model", "vela"}, Fn: testVelaHaluGrounding,
	})
}

type haluProbeResponse struct {
	Mode             string   `json:"mode"`
	Persisted        bool     `json:"persisted"`
	BackendCalls     bool     `json:"backend_calls"`
	Enabled          bool     `json:"enabled"`
	Eligible         bool     `json:"eligible"`
	Resolved         bool     `json:"resolved"`
	Detected         bool     `json:"detected"`
	Action           string   `json:"action"`
	Reason           string   `json:"reason"`
	Score            *float32 `json:"score"`
	UnsupportedSpans []string `json:"unsupported_spans"`
}

func testVelaHaluGrounding(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	port, cleanup, err := setupRouterAPIConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer cleanup()
	return RunVelaHaluContract(ctx, "http://localhost:"+port, opts.SetDetails)
}

// RunVelaHaluContract shares real-detector acceptance between local CLI stacks
// and the Kubernetes profile, including the published task input budget.
func RunVelaHaluContract(ctx context.Context, apiURL string, setDetails func(map[string]interface{})) error {
	httpClient := &http.Client{Timeout: 120 * time.Second}
	probes := []struct {
		name, context, question, answer string
		detected, rejected              bool
	}{
		{"supported", "The museum opens at 10:00 on Tuesday.", "When does the museum open on Tuesday?", "The museum opens at 10:00 on Tuesday.", false, false},
		{"contradictory_time", "The museum opens at 10:00 on Tuesday.", "When does the museum open on Tuesday?", "The museum opens at 09:00 on Tuesday.", true, false},
		{"unicode", "Élodie lives in Paris. 王明住在北京。", "Where do Élodie and 王明 live?", "Élodie lives in Berlin. 王明住在上海。", true, false},
		{"task_budget", strings.Repeat("hello ", 9000), "When does the museum open?", "The museum opens at 10:00.", false, true},
	}
	for _, probe := range probes {
		body, err := json.Marshal(map[string]interface{}{
			"binding": map[string]string{"recipe": "default", "decision": "grounded-answer"},
			"mode":    "probe", "fact_check_needed": true, "context": probe.context, "question": probe.question, "response": probe.answer,
		})
		if err != nil {
			return err
		}
		request, err := http.NewRequestWithContext(ctx, http.MethodPost, apiURL+"/api/v1/plugins/hallucination/preview", bytes.NewReader(body))
		if err != nil {
			return err
		}
		request.Header.Set("Content-Type", "application/json")
		response, err := httpClient.Do(request)
		if err != nil {
			return fmt.Errorf("%s: %w", probe.name, err)
		}
		data, readErr := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		response.Body.Close()
		if readErr != nil {
			return readErr
		}
		if response.StatusCode != http.StatusOK {
			return fmt.Errorf("%s: status %d: %s", probe.name, response.StatusCode, data)
		}
		var result haluProbeResponse
		if err := json.Unmarshal(data, &result); err != nil {
			return err
		}
		if result.Mode != "probe" || result.Persisted || !result.BackendCalls || !result.Enabled || !result.Eligible {
			return fmt.Errorf("%s: detector was not invoked: %s", probe.name, data)
		}
		if probe.rejected {
			if result.Resolved || result.Detected || result.Reason != "detection_failed" {
				return fmt.Errorf("over-budget Halu pair was silently truncated or accepted: %s", data)
			}
			continue
		}
		if !result.Resolved || result.Detected != probe.detected {
			return fmt.Errorf("%s: incorrect grounded verdict: %s", probe.name, data)
		}
		if probe.detected {
			if result.Action != "header" || result.Score == nil || *result.Score <= .5 || len(result.UnsupportedSpans) == 0 {
				return fmt.Errorf("%s: missing published operating-point evidence: %s", probe.name, data)
			}
			for _, span := range result.UnsupportedSpans {
				if span == "" || !utf8.ValidString(span) || !strings.Contains(probe.answer, span) {
					return fmt.Errorf("%s: invalid UTF-8 answer span %q", probe.name, span)
				}
			}
		} else if len(result.UnsupportedSpans) != 0 {
			return fmt.Errorf("supported answer returned unsupported spans: %s", data)
		}
	}
	if setDetails != nil {
		setDetails(map[string]interface{}{"model": "Vela-1.0-Encoder-307M-Halu", "total_tests": len(probes), "correct_tests": len(probes), "minimum_accuracy_rate": 100.0, "accuracy_rate": 100.0})
	}
	return nil
}
