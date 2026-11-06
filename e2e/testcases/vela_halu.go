package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"slices"
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

// velaHaluSignal is the hallucination rule the grounded-answer decision's
// response_guard reads.
const velaHaluSignal = "hallucination:ungrounded_claims"

type guardProbeResponse struct {
	Mode         string   `json:"mode"`
	Persisted    bool     `json:"persisted"`
	BackendCalls bool     `json:"backend_calls"`
	Action       string   `json:"action"`
	Warnings     []string `json:"warnings"`
	Rules        []struct {
		State string `json:"state"`
	} `json:"rules"`
	Signals []guardProbeSignal `json:"signals"`
}

type guardProbeSignal struct {
	Signal           string   `json:"signal"`
	Verdict          string   `json:"verdict"`
	Reason           string   `json:"reason"`
	Score            *float64 `json:"score"`
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
// and the Kubernetes profile, including the published task input budget. It
// probes the response_guard of the grounded-answer decision, whose one rule
// reads the hallucination rule over each answer and its context.
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
			"mode":    "probe", "context": probe.context, "question": probe.question, "response": probe.answer,
		})
		if err != nil {
			return err
		}
		request, err := http.NewRequestWithContext(ctx, http.MethodPost, apiURL+"/api/v1/plugins/response_guard/preview", bytes.NewReader(body))
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
		var result guardProbeResponse
		if err := json.Unmarshal(data, &result); err != nil {
			return err
		}
		if result.Mode != "probe" || result.Persisted || !result.BackendCalls || len(result.Rules) != 1 {
			return fmt.Errorf("%s: the guard did not read the answer: %s", probe.name, data)
		}
		index := slices.IndexFunc(result.Signals, func(signal guardProbeSignal) bool { return signal.Signal == velaHaluSignal })
		if index < 0 {
			return fmt.Errorf("%s: the detector was not invoked: %s", probe.name, data)
		}
		signal := result.Signals[index]
		if probe.rejected {
			if signal.Verdict != "unavailable" || signal.Reason != "hallucination_evaluation_failed" || result.Rules[0].State != "unknown" {
				return fmt.Errorf("over-budget Halu pair was silently truncated or accepted: %s", data)
			}
			continue
		}
		if detected := signal.Verdict == "detected"; detected != probe.detected {
			return fmt.Errorf("%s: incorrect grounded verdict: %s", probe.name, data)
		}
		if probe.detected {
			if result.Action != "annotate" || !slices.Contains(result.Warnings, "hallucination") || signal.Score == nil || *signal.Score <= .5 || len(signal.UnsupportedSpans) == 0 {
				return fmt.Errorf("%s: missing published operating-point evidence: %s", probe.name, data)
			}
			for _, span := range signal.UnsupportedSpans {
				if span == "" || !utf8.ValidString(span) || !strings.Contains(probe.answer, span) {
					return fmt.Errorf("%s: invalid UTF-8 answer span %q", probe.name, span)
				}
			}
		} else if len(signal.UnsupportedSpans) != 0 || result.Action != "none" {
			return fmt.Errorf("supported answer was flagged: %s", data)
		}
	}
	if setDetails != nil {
		setDetails(map[string]interface{}{"model": "Vela-1.0-Encoder-307M-Halu", "total_tests": len(probes), "correct_tests": len(probes), "minimum_accuracy_rate": 100.0, "accuracy_rate": 100.0})
	}
	return nil
}
