package testcases

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// routerStub answers chat completions as the Router would; answer decides each
// response's delay and x-vsr-response-path by request order.
func routerStub(t *testing.T, answer func(n int) (time.Duration, string)) string {
	t.Helper()
	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		delay, path := answer(int(calls.Add(1)))
		time.Sleep(delay)
		w.Header().Set("x-vsr-response-path", path)
		_, _ = w.Write([]byte(`{"object":"chat.completion"}`))
	}))
	t.Cleanup(server.Close)
	address, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	return address.Port()
}

func TestJudgeProgressiveAnswerAcceptsOnlyRouterAnswersInsideTheDeadline(t *testing.T) {
	cases := []struct {
		name    string
		result  StressTestResult
		success bool
		reason  string
	}{
		{
			name:    "routed upstream",
			result:  StressTestResult{Success: true, StatusCode: 200, ResponsePath: "upstream", Decision: "math_decision", Duration: 30 * time.Second},
			success: true,
		},
		{
			name:    "blocked when the signals' deadline passed",
			result:  StressTestResult{Success: true, StatusCode: 200, ResponsePath: "fast_response", Decision: "block_jailbreak", Duration: 45 * time.Second},
			success: true,
		},
		{
			name:    "default route after a refused signal",
			result:  StressTestResult{Success: true, StatusCode: 200, ResponsePath: "upstream", Duration: time.Second},
			success: true,
		},
		{
			name:   "not produced by the Router",
			result: StressTestResult{Success: true, StatusCode: 200, Duration: time.Second},
			reason: "without x-vsr-response-path",
		},
		{
			name:   "the Router's error path",
			result: StressTestResult{Success: true, StatusCode: 200, ResponsePath: "error", Duration: time.Second},
			reason: "error path",
		},
		{
			name:   "past the gateway's deadline",
			result: StressTestResult{Success: true, StatusCode: 200, ResponsePath: "upstream", Duration: 61 * time.Second},
			reason: "past the gateway's 1m0s deadline",
		},
		{
			name:   "gateway error",
			result: StressTestResult{StatusCode: 504, ErrorMessage: "status 504: upstream request timeout", Duration: 60 * time.Second},
			reason: "status 504",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result := tc.result
			judgeProgressiveAnswer(&result)
			if result.Success != tc.success {
				t.Fatalf("success = %v, want %v (%s)", result.Success, tc.success, result.ErrorMessage)
			}
			if !tc.success && !strings.Contains(result.ErrorMessage, tc.reason) {
				t.Fatalf("error %q does not name %q", result.ErrorMessage, tc.reason)
			}
		})
	}
}

func TestStageStatsReportAnswersByResponsePath(t *testing.T) {
	stage := calculateStageStats(ProgressiveStageResult{
		TotalReqs:    4,
		SuccessCount: 3,
		FailureCount: 1,
		MinDuration:  time.Hour,
		Results: []StressTestResult{
			{Success: true, ResponsePath: "fast_response", Decision: "block_jailbreak", Duration: 45 * time.Second},
			{Success: true, ResponsePath: "upstream", Duration: time.Second},
			{Success: true, ResponsePath: "upstream", Decision: "math_decision", Duration: 30 * time.Second},
			{ResponsePath: "upstream", Duration: 61 * time.Second},
		},
	})
	if stage.ResponsePaths["upstream"] != 2 || stage.ResponsePaths["fast_response"] != 1 || stage.Undecided != 1 {
		t.Fatalf("paths = %v, undecided = %d", stage.ResponsePaths, stage.Undecided)
	}
}

func TestBaselineLatencyIgnoresCacheHits(t *testing.T) {
	port := routerStub(t, func(n int) (time.Duration, string) {
		if n%2 == 1 {
			return 0, "cache"
		}
		return 50 * time.Millisecond, "upstream"
	})
	baseline, err := measureBaselineLatency(context.Background(), port)
	if err != nil {
		t.Fatal(err)
	}
	if baseline < 50*time.Millisecond {
		t.Fatalf("baseline %v includes a cache hit", baseline)
	}
}

func TestWaitForDrainReturnsOnceTheRouterAnswersInItsUnloadedTime(t *testing.T) {
	port := routerStub(t, func(n int) (time.Duration, string) {
		switch n {
		case 1:
			return 1500 * time.Millisecond, "fast_response" // still queued behind the burst
		case 2:
			return 0, "cache" // a cache hit evaluates no model
		case 3:
			return 0, "upstream" // refused work evaluates no model either
		default:
			return 200 * time.Millisecond, "upstream"
		}
	})
	drained, err := waitForDrain(context.Background(), port, 200*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	if drained < 1700*time.Millisecond {
		t.Fatalf("drained after %v, before an answer in the unloaded time", drained)
	}
}
