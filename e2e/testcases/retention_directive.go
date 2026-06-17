package testcases

import (
	"context"
	"fmt"
	"net/http"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("retention-directive", pkgtestcases.TestCase{
		Description: "Verify EMIT retention directives surface as x-vsr-retention-* response headers (issue #2009)",
		Tags:        []string{"kubernetes", "routing", "retention", "headers"},
		Fn:          testRetentionDirective,
	})
}

// retentionProbeKeyword is the sentinel token wired into the kubernetes
// profile's retention_probe_decision (see e2e/profiles/ai-gateway/values.yaml).
// A prompt containing it deterministically routes to that decision, which emits
//
//	retention { ttl_turns: 0, keep_current_model: true, prefer_prefix_retention: true }
//
// so the router must mirror those fields onto the response as x-vsr-retention-*
// headers.
const retentionProbeKeyword = "__RETENTION_PROBE__"

// testRetentionDirective asserts the response-header half of the EMIT retention
// contract end-to-end: every explicitly-set field is emitted, an explicit
// ttl_turns: 0 is preserved (tri-state regression guard), and an unset field
// (drop) is omitted.
func testRetentionDirective(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing retention directive response headers")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	query := "Diagnostics request " + retentionProbeKeyword + " please run."

	response, err := sendLocalChatCompletion(ctx, localPort, "MoM", query, 30*time.Second, false)
	if err != nil {
		return fmt.Errorf("retention directive request failed: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		logUnexpectedChatCompletionStatus(opts.Verbose, response, "retention-directive", "Query: "+query)
		return fmt.Errorf("retention directive request: %s", formatUnexpectedChatCompletionStatus(response))
	}

	// Guard: confirm the keyword rule actually matched the retention probe
	// decision. Without this, a silently non-matching rule would make the
	// header assertions vacuous (all absent, but for the wrong reason).
	if decision := response.Headers.Get("x-vsr-selected-decision"); decision != "retention_probe_decision" {
		logUnexpectedChatCompletionStatus(opts.Verbose, response, "retention-directive", "Query: "+query)
		return fmt.Errorf("expected x-vsr-selected-decision=retention_probe_decision, got %q", decision)
	}

	// Every explicitly-set retention field must be emitted. ttl_turns is the
	// tri-state regression guard (issue #2009): an explicit 0 must still be
	// emitted, not dropped by a ">0" gate.
	expected := []struct {
		header string
		want   string
	}{
		{"x-vsr-retention-ttl-turns", "0"},
		{"x-vsr-retention-keep-current-model", "true"},
		{"x-vsr-retention-prefer-prefix", "true"},
	}
	for _, e := range expected {
		got := response.Headers.Get(e.header)
		if got != e.want {
			return fmt.Errorf("header %s: expected %q, got %q", e.header, e.want, got)
		}
		if opts.Verbose {
			fmt.Printf("[Test]   %s = %q ✓\n", e.header, got)
		}
	}

	// drop was not set on the probe decision; tri-state semantics require an
	// unset field to be omitted rather than sent as a default.
	if got := response.Headers.Get("x-vsr-retention-drop"); got != "" {
		return fmt.Errorf("x-vsr-retention-drop must be absent for an unset field, got %q", got)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"decision":                           "retention_probe_decision",
			"x-vsr-retention-ttl-turns":          "0",
			"x-vsr-retention-keep-current-model": "true",
			"x-vsr-retention-prefer-prefix":      "true",
			"x-vsr-retention-drop-absent":        true,
		})
	}

	if opts.Verbose {
		fmt.Println("[Test] Retention directive response headers verified ✓")
	}
	return nil
}
