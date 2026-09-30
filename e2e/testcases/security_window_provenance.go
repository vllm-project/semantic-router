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

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("security-window-provenance", pkgtestcases.TestCase{
		Description: "Verify routing preview names the window a guard score came from",
		Tags:        []string{"kubernetes", "apiserver", "classification", "security", "jailbreak", "api"},
		Fn:          testSecurityWindowProvenance,
	})
}

// Guard scans a prompt longer than one window in overlapping windows and keeps
// the riskiest one, so the score a threshold reads can belong to a part of the
// prompt rather than to all of it (issue #3939). This drives the preview
// boundary with a prompt that needs more than one window and a prompt that fits
// in one, and reads the window the scan reported next to the score. Only the
// deployed classifier produces these: the window count comes from its
// tokenizer, which no mock reproduces.
func testSecurityWindowProvenance(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	sentences := []string{
		"Sedimentary rock forms when layers of sand, mud and organic material settle and compact over long periods. ",
		"Limestone consists mostly of calcium carbonate from the shells of marine organisms. ",
		"Erosion by wind and water gradually reshapes cliffs, valleys and coastlines. ",
		"Fossils preserved in these layers record the sequence of life across geological time. ",
	}
	var builder strings.Builder
	for i := 0; i < 96; i++ {
		builder.WriteString(sentences[i%len(sentences)])
	}

	scanned, err := previewSignalValues(ctx, session, builder.String())
	if err != nil {
		return err
	}
	single, err := previewSignalValues(ctx, session, "What is the capital of France?")
	if err != nil {
		return err
	}

	rule, err := jailbreakSignalRule(scanned)
	if err != nil {
		return err
	}
	windows, start, end, err := jailbreakWindow(scanned, rule)
	if err != nil {
		return fmt.Errorf("long prompt: %w", err)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"rule": rule, "windows": windows, "window_start": start, "window_end": end,
		})
	}
	if opts.Verbose {
		fmt.Printf("[Test] rule=%s windows=%.0f window=[%.0f,%.0f)\n", rule, windows, start, end)
	}
	if windows < 2 {
		return fmt.Errorf(
			"a prompt of %d runes reported %.0f window(s), so the scan is not reporting the windows it read",
			builder.Len(), windows)
	}
	if end <= start {
		return fmt.Errorf("the reported window is empty: [%.0f,%.0f)", start, end)
	}

	shortWindows, shortStart, _, err := jailbreakWindow(single, rule)
	if err != nil {
		return fmt.Errorf("short prompt: %w", err)
	}
	if shortWindows != 1 || shortStart != 0 {
		return fmt.Errorf(
			"a prompt inside one window reported %.0f window(s) starting at %.0f, so the count does not describe the scan",
			shortWindows, shortStart)
	}
	return nil
}

// jailbreakSignalRule returns the jailbreak rule the deployment scored, taken
// from the score key rather than from a name this case would have to keep in
// step with the profile.
func jailbreakSignalRule(values map[string]float64) (string, error) {
	for key := range values {
		if strings.HasPrefix(key, "jailbreak:") && !strings.Contains(strings.TrimPrefix(key, "jailbreak:"), ":") {
			return strings.TrimPrefix(key, "jailbreak:"), nil
		}
	}
	return "", fmt.Errorf("the preview reported no jailbreak signal value: %v", values)
}

func jailbreakWindow(values map[string]float64, rule string) (windows, start, end float64, err error) {
	key := "jailbreak:" + rule
	for suffix, target := range map[string]*float64{
		":windows": &windows, ":window_start": &start, ":window_end": &end,
	} {
		value, ok := values[key+suffix]
		if !ok {
			return 0, 0, 0, fmt.Errorf("the score carries no %s%s: %v", key, suffix, values)
		}
		*target = value
	}
	return windows, start, end, nil
}

func previewSignalValues(
	ctx context.Context,
	session *fixtures.ServiceSession,
	text string,
) (map[string]float64, error) {
	payload, err := json.Marshal(map[string]interface{}{
		"model":    "MoM",
		"messages": []map[string]string{{"role": "user", "content": text}},
	})
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost,
		session.URL("/api/v1/routing/preview?trace=true"), bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := session.HTTPClient(60 * time.Second).Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("routing preview returned %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}
	var document struct {
		SignalValues map[string]float64 `json:"signal_values"`
		SignalErrors map[string]string  `json:"signal_errors"`
	}
	if err := json.Unmarshal(body, &document); err != nil {
		return nil, fmt.Errorf("routing preview response is not JSON: %w", err)
	}
	if len(document.SignalErrors) != 0 {
		return nil, fmt.Errorf("the preview reported signal errors: %v", document.SignalErrors)
	}
	return document.SignalValues, nil
}
