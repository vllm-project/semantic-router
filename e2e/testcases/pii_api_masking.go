package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("pii-api-masking", pkgtestcases.TestCase{
		Description: "Verify the PII API reports a remote deployment's span at rune offsets and masks only that value",
		Tags:        []string{"kubernetes", "apiserver", "systemone", "pii", "api"},
		Fn:          testPIIAPIMasking,
	})
}

// testPIIAPIMasking asks the PII API to mask a text whose marked name follows
// multi-byte text. The profile's System One fixture labels the name in code
// points, so its rune positions are right only if the router converts the
// offsets, and a longer word that starts with the name must stay unmasked.
func testPIIAPIMasking(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	text := "様が来ました: __PII_SPAN__ PERSON Zorblax replied, and Zorblaxian food was served."
	body, err := json.Marshal(map[string]interface{}{
		"text": text,
		"options": map[string]interface{}{
			"mask_entities":      true,
			"return_positions":   true,
			"reveal_entity_text": true,
		},
	})
	if err != nil {
		return fmt.Errorf("marshal /api/v1/diagnostics/classify/pii payload: %w", err)
	}
	resp, err := postJSON(ctx, session.HTTPClient(30*time.Second), http.MethodPost, session.URL("/api/v1/diagnostics/classify/pii"), body)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected /api/v1/diagnostics/classify/pii status 200, got %d: %s", resp.StatusCode, string(resp.Body))
	}
	var document struct {
		Entities   []piiOffsetEntity `json:"entities"`
		MaskedText string            `json:"masked_text"`
	}
	if err := json.Unmarshal(resp.Body, &document); err != nil {
		return fmt.Errorf("decode /api/v1/diagnostics/classify/pii response: %w", err)
	}
	if len(document.Entities) != 1 || document.Entities[0].Type != "PERSON" {
		return fmt.Errorf("expected the one PERSON span the deployment returned, got %d: %v", len(document.Entities), document.Entities)
	}
	if err := assertRuneOffsets(text, document.Entities); err != nil {
		return err
	}
	if strings.Count(document.MaskedText, "Zorblax") != 1 || !strings.Contains(document.MaskedText, "Zorblaxian") {
		return fmt.Errorf("masked_text must hide Zorblax and keep the longer word Zorblaxian, got %q", document.MaskedText)
	}
	return nil
}
