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
	pkgtestcases.Register("pii-repeated-value-masking", pkgtestcases.TestCase{
		Description: "Verify masked_text hides every copy of a PII value the backend labelled only once",
		Tags:        []string{"kubernetes", "apiserver", "classification", "pii", "api"},
		Fn:          testPIIRepeatedValueMasking,
	})
}

// testPIIRepeatedValueMasking asks the PII API to mask a text that names the
// same person twice. The profile's token_spans.v1 stub labels only the first
// copy of the marked word, which is what Vela 2.0 does with a repeated name,
// so the second copy is masked only if the router covers it.
func testPIIRepeatedValueMasking(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	text := "__PII_SPAN__ PERSON Zorblax 様が来ました. Later Zorblax replied, and Zorblaxian food was served."
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

	if len(document.Entities) != 2 {
		return fmt.Errorf("expected both copies of Zorblax as entities, got %d: %v", len(document.Entities), document.Entities)
	}
	if err := assertRuneOffsets(text, document.Entities); err != nil {
		return err
	}
	if strings.Count(document.MaskedText, "Zorblax") != 1 || !strings.Contains(document.MaskedText, "Zorblaxian") {
		return fmt.Errorf("masked_text must hide both copies of Zorblax and keep the longer word Zorblaxian, got %q", document.MaskedText)
	}
	return nil
}
