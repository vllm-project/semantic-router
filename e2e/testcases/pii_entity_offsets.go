package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("pii-entity-offsets", pkgtestcases.TestCase{
		Description: "Verify /api/v1/classify/pii reports entity positions as code-point offsets",
		Tags:        []string{"kubernetes", "apiserver", "classification", "pii", "api"},
		Fn:          testPIIEntityOffsets,
	})
}

type piiOffsetEntity struct {
	Type     string `json:"type"`
	Value    string `json:"value"`
	StartPos *int   `json:"start_position"`
	EndPos   *int   `json:"end_position"`
}

type piiOffsetResponse struct {
	Entities []piiOffsetEntity `json:"entities"`
}

func testPIIEntityOffsets(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	httpClient := session.HTTPClient(30 * time.Second)
	url := session.URL("/api/v1/classify/pii")

	// Multi-byte text: byte offsets would run past the end of the string.
	nonASCII := "こんにちは、私は John Smith です。電話は 415-555-0134"
	entities, err := classifyPIIWithPositions(ctx, httpClient, url, nonASCII)
	if err != nil {
		return err
	}
	if len(entities) == 0 {
		return fmt.Errorf("expected at least one PII entity in %q", nonASCII)
	}
	if err := assertRuneOffsets(nonASCII, entities); err != nil {
		return err
	}

	// An entity at the start of the text has to report start_position 0.
	atZero := "John Smith called yesterday."
	entities, err = classifyPIIWithPositions(ctx, httpClient, url, atZero)
	if err != nil {
		return err
	}
	if err := assertRuneOffsets(atZero, entities); err != nil {
		return err
	}
	for _, entity := range entities {
		if *entity.StartPos == 0 {
			return nil
		}
	}
	return fmt.Errorf("expected an entity at start_position 0 in %q, got %v", atZero, entities)
}

func classifyPIIWithPositions(
	ctx context.Context,
	httpClient *http.Client,
	url string,
	text string,
) ([]piiOffsetEntity, error) {
	body, err := json.Marshal(map[string]interface{}{
		"text": text,
		"options": map[string]interface{}{
			"return_positions":     true,
			"reveal_entity_text":   true,
			"confidence_threshold": 0.3,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("marshal /api/v1/classify/pii payload: %w", err)
	}

	resp, err := postJSON(ctx, httpClient, http.MethodPost, url, body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("expected /api/v1/classify/pii status 200, got %d: %s", resp.StatusCode, string(resp.Body))
	}

	var document piiOffsetResponse
	if err := json.Unmarshal(resp.Body, &document); err != nil {
		return nil, fmt.Errorf("decode /api/v1/classify/pii response: %w", err)
	}
	return document.Entities, nil
}

// assertRuneOffsets slices text by the reported positions, treating them as
// code-point offsets, and checks the result matches the entity value.
func assertRuneOffsets(text string, entities []piiOffsetEntity) error {
	runes := []rune(text)
	for _, entity := range entities {
		if entity.StartPos == nil || entity.EndPos == nil {
			return fmt.Errorf("%s: positions were requested but not returned", entity.Type)
		}
		start, end := *entity.StartPos, *entity.EndPos
		if start < 0 || end > len(runes) || start > end {
			return fmt.Errorf("%s: span [%d,%d) is not within a %d code-point string", entity.Type, start, end, len(runes))
		}
		if got := string(runes[start:end]); got != entity.Value {
			return fmt.Errorf("%s: slicing by the reported offsets gave %q, want %q", entity.Type, got, entity.Value)
		}
	}
	return nil
}
