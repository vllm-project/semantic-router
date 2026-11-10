package testcases

import (
	"context"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("pii-long-text", pkgtestcases.TestCase{
		Description: "Verify /api/v1/diagnostics/classify/pii detects entities past the classifier's sequence limit",
		Tags:        []string{"kubernetes", "apiserver", "classification", "pii", "api"},
		Fn:          testPIILongText,
	})
}

// A model reads at most its own sequence limit, so a call that sent only the
// start of a long text would never see its end. This drives the endpoint with
// the entity placed past any one sequence window; the profile's System One
// fixture reads all of what it is sent, so it finds the marked entity only if
// the Router sends the whole text.
func testPIILongText(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	httpClient := session.HTTPClient(60 * time.Second)
	url := session.URL("/api/v1/diagnostics/classify/pii")

	// Filler carries no names, numbers or dates, so any entity reported here
	// belongs to the trailing secret.
	sentences := []string{
		"Sailors used the stars to navigate before mechanical instruments existed. ",
		"The compass then made direction independent of a clear night sky. ",
		"Radio beacons later fixed a position without any view of the horizon. ",
		"Satellite systems eventually replaced every one of the earlier techniques. ",
	}
	var builder strings.Builder
	for i := 0; i < 96; i++ {
		builder.WriteString(sentences[i%len(sentences)])
	}
	filler := builder.String()

	const secret = "Contact John Doe at john.doe@example.com or 555-123-4567. __PII_SPAN__ EMAIL_ADDRESS john.doe@example.com"
	text := filler + secret
	secretStart := utf8.RuneCountInString(filler)

	entities, err := classifyPIIWithPositions(ctx, httpClient, url, text)
	if err != nil {
		return err
	}
	if len(entities) == 0 {
		return fmt.Errorf(
			"expected PII past the classifier sequence limit to be detected in a %d-rune text, got none",
			utf8.RuneCountInString(text))
	}
	if err := assertRuneOffsets(text, entities); err != nil {
		return err
	}

	for _, entity := range entities {
		if *entity.StartPos >= secretStart {
			return nil
		}
	}
	return fmt.Errorf(
		"expected an entity at or past rune %d, where the trailing secret starts, got %v",
		secretStart, entities)
}
