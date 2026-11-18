package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// The agents listener of the standalone profile binds keyword rules to the
// vSR plugins' guard and redact.
const (
	standaloneRulesService = "semantic-router"
	standaloneRulesPort    = "8898"
	standaloneRulesKey     = "e2e-rules-key"
)

func init() {
	pkgtestcases.Register("standalone-listener-rules", pkgtestcases.TestCase{
		Description: "A listener's listing names its guard and redact rules without their patterns, POST /v1/systemone/rules matches guard rules and returns merged redact spans in code points, and a call without the key is refused",
		Tags:        []string{"standalone", "systemone", "vsr", "rules"},
		Fn:          testStandaloneListenerRules,
	})
}

type listenerRulesAnswer struct {
	Revision string `json:"revision"`
	Parts    []struct {
		ID       string   `json:"id"`
		Coverage string   `json:"coverage"`
		Matches  []string `json:"matches"`
		Spans    []struct {
			Start int    `json:"start"`
			End   int    `json:"end"`
			Rule  string `json:"rule"`
		} `json:"spans"`
		Unchecked []json.RawMessage `json:"unchecked"`
	} `json:"parts"`
}

func testStandaloneListenerRules(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceEndpointSession(ctx, client, opts, standaloneNamespace, standaloneRulesService, standaloneRulesPort)
	if err != nil {
		return err
	}
	defer session.Close()
	client30 := session.HTTPClient(30 * time.Second)
	key := map[string]string{"Authorization": "Bearer " + standaloneRulesKey}

	listing, err := fixtures.DoGETRequestWithHeaders(ctx, client30, session.URL("/v1/systemone/models"), key)
	if err != nil {
		return err
	}
	var listed struct {
		VSR struct {
			Rules struct {
				Revision string `json:"revision"`
				Guard    []struct {
					Name     string `json:"name"`
					Required bool   `json:"required"`
				} `json:"guard"`
				Redact []struct {
					Name string `json:"name"`
				} `json:"redact"`
			} `json:"rules"`
		} `json:"vsr"`
	}
	if listing.StatusCode != 200 || listing.DecodeJSON(&listed) != nil || len(listed.VSR.Rules.Guard) != 1 ||
		listed.VSR.Rules.Guard[0].Name != "prod_database" || !listed.VSR.Rules.Guard[0].Required || len(listed.VSR.Rules.Redact) != 2 {
		return fmt.Errorf("the listing does not name the bound rules: %d %s", listing.StatusCode, truncateString(string(listing.Body), 600))
	}
	if strings.Contains(string(listing.Body), "prod-db") || strings.Contains(string(listing.Body), `"recipe"`) {
		return fmt.Errorf("the listing publishes a rule's pattern or recipe: %s", truncateString(string(listing.Body), 600))
	}

	guard, err := askListenerRules(ctx, session, key, "guard", "psql --host=prod-db.internal -c 'select 1'", "ls -la")
	if err != nil {
		return err
	}
	if guard.Revision != listed.VSR.Rules.Revision || len(guard.Parts) != 2 || strings.Join(guard.Parts[0].Matches, ",") != "prod_database" ||
		len(guard.Parts[1].Matches) != 0 || guard.Parts[0].Coverage != "complete" {
		return fmt.Errorf("guard answered %+v, want prod_database on the first part only under revision %s", guard, listed.VSR.Rules.Revision)
	}

	redact, err := askListenerRules(ctx, session, key, "redact", "계정 E-104233 paid 99887766")
	if err != nil {
		return err
	}
	spans, _ := json.Marshal(redact.Parts[0].Spans)
	if string(spans) != `[{"start":3,"end":11,"rule":"employee_id"},{"start":17,"end":25,"rule":"long_number"}]` {
		return fmt.Errorf("redact spans = %s, want the employee ID and the number in code points, the overlap merged", spans)
	}

	refused, err := fixtures.DoPOSTRequest(ctx, client30, session.URL("/v1/systemone/rules"), map[string]any{
		"app": "guard", "parts": []map[string]string{{"id": "p", "text": "ls"}},
	})
	if err != nil {
		return err
	}
	if refused.StatusCode != 401 {
		return fmt.Errorf("a rules call without the listener's key answered %d", refused.StatusCode)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"revision": guard.Revision, "redact_spans": string(spans)})
	}
	return nil
}

func askListenerRules(ctx context.Context, session *fixtures.ServiceSession, key map[string]string, app string, texts ...string) (listenerRulesAnswer, error) {
	parts := make([]map[string]string, len(texts))
	for i, text := range texts {
		parts[i] = map[string]string{"id": fmt.Sprintf("p%d", i+1), "text": text}
	}
	var answer listenerRulesAnswer
	response, err := fixtures.DoPOSTRequestWithHeaders(ctx, session.HTTPClient(30*time.Second), session.URL("/v1/systemone/rules"),
		map[string]any{"app": app, "parts": parts}, key)
	if err != nil {
		return answer, err
	}
	if response.StatusCode != 200 || response.DecodeJSON(&answer) != nil {
		return answer, fmt.Errorf("POST /v1/systemone/rules %s answered %d: %s", app, response.StatusCode, truncateString(string(response.Body), 600))
	}
	return answer, nil
}
