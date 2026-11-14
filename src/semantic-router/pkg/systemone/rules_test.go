package systemone

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// rulesConfig binds two keyword rules of the default recipe to guard and
// redact on a listener with a key.
func rulesConfig(pattern, enforce string) (*config.RouterConfig, *config.Listener) {
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"primary": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/vela3-lite"}}
	cfg.KeywordRules = []config.KeywordRule{
		{Name: "prod_database", Description: "touches the production database", Operator: "OR", Method: "regex", Keywords: []string{pattern}},
		{Name: "employee_id", Description: "an employee ID", Operator: "OR", Method: "regex", Keywords: []string{`\bE-\d{6}\b`}},
	}
	rules := &config.VSRRules{
		Recipe: string(config.DefaultRecipeName),
		Guard:  []config.VSRRule{{SignalRef: config.SignalRef{Type: "keyword", Name: "prod_database"}, Enforce: enforce}},
		Redact: []config.VSRRule{{SignalRef: config.SignalRef{Type: "keyword", Name: "employee_id"}}},
	}
	listener := &config.Listener{Name: "agents", APIKeys: []string{"secret"}, APIs: config.ListenerAPIs{"systemone": {
		Models: []string{"vllm-sr/vela3-lite"}, VSR: &config.VSRProfile{Rules: rules},
	}}}
	return cfg, listener
}

func listedRules(t *testing.T, cfg *config.RouterConfig, listener *config.Listener) (map[string]any, string) {
	t.Helper()
	request := httptest.NewRequest(http.MethodGet, "/v1/systemone/models", nil)
	request.Header.Set("Authorization", "Bearer secret")
	response := httptest.NewRecorder()
	Handler(cfg, listener, nil)(response, request)
	var listed struct {
		VSR struct {
			Rules map[string]any `json:"rules"`
		} `json:"vsr"`
	}
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &listed) != nil {
		t.Fatalf("listing: %d %s", response.Code, response.Body.String())
	}
	return listed.VSR.Rules, response.Body.String()
}

// The listing names each bound rule with its description and whether it is
// required, never its pattern, type or recipe, and its revision follows the
// rules' definitions and enforcement.
func TestTheListingNamesTheListenersRulesOnly(t *testing.T) {
	cfg, listener := rulesConfig(`prod-db\.internal`, "required")
	rules, body := listedRules(t, cfg, listener)
	guard, _ := json.Marshal(rules["guard"])
	if string(guard) != `[{"description":"touches the production database","name":"prod_database","required":true}]` {
		t.Fatalf("guard = %s", guard)
	}
	for _, secret := range []string{"prod-db", `E-\\d`, "keyword", `"recipe"`} {
		if strings.Contains(body, secret) {
			t.Fatalf("the listing publishes %q: %s", secret, body)
		}
	}
	revision, _ := rules["revision"].(string)
	if len(revision) != 12 {
		t.Fatalf("revision = %q, want 12 hex digits", revision)
	}
	for _, changed := range []struct{ pattern, enforce string }{{`prod-db2\.internal`, "required"}, {`prod-db\.internal`, ""}} {
		cfg, listener := rulesConfig(changed.pattern, changed.enforce)
		if other, _ := listedRules(t, cfg, listener); other["revision"] == revision {
			t.Fatalf("revision %s did not change with %+v", revision, changed)
		}
	}
}

// fakeRules answers every part with the rule names it was given, so a test
// sees what the handler passed on.
type fakeRules struct {
	asked []string
	err   error
}

func (f *fakeRules) EvaluateListenerRules(_ context.Context, listener string, rules *config.VSRRules, app string, parts []RulePart) ([]RulePartResult, error) {
	f.asked = append(f.asked, fmt.Sprintf("%s/%s/%s/%d", listener, rules.Recipe, app, len(parts)))
	if f.err != nil {
		return nil, f.err
	}
	results := make([]RulePartResult, len(parts))
	for i, part := range parts {
		results[i] = RulePartResult{ID: part.ID, Coverage: "complete", Matches: []string{"prod_database"}, Spans: []RuleSpan{}, Unchecked: []RuleUnchecked{}}
	}
	return results, nil
}

func askRules(handler http.HandlerFunc, body, key string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(http.MethodPost, RulesPath, strings.NewReader(body))
	if key != "" {
		request.Header.Set("Authorization", "Bearer "+key)
	}
	response := httptest.NewRecorder()
	handler(response, request)
	return response
}

// The rules endpoint answers on the listener's key, refuses what it cannot
// bound, and answers the evaluator's results in its own JSON, whatever schema
// the listener speaks.
func TestTheRulesEndpointServesTheListenersRules(t *testing.T) {
	const guard = `{"app":"guard","parts":[{"id":"p1","text":"psql --host=prod-db.internal"}]}`
	for _, schema := range []string{"vllm-sr/5.0", "vllm-sr/4.0"} {
		cfg, listener := rulesConfig(`prod-db\.internal`, "required")
		grant := listener.APIs["systemone"]
		grant.Schema = schema
		listener.APIs["systemone"] = grant
		evaluator := &fakeRules{}
		handler := HandlerWithRules(cfg, listener, nil, evaluator)
		response := askRules(handler, guard, "secret")
		var answered struct {
			Revision string           `json:"revision"`
			Parts    []RulePartResult `json:"parts"`
		}
		if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &answered) != nil ||
			len(answered.Revision) != 12 || len(answered.Parts) != 1 || answered.Parts[0].Matches[0] != "prod_database" {
			t.Fatalf("%s: %d %s", schema, response.Code, response.Body.String())
		}
		if strings.Join(evaluator.asked, ",") != "agents/default/guard/1" {
			t.Fatalf("%s: evaluator asked %v", schema, evaluator.asked)
		}
	}

	cfg, listener := rulesConfig(`prod-db\.internal`, "required")
	evaluator := &fakeRules{}
	handler := HandlerWithRules(cfg, listener, nil, evaluator)
	many := make([]string, maxRuleParts+1)
	for i := range many {
		many[i] = fmt.Sprintf(`{"id":"p%d","text":"x"}`, i)
	}
	for _, tc := range []struct {
		name, body, key string
		handler         http.HandlerFunc
		status          int
		code            string
	}{
		{"no key", guard, "", handler, http.StatusUnauthorized, "invalid_api_key"},
		{"wrong key", guard, "other", handler, http.StatusUnauthorized, "invalid_api_key"},
		{"no evaluator", guard, "secret", Handler(cfg, listener, nil), http.StatusNotFound, "rules_not_published"},
		{"an app that takes no rules", `{"app":"shield","parts":[{"id":"p","text":"x"}]}`, "secret", handler, http.StatusBadRequest, "invalid_request"},
		{"no parts", `{"app":"guard","parts":[]}`, "secret", handler, http.StatusBadRequest, "invalid_request"},
		{"repeated id", `{"app":"guard","parts":[{"id":"p","text":"a"},{"id":"p","text":"b"}]}`, "secret", handler, http.StatusBadRequest, "invalid_request"},
		{"unknown field", `{"app":"guard","parts":[{"id":"p","text":"a","rule":"x"}]}`, "secret", handler, http.StatusBadRequest, "invalid_request"},
		{"a value after the request", `{"app":"guard","parts":[{"id":"p","text":"a"}]}]`, "secret", handler, http.StatusBadRequest, "invalid_request"},
		{"a second request", `{"app":"guard","parts":[{"id":"p","text":"a"}]} {"app":"redact","parts":[{"id":"q","text":"b"}]}`, "secret", handler, http.StatusBadRequest, "invalid_request"},
		{"too many parts", `{"app":"guard","parts":[` + strings.Join(many, ",") + `]}`, "secret", handler, http.StatusRequestEntityTooLarge, "request_too_large"},
		{"too much text", `{"app":"guard","parts":[{"id":"p","text":"` + strings.Repeat("a", maxRuleText+1) + `"}]}`, "secret", handler, http.StatusRequestEntityTooLarge, "request_too_large"},
	} {
		response := askRules(tc.handler, tc.body, tc.key)
		var refused struct {
			Error struct {
				Code string `json:"code"`
			} `json:"error"`
		}
		if response.Code != tc.status || json.Unmarshal(response.Body.Bytes(), &refused) != nil || refused.Error.Code != tc.code {
			t.Errorf("%s: %d %s, want %d %s", tc.name, response.Code, response.Body.String(), tc.status, tc.code)
		}
	}
	if len(evaluator.asked) != 0 {
		t.Fatalf("a refused request reached the evaluator: %v", evaluator.asked)
	}

	unbound, listenerWithoutRedact := rulesConfig(`prod-db\.internal`, "")
	listenerWithoutRedact.APIs["systemone"].VSR.Rules.Redact = nil
	if response := askRules(HandlerWithRules(unbound, listenerWithoutRedact, nil, evaluator), `{"app":"redact","parts":[{"id":"p","text":"x"}]}`, "secret"); response.Code != http.StatusNotFound {
		t.Fatalf("redact with no rules: %d %s", response.Code, response.Body.String())
	}

	slots := rulesSlots(listener.Name)
	for range maxRulesInFlight {
		slots <- struct{}{}
	}
	busy := askRules(handler, guard, "secret")
	for range maxRulesInFlight {
		<-slots
	}
	if busy.Code != http.StatusTooManyRequests || busy.Header().Get("Retry-After") == "" {
		t.Fatalf("a listener at its limit answered %d %s", busy.Code, busy.Body.String())
	}

	// The evaluator's error names the request's text; the answer must not.
	failing := HandlerWithRules(cfg, listener, nil, &fakeRules{err: errors.New("could not read psql --host=prod-db.internal")})
	if response := askRules(failing, guard, "secret"); response.Code != http.StatusServiceUnavailable || strings.Contains(response.Body.String(), "prod-db") {
		t.Fatalf("an evaluator error answered %d %s", response.Code, response.Body.String())
	}
}
