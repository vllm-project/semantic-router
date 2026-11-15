package handlers

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

// Try it reaches the listener's rules through the management transport with
// the listener's own key, which the browser never sends or sees, and the
// Router's answer comes back as it is.
func TestVSRRulesTryForwardsWithTheListenersKey(t *testing.T) {
	var forwarded systemone.ForwardRequest
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != systemone.ForwardPath || r.Header.Get("Authorization") != "Bearer trusted-management" {
			t.Errorf("unexpected management call: %s %s", r.URL.Path, r.Header.Get("Authorization"))
		}
		if err := json.NewDecoder(r.Body).Decode(&forwarded); err != nil {
			t.Fatal(err)
		}
		_, _ = io.WriteString(w, `{"revision":"r1","parts":[{"id":"try","coverage":"complete","matches":["prod_database"],"spans":[],"unchecked":[]}]}`)
	}))
	defer upstream.Close()
	keyed := writeKeyedListenerConfig(t)
	data, err := os.ReadFile(keyed)
	if err != nil {
		t.Fatal(err)
	}
	// The fixture's listener binds rules on System One; a second, keyed listener binds none.
	ruled := strings.Replace(string(data), "      - "+listenerKeyFixture+"\n",
		"      - "+listenerKeyFixture+"\n    apis:\n      systemone:\n        vsr:\n          rules:\n            recipe: default\n            guard:\n              - {type: keyword, name: prod_database}\n"+
			"  - name: internal\n    address: 0.0.0.0\n    port: 8802\n    api_keys: [internal-key]\n    apis:\n      systemone: {}\n", 1)
	if ruled == string(data) {
		t.Fatal("fixture listener not found")
	}
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(configPath, []byte(ruled), 0o600); err != nil {
		t.Fatal(err)
	}
	handler := VSRRulesTryHandler(configPath, upstream.URL, decisionModelCredentialProvider{token: "trusted-management"})

	body := `{"listener":"public","request":{"app":"guard","parts":[{"id":"try","text":"psql --host=prod-db.internal"}]}}`
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/api/vsr/rules/try", strings.NewReader(body))
	request.Header.Set("Authorization", "Bearer browser-session")
	handler(response, request)

	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"matches":["prod_database"]`) {
		t.Fatalf("try answered %d %s", response.Code, response.Body.String())
	}
	if forwarded.Listener != "public" || forwarded.Method != http.MethodPost || forwarded.Path != systemone.RulesPath ||
		forwarded.Authorization != "Bearer "+listenerKeyFixture || !strings.Contains(string(forwarded.Request), `"app":"guard"`) {
		t.Fatalf("forwarded %+v", forwarded)
	}
	if strings.Contains(response.Body.String(), listenerKeyFixture) {
		t.Fatal("the answer carries the listener's key")
	}

	for name, bad := range map[string]string{
		"unknown listener":               `{"listener":"nobody","request":{"app":"guard","parts":[]}}`,
		"a listener that binds no rules": `{"listener":"internal","request":{"app":"guard","parts":[]}}`,
		"no listener":                    `{"request":{"app":"guard","parts":[]}}`,
		"extra field":                    `{"listener":"public","request":{},"key":"x"}`,
	} {
		response := httptest.NewRecorder()
		handler(response, httptest.NewRequest(http.MethodPost, "/api/vsr/rules/try", strings.NewReader(bad)))
		if response.Code < 400 || response.Code >= 500 {
			t.Errorf("%s: %d %s", name, response.Code, response.Body.String())
		}
	}
}
