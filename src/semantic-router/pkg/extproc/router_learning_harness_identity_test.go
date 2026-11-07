package extproc

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type harnessSupportManifest struct {
	Clients []harnessSupportClient `yaml:"clients"`
}

type harnessSupportClient struct {
	Client   string                  `yaml:"client"`
	Versions []harnessSupportVersion `yaml:"versions"`
}

type harnessSupportVersion struct {
	Version  string                 `yaml:"version"`
	Level    string                 `yaml:"level"`
	Identity []harnessSupportHeader `yaml:"identity"`
	Evidence struct {
		Captures []string `yaml:"captures"`
	} `yaml:"evidence"`
}

type harnessSupportHeader struct {
	Name        string   `yaml:"name"`
	Role        string   `yaml:"role"`
	Placeholder string   `yaml:"placeholder"`
	On          []string `yaml:"on"`
}

type harnessCaptureIdentity struct {
	Client   string `json:"client"`
	Version  string `json:"version"`
	Identity struct {
		Headers []struct {
			Name        string   `json:"name"`
			Placeholder string   `json:"placeholder"`
			On          []string `json:"on"`
		} `json:"headers"`
	} `json:"identity"`
	Subagent json.RawMessage `json:"subagent"`
}

func TestProtectionIdentityGroupsHarnessCaptures(t *testing.T) {
	root := harnessSupportRepoRoot(t)
	manifest := loadHarnessSupportManifest(t, root)
	agentHeader := (config.RouterLearningProtectionConfig{}).HeaderName("agent")
	for _, client := range manifest.Clients {
		for _, version := range client.Versions {
			if version.Level == "R3" && agentHeader == "" {
				t.Fatalf("%s %s claims R3 but protection has no identity.headers.agent", client.Client, version.Version)
			}
			for _, name := range version.Evidence.Captures {
				capture := loadHarnessCapture(t, root, name)
				if capture.Client != client.Client || capture.Version != version.Version {
					t.Fatalf("%s records %s %s, manifest says %s %s", name, capture.Client, capture.Version, client.Client, version.Version)
				}
				if err := matchHarnessIdentity(version.Identity, capture); err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				assertHarnessProtectionLevel(t, version, capture)
			}
		}
	}
}

func TestProtectionIdentitySessionAndConversationKeys(t *testing.T) {
	session := "x-session-id"
	conversation := "x-conversation-id"
	parent := map[string]string{session: "placeholder-session", conversation: "placeholder-conversation"}
	same := map[string]string{session: "placeholder-session", conversation: "placeholder-conversation"}
	otherConversation := map[string]string{session: "placeholder-session", conversation: "placeholder-conversation-other"}

	conversationKey := protectionMemoryKey(t, config.RouterLearningScopeConversation, session, conversation, parent)
	if protectionMemoryKey(t, config.RouterLearningScopeConversation, session, conversation, same) != conversationKey {
		t.Fatal("the same session and conversation did not share a protection key")
	}
	if protectionMemoryKey(t, config.RouterLearningScopeConversation, session, conversation, otherConversation) == conversationKey {
		t.Fatal("a different conversation shared the protection key")
	}

	subagent := map[string]string{
		session:                  "placeholder-session",
		"x-claude-code-agent-id": "placeholder-agent",
	}
	sessionKey := protectionMemoryKey(t, config.RouterLearningScopeSession, session, conversation, parent)
	if protectionMemoryKey(t, config.RouterLearningScopeSession, session, conversation, subagent) != sessionKey {
		t.Fatal("a subagent agent header split a session-scoped protection key")
	}
	otherSession := map[string]string{session: "placeholder-session-other"}
	if protectionMemoryKey(t, config.RouterLearningScopeSession, session, conversation, otherSession) == sessionKey {
		t.Fatal("a different session shared the protection key")
	}
}

func assertHarnessProtectionLevel(t *testing.T, version harnessSupportVersion, capture harnessCaptureIdentity) {
	t.Helper()
	turn0 := harnessHeadersForRequest(capture, "turn-0")
	turn1 := harnessHeadersForRequest(capture, "turn-1")
	switch version.Level {
	case "R0":
		if len(version.Identity) != 0 || len(capture.Identity.Headers) != 0 {
			t.Fatalf("%s %s is R0 but records identity headers", capture.Client, capture.Version)
		}
		if _, ok := harnessProtectionIdentity(config.RouterLearningScopeConversation, "", "", turn0); ok {
			t.Fatal("R0 turn-0 formed a protection key")
		}
		if _, ok := harnessProtectionIdentity(config.RouterLearningScopeConversation, "", "", turn1); ok {
			t.Fatal("R0 turn-1 formed a protection key")
		}
	case "R1":
		session, conversation := harnessRoleHeader(t, version, "session"), harnessRoleHeader(t, version, "conversation")
		key0 := protectionMemoryKey(t, config.RouterLearningScopeConversation, session.Name, conversation.Name, turn0)
		key1 := protectionMemoryKey(t, config.RouterLearningScopeConversation, session.Name, conversation.Name, turn1)
		if key0 != key1 {
			t.Fatalf("R1 turns did not share a protection key: %q vs %q", key0, key1)
		}
		changed := copyHarnessHeaders(turn0)
		changed[conversation.Name] = conversation.Placeholder + "-other"
		if protectionMemoryKey(t, config.RouterLearningScopeConversation, session.Name, conversation.Name, changed) == key0 {
			t.Fatal("R1 conversation change kept the same protection key")
		}
	case "R2":
		session := harnessRoleHeader(t, version, "session")
		key0 := protectionMemoryKey(t, config.RouterLearningScopeSession, session.Name, "", turn0)
		key1 := protectionMemoryKey(t, config.RouterLearningScopeSession, session.Name, "", turn1)
		if key0 != key1 {
			t.Fatalf("R2 turns did not share a protection key: %q vs %q", key0, key1)
		}
		for _, request := range session.On {
			if request != "subagent" {
				continue
			}
			subagent := harnessHeadersForRequest(capture, "subagent")
			if protectionMemoryKey(t, config.RouterLearningScopeSession, session.Name, "", subagent) != key0 {
				t.Fatal("R2 subagent did not share the session protection key")
			}
		}
	case "R3":
		session := harnessRoleHeader(t, version, "session")
		agent := harnessRoleHeader(t, version, "agent")
		parent := harnessHeadersForRequest(capture, "turn-0")
		child := copyHarnessHeaders(parent)
		child[agent.Name] = agent.Placeholder + "-other"
		parentKey := protectionMemoryKey(t, config.RouterLearningScopeSession, session.Name, "", parent)
		childKey := protectionMemoryKey(t, config.RouterLearningScopeSession, session.Name, "", child)
		if parentKey == childKey {
			t.Fatal("R3 agent change kept the same protection key")
		}
	default:
		t.Fatalf("unknown harness level %q", version.Level)
	}
}

func protectionMemoryKey(t *testing.T, scope, sessionHeader, conversationHeader string, headers map[string]string) string {
	t.Helper()
	key, ok := harnessProtectionIdentity(scope, sessionHeader, conversationHeader, headers)
	if !ok || key == "" {
		t.Fatalf("protection identity missing for scope %s headers %#v", scope, headers)
	}
	return key
}

func harnessProtectionIdentity(scope, sessionHeader, conversationHeader string, headers map[string]string) (string, bool) {
	cfg := config.RouterLearningProtectionConfig{Scope: scope}
	if sessionHeader != "" {
		cfg.Identity.Headers.Session = &sessionHeader
	}
	if conversationHeader != "" {
		cfg.Identity.Headers.Conversation = &conversationHeader
	}
	identity, ok := (&OpenAIRouter{}).protectionIdentity(&RequestContext{Headers: headers}, cfg)
	if !ok {
		return "", false
	}
	return identity.memoryKey, true
}

func harnessRoleHeader(t *testing.T, version harnessSupportVersion, role string) harnessSupportHeader {
	t.Helper()
	var found *harnessSupportHeader
	for i := range version.Identity {
		if version.Identity[i].Role != role {
			continue
		}
		if found != nil {
			t.Fatalf("duplicate %s identity header", role)
		}
		found = &version.Identity[i]
	}
	if found == nil {
		t.Fatalf("level %s is missing the %s header", version.Level, role)
	}
	return *found
}

func harnessHeadersForRequest(capture harnessCaptureIdentity, request string) map[string]string {
	headers := make(map[string]string)
	for _, header := range capture.Identity.Headers {
		for _, on := range header.On {
			if on == request {
				headers[header.Name] = header.Placeholder
			}
		}
	}
	return headers
}

func copyHarnessHeaders(headers map[string]string) map[string]string {
	copied := make(map[string]string, len(headers))
	for key, value := range headers {
		copied[key] = value
	}
	return copied
}

func matchHarnessIdentity(manifest []harnessSupportHeader, capture harnessCaptureIdentity) error {
	if len(manifest) != len(capture.Identity.Headers) {
		return errHarnessIdentityMismatch
	}
	recorded := make(map[string]struct {
		Placeholder string
		On          []string
	}, len(capture.Identity.Headers))
	for _, header := range capture.Identity.Headers {
		recorded[header.Name] = struct {
			Placeholder string
			On          []string
		}{header.Placeholder, header.On}
	}
	for _, header := range manifest {
		got, ok := recorded[header.Name]
		if !ok || got.Placeholder != header.Placeholder || !reflect.DeepEqual(got.On, header.On) {
			return errHarnessIdentityMismatch
		}
	}
	return nil
}

var errHarnessIdentityMismatch = harnessIdentityError("capture identity does not match the manifest")

type harnessIdentityError string

func (err harnessIdentityError) Error() string { return string(err) }

func loadHarnessSupportManifest(t *testing.T, root string) harnessSupportManifest {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(root, "e2e", "testcases", "testdata", "agent_clients", "harness-support.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest harnessSupportManifest
	if err := yaml.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	return manifest
}

func loadHarnessCapture(t *testing.T, root, name string) harnessCaptureIdentity {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(root, "e2e", "testcases", "testdata", "agent_clients", name))
	if err != nil {
		t.Fatal(err)
	}
	var capture harnessCaptureIdentity
	if err := json.Unmarshal(data, &capture); err != nil {
		t.Fatal(err)
	}
	return capture
}

func harnessSupportRepoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", "..", ".."))
}
