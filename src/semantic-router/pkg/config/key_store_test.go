package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const storedKeysConfig = `version: v0.5
listeners:
  - name: http
    port: 8899
    # clients send this key
    api_keys: [gateway-key-one, "${GATEWAY_KEY}"]
providers:
  models:
    - name: gpt
      provider_model_id: gpt-4o-mini
      api_format: chat_completions
      backend_refs:
        - name: openai
          provider: openai
          endpoint: https://api.openai.com/v1
          api_key: provider-key-one
    - name: gpt-env
      provider_model_id: gpt-4o-mini
      api_format: chat_completions
      backend_refs:
        - name: openai-env
          provider: openai
          endpoint: https://api.openai.com/v1
          api_key: ${OPENAI_API_KEY}
recipes:
  - name: default
`

func TestStoringKeysWritesReferencesAPlanCanCompare(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "provider-keys")
	save, err := StoreLiteralKeys([]byte(storedKeysConfig), KeyDir{Dir: dir}, true, true)
	if err != nil {
		t.Fatal(err)
	}
	stored := save.Data
	for _, key := range []string{"gateway-key-one", "provider-key-one"} {
		if strings.Contains(string(stored), key) {
			t.Fatalf("the stored document holds %s:\n%s", key, stored)
		}
	}
	for _, kept := range []string{"${GATEWAY_KEY}", "api_key: ${OPENAI_API_KEY}", "# clients send this key"} {
		if !strings.Contains(string(stored), kept) {
			t.Fatalf("the stored document lost %q:\n%s", kept, stored)
		}
	}
	t.Setenv("GATEWAY_KEY", "gateway-key-env")
	cfg, err := ParseYAMLBytes(stored)
	if err != nil {
		t.Fatal(err)
	}
	if keys := strings.Join(cfg.Listeners[0].ClientKeys(), ","); keys != "gateway-key-env,gateway-key-one" {
		t.Fatalf("client keys = %s", keys)
	}
	if auth := cfg.ModelConfig["gpt"].AuthoredModel.BackendRefs[0].Auth; auth == nil || filepath.Dir(auth.APIKeyFile) != dir {
		t.Fatalf("backend auth = %+v, want a file in %s", auth, dir)
	}

	// A plan stores nothing, and reads a key already stored as the save did.
	plan, err := StoreLiteralKeys([]byte(storedKeysConfig), KeyDir{Dir: dir}, false, true)
	if planned := plan.Data; err != nil || string(planned) != string(stored) {
		t.Fatalf("plan = %s (%v), want the stored document", plan.Data, err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 2 {
		t.Fatalf("key files = %v (%v), want one per key", entries, err)
	}
	fresh := strings.Replace(storedKeysConfig, "provider-key-one", "provider-key-two", 1)
	if plan, err = StoreLiteralKeys([]byte(fresh), KeyDir{Dir: dir}, false, true); err != nil || !strings.Contains(string(plan.Data), "api_key: provider-key-two") || len(plan.Kept) != 0 {
		t.Fatalf("a plan must keep a key it has not stored, without a warning: %s (%v)", plan.Data, err)
	}
}

func TestListenerKeysStayForAnEnvoyListener(t *testing.T) {
	save, err := StoreLiteralKeys([]byte(storedKeysConfig), KeyDir{Dir: t.TempDir()}, true, false)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(save.Data), "gateway-key-one") || strings.Contains(string(save.Data), "provider-key-one") {
		t.Fatalf("extproc renders listener keys into Envoy; only the backend key moves:\n%s", save.Data)
	}
}

func TestADocumentWithoutLiteralKeysComesBackAsWritten(t *testing.T) {
	document := []byte(strings.NewReplacer("api_key: provider-key-one", "auth: {api_key_env: OPENAI_API_KEY}", "gateway-key-one, ", "").Replace(storedKeysConfig))
	save, err := StoreLiteralKeys(document, KeyDir{Dir: t.TempDir()}, true, true)
	if err != nil || string(save.Data) != string(document) {
		t.Fatalf("stored = %s (%v), want the document unchanged", save.Data, err)
	}
}

// Only KeyFileGroupEnv, which `vllm-sr serve` sets, shares key files, and
// only for reading: a setgid parent alone shares nothing, and a key directory
// made before sharing was on is fixed.
func TestKeyFilesAreSharedOnlyWithTheGroupServeNames(t *testing.T) {
	for _, tc := range []struct {
		name          string
		group         int
		dirMode, mode os.FileMode
	}{
		{"private", 0, 0o700, 0o600},
		{"shared", os.Getgid(), 0o750, 0o640},
	} {
		t.Run(tc.name, func(t *testing.T) {
			parent := t.TempDir()
			if err := os.Chmod(parent, 0o770|os.ModeSetgid); err != nil {
				t.Fatal(err)
			}
			dir := filepath.Join(parent, "provider-keys")
			if err := os.Mkdir(dir, 0o700); err != nil {
				t.Fatal(err)
			}
			ref, created, err := KeyDir{Dir: dir, Group: tc.group}.Ref("provider-key-one", true)
			if err != nil || !created {
				t.Fatalf("ref = %+v, created %v (%v)", ref, created, err)
			}
			for path, want := range map[string]os.FileMode{ref.File: tc.mode, dir: tc.dirMode | os.ModeDir} {
				if info, statErr := os.Stat(path); statErr != nil || info.Mode()&(os.ModePerm|os.ModeDir) != want {
					t.Fatalf("%s mode = %v (%v), want %v", path, info.Mode(), statErr, want)
				}
			}
		})
	}
}

// The Router expands only uppercase names, so ${gateway_key} is a literal key
// and must move; a reference with a default stays; $$ is a literal $.
func TestStoringKeysReadsReferencesAsTheRouterDoes(t *testing.T) {
	document := strings.Replace(storedKeysConfig, `api_keys: [gateway-key-one, "${GATEWAY_KEY}"]`,
		`api_keys: ["${gateway_key}", "${GATEWAY_KEY:-gateway-fallback}", "pay$$key"]`, 1)
	dir := t.TempDir()
	save, err := StoreLiteralKeys([]byte(document), KeyDir{Dir: dir}, true, true)
	if err != nil {
		t.Fatal(err)
	}
	stored := save.Data
	if strings.Contains(string(stored), "${gateway_key}") || strings.Contains(string(stored), "pay$$key") || !strings.Contains(string(stored), "${GATEWAY_KEY:-gateway-fallback}") {
		t.Fatalf("stored document:\n%s", stored)
	}
	cfg, err := ParseYAMLBytes(stored)
	if err != nil {
		t.Fatal(err)
	}
	if keys := strings.Join(cfg.Listeners[0].ClientKeys(), ","); keys != "gateway-fallback,${gateway_key},pay$key" {
		t.Fatalf("client keys = %s, want what the Router read from the original", keys)
	}
}

func TestAKeyWithSurroundingSpaceIsStoredOnce(t *testing.T) {
	document := []byte(strings.Replace(storedKeysConfig, "api_key: provider-key-one", `api_key: " provider-key-one "`, 1))
	dir := t.TempDir()
	save, err := StoreLiteralKeys(document, KeyDir{Dir: dir}, true, true)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := StoreLiteralKeys(document, KeyDir{Dir: dir}, false, true)
	if err != nil || string(plan.Data) != string(save.Data) {
		t.Fatalf("plan after the save = %s (%v), want the saved document", plan.Data, err)
	}
}

func TestAnAnchoredKeyMovesWithItsAliases(t *testing.T) {
	document := strings.Replace(storedKeysConfig, "api_key: provider-key-one", "api_key: &openai provider-key-one", 1)
	document = strings.Replace(document, "api_key: ${OPENAI_API_KEY}", "api_key: *openai", 1)
	save, err := StoreLiteralKeys([]byte(document), KeyDir{Dir: filepath.Join(t.TempDir(), "keys")}, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(save.Data), "provider-key-one") {
		t.Fatalf("stored document holds the key:\n%s", save.Data)
	}
	t.Setenv("GATEWAY_KEY", "gateway-key-env")
	if _, err = ParseYAMLBytes(save.Data); err != nil {
		t.Fatalf("the stored document no longer loads: %v\n%s", err, save.Data)
	}
}

// keepingStore takes no key, as a ConfigMap target without a key Secret.
type keepingStore struct{}

func (keepingStore) Ref(string, bool) (KeyRef, bool, error) { return KeyRef{}, false, nil }
func (keepingStore) Remove(KeyRef) error                    { return nil }

func TestAKeyTheStoreCannotTakeStaysWithAWarning(t *testing.T) {
	save, err := StoreLiteralKeys([]byte(storedKeysConfig), keepingStore{}, true, true)
	if err != nil || string(save.Data) != storedKeysConfig {
		t.Fatalf("save = %s (%v), want the document as written", save.Data, err)
	}
	warnings := save.Warnings()
	if len(warnings) != 2 || warnings[0].Field != "providers.models[gpt].backend_refs[openai].api_key" || warnings[1].Field != "listeners[http].api_keys[0]" {
		t.Fatalf("warnings = %+v", warnings)
	}
	for _, warning := range warnings {
		if strings.Contains(warning.Message, "key-one") {
			t.Fatalf("a warning exposes the key: %+v", warning)
		}
	}
	if plan, _ := StoreLiteralKeys([]byte(storedKeysConfig), keepingStore{}, false, true); len(plan.Kept) != 0 {
		t.Fatal("a plan warns about a key it does not store")
	}
}

func TestADiscardedSaveRemovesOnlyTheKeysItStored(t *testing.T) {
	dir := t.TempDir()
	store := KeyDir{Dir: dir}
	earlier, _, err := store.Ref("provider-key-one", true)
	if err != nil {
		t.Fatal(err)
	}
	save, err := StoreLiteralKeys([]byte(storedKeysConfig), store, true, true)
	if err != nil {
		t.Fatal(err)
	}
	save.Discard()
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 1 || filepath.Join(dir, entries[0].Name()) != earlier.File {
		t.Fatalf("key files after the discard = %v (%v), want only the one stored before", entries, err)
	}
}
