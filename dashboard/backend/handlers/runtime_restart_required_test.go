package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"gopkg.in/yaml.v3"
)

const testDocument = "4f2a"

func restartAttempt(code, message string) *routerConfigAttempt {
	return &routerConfigAttempt{DocumentHash: testDocument, Status: "failed", Reasons: []routerConfigReason{{Code: code, Message: message}}}
}

func TestJudgeRouterAttempt(t *testing.T) {
	for _, test := range []struct {
		name            string
		response        routerConfigHash
		done, restart   bool
		verdictContains string
	}{
		{name: "the Router has not read the document", response: routerConfigHash{GeneratedRuntimeHash: "older"}},
		{name: "an older attempt", response: routerConfigHash{GeneratedRuntimeHash: testDocument, Activation: &routerConfigAttempt{DocumentHash: "older", Status: "failed"}}},
		{name: "still preparing", response: routerConfigHash{GeneratedRuntimeHash: testDocument, Activation: &routerConfigAttempt{DocumentHash: testDocument, Status: "preparing"}}},
		{name: "hot-reloaded", response: routerConfigHash{GeneratedRuntimeHash: testDocument, ActiveRuntimeHash: testDocument, Activation: &routerConfigAttempt{DocumentHash: testDocument, Status: "active"}}, done: true},
		{name: "superseded", response: routerConfigHash{GeneratedRuntimeHash: testDocument, Activation: &routerConfigAttempt{DocumentHash: testDocument, Status: "superseded"}}, done: true},
		{name: "refused for a restart", response: routerConfigHash{GeneratedRuntimeHash: testDocument, Activation: restartAttempt(restartRequiredCode, "the port changes from 8899 to 9000")}, done: true, restart: true, verdictContains: "8899 to 9000"},
		{name: "refused for another reason", response: routerConfigHash{GeneratedRuntimeHash: testDocument, Activation: restartAttempt("build_failed", "failed to connect to redis")}, done: true, verdictContains: "the Router did not apply it: failed to connect to redis"},
	} {
		t.Run(test.name, func(t *testing.T) {
			verdict, done := judgeRouterAttempt(test.response, "sha256:"+testDocument)
			_, restart := asRestartNeeded(verdict)
			if done != test.done || restart != test.restart || (verdict == nil) != (test.verdictContains == "") ||
				(verdict != nil && !strings.Contains(verdict.Error(), test.verdictContains)) {
				t.Fatalf("judge = (%v, %v), want done=%v restart=%v verdict %q", verdict, done, test.done, test.restart, test.verdictContains)
			}
		})
	}
}

func TestAwaitRouterVerdictWaitsForTheRoutersVerdict(t *testing.T) {
	var polls atomic.Int32
	router := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/config/hash" {
			http.NotFound(w, r)
			return
		}
		response := routerConfigHash{GeneratedRuntimeHash: testDocument, Activation: &routerConfigAttempt{DocumentHash: testDocument, Status: "preparing"}}
		if polls.Add(1) > 2 {
			response.Activation = restartAttempt(restartRequiredCode, "listeners[http].port: restart the Router to apply it")
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer router.Close()
	ConfigureRouterVerdict(router.URL, nil)
	defer ConfigureRouterVerdict("", nil)

	restart, ok := asRestartNeeded(awaitRouterVerdict(context.Background(), testDocument))
	if !ok || !strings.Contains(restart.detail, "listeners[http].port") || polls.Load() < 3 {
		t.Fatalf("verdict = %v after %d polls", restart, polls.Load())
	}
}

func TestWithoutARouterEverySavedConfigHotReloads(t *testing.T) {
	ConfigureRouterVerdict("", nil)
	if verdict := awaitRouterVerdict(context.Background(), testDocument); verdict != nil {
		t.Fatalf("verdict = %v with no Router configured", verdict)
	}
}

// A save the Router refuses for a reason other than a restart, such as a store
// it cannot reach, is a failed save on every save path: the previous config is
// put back and the answer carries the Router's reason instead of a success.
func TestASaveTheRouterRefusesIsNotASuccess(t *testing.T) {
	var path, served string
	router := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		data, _ := os.ReadFile(path)
		digest := strings.TrimPrefix(activationDigest(data), "sha256:")
		attempt := &routerConfigAttempt{DocumentHash: digest, Status: "active"}
		if digest != served {
			attempt.Status = "failed"
			attempt.Reasons = []routerConfigReason{{Code: "build_failed", Message: "failed to connect to redis"}}
		}
		_ = json.NewEncoder(w).Encode(routerConfigHash{GeneratedRuntimeHash: digest, ActiveRuntimeHash: served, Activation: attempt})
	}))
	defer router.Close()
	ConfigureRouterVerdict(router.URL, nil)
	defer ConfigureRouterVerdict("", nil)

	full, err := json.Marshal(canonicalConfigBody("127.0.0.1:8001"))
	if err != nil {
		t.Fatal(err)
	}
	deploy, err := json.Marshal(DeployRequest{YAML: "recipes:\n  - name: default\n    decisions:\n      - name: deployed\n        priority: 5\n" +
		"        rules: {operator: AND, conditions: [{type: domain, name: business}]}\n        modelRefs: [{model: test-model}]\n"})
	if err != nil {
		t.Fatal(err)
	}
	saves := map[string]func(path, root string) http.HandlerFunc{
		"full update":       func(path, root string) http.HandlerFunc { return UpdateConfigHandler(path, false, root) },
		"global update":     func(path, root string) http.HandlerFunc { return UpdateRouterDefaultsHandler(path, false, root) },
		"raw global update": func(path, root string) http.HandlerFunc { return UpdateGlobalConfigYAMLHandler(path, false, root) },
		"dsl deploy":        func(path, root string) http.HandlerFunc { return DeployHandler(path, false, root) },
	}
	bodies := map[string][]byte{
		"full update": full, "global update": []byte(`{"router":{"strategy":"priority"}}`),
		"raw global update": []byte("router:\n  strategy: priority\n"), "dsl deploy": deploy,
	}
	for name, save := range saves {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			path = filepath.Join(root, "config.yaml")
			previous, err := yaml.Marshal(canonicalConfigBody("127.0.0.1:8000"))
			if err != nil {
				t.Fatal(err)
			}
			if err = os.WriteFile(path, previous, 0o600); err != nil {
				t.Fatal(err)
			}
			served = strings.TrimPrefix(activationDigest(previous), "sha256:")
			response := httptest.NewRecorder()
			save(path, root)(response, httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodies[name])))
			if response.Code != http.StatusInternalServerError ||
				!strings.Contains(response.Body.String(), "the Router did not apply it: failed to connect to redis. Previous config restored.") {
				t.Fatalf("save answered %d: %s, want the Router's refusal", response.Code, response.Body.String())
			}
			if data, _ := os.ReadFile(path); !bytes.Equal(data, previous) {
				t.Fatalf("config after a refused save:\n%s\nwant the previous one back", data)
			}
		})
	}
}

func TestRecordRestartRecordsThePendingActivation(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "runtime-config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.5\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	message, err := recordRestart(configPath, envoyRestartDetail)
	if err != nil || message != "Restart required: run `vllm-sr serve` to apply." {
		t.Fatalf("recordRestart = %q, %v", message, err)
	}
	data, readErr := os.ReadFile(pendingActivationPath(configPath, pendingActivationSuffix))
	if readErr != nil {
		t.Fatalf("no pending activation recorded: %v", readErr)
	}
	var record pendingActivationRecord
	if err := json.Unmarshal(data, &record); err != nil || record.Reason != activationReasonRestart || record.Detail != envoyRestartDetail || len(record.ConfigSHA256) != 64 {
		t.Fatalf("record = %s, %v", data, err)
	}

	if err := os.WriteFile(pendingActivationPath(configPath, serveHeartbeatSuffix), []byte(`{"pid": 1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := restartRequiredMessage(configPath); got != "Restart required: `vllm-sr serve` is applying it." {
		t.Fatalf("message with an attached CLI = %q", got)
	}
}

func TestASavedConfigThatNeedsARestartIsRecordedOnce(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "runtime-config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.5\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	original := propagateConfig
	t.Cleanup(func() { propagateConfig = original })
	propagateConfig = func(string, string) error { return &restartNeededError{detail: "listener http moved"} }

	message, err := applyWrittenConfig(configPath, filepath.Dir(configPath), []byte("previous\n"), true)
	if err != nil || message != "Restart required: run `vllm-sr serve` to apply." {
		t.Fatalf("applyWrittenConfig = %q, %v", message, err)
	}
	var record pendingActivationRecord
	data, err := os.ReadFile(pendingActivationPath(configPath, pendingActivationSuffix))
	if err != nil || json.Unmarshal(data, &record) != nil || record.Detail != "listener http moved" {
		t.Fatalf("record = %s, %v", data, err)
	}
}

func TestAFailedSaveRestoresWithoutRecordingARestart(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "runtime-config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.5\n# rejected\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	original := propagateConfig
	t.Cleanup(func() { propagateConfig = original })
	calls := 0
	propagateConfig = func(string, string) error {
		calls++
		if calls == 1 {
			return errors.New("the Router rejected the config")
		}
		return &restartNeededError{detail: envoyRestartDetail}
	}

	_, err := applyWrittenConfig(configPath, filepath.Dir(configPath), []byte("version: v0.5\n"), true)
	if err == nil || !strings.Contains(formatRuntimeApplyError("Failed to apply config to runtime", err), "Previous config restored.") {
		t.Fatalf("applyWrittenConfig error = %v, want the rejection with the previous config restored", err)
	}
	if _, statErr := os.Stat(pendingActivationPath(configPath, pendingActivationSuffix)); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("restoring the previous config recorded a restart: %v", statErr)
	}
}

func TestRestartRequiredResponseIsAccepted(t *testing.T) {
	recorder := httptest.NewRecorder()
	writeRestartRequiredResponse(recorder, "20261007-1", "Restart required: run `vllm-sr serve` to apply.", nil)
	var body map[string]string
	if err := json.Unmarshal(recorder.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if recorder.Code != http.StatusAccepted || body["status"] != "restart_required" || body["version"] != "20261007-1" || !strings.HasPrefix(body["message"], "Restart required") {
		t.Fatalf("response %d %v", recorder.Code, body)
	}
}
