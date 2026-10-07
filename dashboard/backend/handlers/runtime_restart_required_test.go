package handlers

import (
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
)

const testDocument = "4f2a"

func restartAttempt(code, message string) *routerConfigAttempt {
	return &routerConfigAttempt{DocumentHash: testDocument, Status: "failed", Reasons: []routerConfigReason{{Code: code, Message: message}}}
}

func TestJudgeRouterAttempt(t *testing.T) {
	for _, test := range []struct {
		name           string
		response       routerConfigHash
		done, restart  bool
		detailContains string
	}{
		{name: "the Router has not read the document", response: routerConfigHash{GeneratedRuntimeHash: "older"}},
		{name: "an older attempt", response: routerConfigHash{GeneratedRuntimeHash: testDocument, Activation: &routerConfigAttempt{DocumentHash: "older", Status: "failed"}}},
		{name: "still preparing", response: routerConfigHash{GeneratedRuntimeHash: testDocument, Activation: &routerConfigAttempt{DocumentHash: testDocument, Status: "preparing"}}},
		{name: "hot-reloaded", response: routerConfigHash{GeneratedRuntimeHash: testDocument, ActiveRuntimeHash: testDocument, Activation: &routerConfigAttempt{DocumentHash: testDocument, Status: "active"}}, done: true},
		{name: "refused for a restart", response: routerConfigHash{GeneratedRuntimeHash: testDocument, Activation: restartAttempt(restartRequiredCode, "the port changes from 8899 to 9000")}, done: true, restart: true, detailContains: "8899 to 9000"},
		{name: "refused for another reason", response: routerConfigHash{GeneratedRuntimeHash: testDocument, Activation: restartAttempt("unsupported", "no")}, done: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			detail, done, restart := judgeRouterAttempt(test.response, "sha256:"+testDocument)
			if done != test.done || restart != test.restart || !strings.Contains(detail, test.detailContains) {
				t.Fatalf("judge = (%q, %v, %v), want done=%v restart=%v", detail, done, restart, test.done, test.restart)
			}
		})
	}
}

func TestRouterRestartReasonWaitsForTheRoutersVerdict(t *testing.T) {
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

	detail, restart := routerRestartReason(context.Background(), testDocument)
	if !restart || !strings.Contains(detail, "listeners[http].port") || polls.Load() < 3 {
		t.Fatalf("verdict = (%q, %v) after %d polls", detail, restart, polls.Load())
	}
}

func TestWithoutARouterEverySavedConfigHotReloads(t *testing.T) {
	ConfigureRouterVerdict("", nil)
	if detail, restart := routerRestartReason(context.Background(), testDocument); restart || detail != "" {
		t.Fatalf("verdict = (%q, %v) with no Router configured", detail, restart)
	}
}

func TestRecordRestartRecordsThePendingActivation(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "runtime-config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\n"), 0o644); err != nil {
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
	if err := os.WriteFile(configPath, []byte("version: v0.3\n"), 0o644); err != nil {
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
	if err := os.WriteFile(configPath, []byte("version: v0.3\n# rejected\n"), 0o644); err != nil {
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

	_, err := applyWrittenConfig(configPath, filepath.Dir(configPath), []byte("version: v0.3\n"), true)
	if err == nil || !strings.Contains(formatRuntimeApplyError("Failed to apply config to runtime", err), "Previous config restored.") {
		t.Fatalf("applyWrittenConfig error = %v, want the rejection with the previous config restored", err)
	}
	if _, statErr := os.Stat(pendingActivationPath(configPath, pendingActivationSuffix)); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("restoring the previous config recorded a restart: %v", statErr)
	}
}

func TestRestartRequiredResponseIsAccepted(t *testing.T) {
	recorder := httptest.NewRecorder()
	writeRestartRequiredResponse(recorder, "20261007-1", "Restart required: run `vllm-sr serve` to apply.")
	var body map[string]string
	if err := json.Unmarshal(recorder.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if recorder.Code != http.StatusAccepted || body["status"] != "restart_required" || body["version"] != "20261007-1" || !strings.HasPrefix(body["message"], "Restart required") {
		t.Fatalf("response %d %v", recorder.Code, body)
	}
}
