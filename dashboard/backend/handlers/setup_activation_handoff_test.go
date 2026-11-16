package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

type setupHandoffResponseWriter struct {
	*httptest.ResponseRecorder
	beforeResponse func()
}

func (w *setupHandoffResponseWriter) Header() http.Header {
	w.beforeResponse()
	return w.ResponseRecorder.Header()
}

func TestSetupActivationKeepsItsAcknowledgementAfterTheCLIStarts(t *testing.T) {
	configPath, root, containerCLICalls := setupActivationRegressionRuntime(t)
	t.Setenv("VLLM_SR_GATEWAY", "standalone")
	heartbeat := pendingActivationPath(configPath, serveHeartbeatSuffix)
	if err := os.WriteFile(heartbeat, []byte(`{"pid":1,"state":"waiting"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	started := false
	w := &setupHandoffResponseWriter{ResponseRecorder: httptest.NewRecorder()}
	w.beforeResponse = func() {
		if !started && pendingActivationRecorded(configPath) {
			if err := os.WriteFile(heartbeat, []byte(`{"pid":1,"state":"starting"}`), 0o644); err != nil {
				t.Fatal(err)
			}
			started = true
		}
	}
	body := mustJSONRaw(t, SetupConfigRequest{Config: mustJSONRaw(t, createValidSetupPatch())})
	asSecretReader(SetupActivateHandler(configPath, false, root, setupmode.New(configPath, false)))(
		w, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body)))

	if !started || w.Code != http.StatusOK {
		t.Fatalf("activation did not hand off to the CLI: started=%v, status=%d, body=%s", started, w.Code, w.Body.String())
	}
	var response SetupActivateResponse
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.Message != "Setup saved. The Router is starting." {
		t.Fatalf("the CLI already started, but activation returned %q", response.Message)
	}
	if calls := containerCLICalls(); len(calls) != 0 {
		t.Fatalf("setup ran a container CLI: %q", calls)
	}
}
