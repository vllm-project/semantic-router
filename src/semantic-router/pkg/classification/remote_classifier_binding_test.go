package classification

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func TestRemoteAliasesShareAdmissionAndDrainBeforeClose(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	var calls atomic.Int32
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			close(started)
			<-release
		}
		_, _ = w.Write([]byte(`[{"label":"yes","score":0.75},{"label":"no","score":0.25}]`))
	}))
	defer func() { unblock(); server.Close() }()
	models := standaloneModelRuntime()
	external := config.ExternalModelConfig{Name: "alias-one", ModelName: "same-model", ModelEndpoint: endpointForTestServer(t, server)}
	backendCfg := &config.RemoteClassifierBackend{Model: external.Name, Contract: config.RemoteClassifierContractLabelDistribution, Protocol: config.RemoteClassifierProtocolHTTPClassify}
	spec := models.remoteSpec("domain_classifier", backendCfg)
	spec.Admission = config.AdmissionConfig{MaxConcurrency: 1, OnOverflow: "shed"}
	makeBackend := func(external config.ExternalModelConfig, spec config.ResolvedModelBinding, timeout time.Duration) *remoteSequenceBinding {
		t.Helper()
		transport, err := newHTTPClassifierInference(&external, newDeclaredLabelMapping([]string{"yes", "no"}), timeout)
		if err != nil {
			t.Fatal(err)
		}
		bound, err := prepareRemoteSequence(models, spec, &external, transport)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = bound.Close() })
		return bound
	}
	first := makeBackend(external, spec, 5*time.Second)
	external.Name, external.TimeoutSeconds, external.MaxResponseBytes = "alias-two", 20, 8192
	external.ModelName = "different-unused-model-name"
	secondSpec := spec
	secondSpec.Name, secondSpec.Recipe = "other-task", "other-recipe"
	secondSpec.Deployment.Revision = "operator-alias-only"
	second := makeBackend(external, secondSpec, 10*time.Second)
	done := make(chan error, 1)
	go func() { _, err := first.Classify(context.Background(), "hold"); done <- err }()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("request did not start")
	}
	if _, err := second.Classify(context.Background(), "overflow"); !errors.Is(err, admission.ErrQueueFull) {
		t.Fatalf("alias bypassed physical admission: %v", err)
	}
	if _, err := second.handle.Call(context.Background(), string(spec.Recipe), "foreign"); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("foreign recipe lookup: %v", err)
	}
	closed := make(chan struct{})
	go func() { _ = first.Close(); close(closed) }()
	select {
	case <-closed:
		t.Fatal("close overtook active request")
	case <-time.After(30 * time.Millisecond):
	}
	unblock()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	<-closed
	out, err := second.Classify(context.Background(), "surviving alias")
	if err != nil || len(out.Probabilities) != 2 || out.Probabilities[0] != 0.75 {
		t.Fatalf("closing one binding damaged sibling: %+v %v", out, err)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("unexpected external calls %d", got)
	}
}
