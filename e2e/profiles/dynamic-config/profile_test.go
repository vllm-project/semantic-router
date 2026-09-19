package dynamicconfig

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
)

type recordingInstaller struct{ options helm.InstallOptions }

func (r *recordingInstaller) Install(_ context.Context, options helm.InstallOptions) error {
	r.options = options
	return nil
}

func TestBootstrapDoesNotWaitForUnpublishedCRs(t *testing.T) {
	installer := &recordingInstaller{}
	if err := NewProfile().deploySemanticRouter(context.Background(), installer, &framework.SetupOptions{ImageTag: "test"}); err != nil {
		t.Fatal(err)
	}
	if installer.options.Wait {
		t.Fatal("Helm readiness wait before CR publication deadlocks startup")
	}
	if installer.options.Namespace != routerNamespace {
		t.Fatal(installer.options.Namespace)
	}
	for _, manifest := range []string{"intelligentpool.yaml", "intelligentroute.yaml"} {
		data, err := os.ReadFile("crds/" + manifest)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(data), "namespace: "+routerNamespace+"\n") {
			t.Fatalf("%s is outside the Router namespace", manifest)
		}
	}
}

func TestWaitForActivatedCRsRequiresCurrentGenerationOfBothResources(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	calls := 0
	err := waitForActivatedCRs(ctx, time.Millisecond, func(_ context.Context, resource string) ([]byte, error) {
		calls++
		if strings.HasPrefix(resource, "intelligentroute/") && calls < 4 {
			return []byte(`{"metadata":{"generation":2},"status":{"conditions":[{"type":"Ready","status":"True","observedGeneration":1}]}}`), nil
		}
		return []byte(`{"metadata":{"generation":2},"status":{"conditions":[{"type":"Ready","status":"True","observedGeneration":2}]}}`), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls < 4 {
		t.Fatal("stale Ready condition was accepted")
	}
}

func TestWaitForActivatedCRsCancels(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := waitForActivatedCRs(ctx, time.Hour, func(context.Context, string) ([]byte, error) { return nil, errors.New("not ready") })
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("wait = %v", err)
	}
}
