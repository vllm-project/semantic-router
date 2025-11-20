package testcases

import (
	"context"
	"fmt"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("llmd-health-check", pkgtestcases.TestCase{
		Description: "LLM-D components readiness and basic chat call",
		Tags:        []string{"llmd", "health"},
		Fn:          llmdHealth,
	})
}

func llmdHealth(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	ns := "default"
	crdGVs := []string{
		"gateway.networking.k8s.io/v1",
		"inference.networking.k8s.io/v1",
	}
	for _, gv := range crdGVs {
		if err := ensureGroupVersion(ctx, client, gv); err != nil {
			return err
		}
	}

	required := []struct {
		ns  string
		dep string
	}{
		{"vllm-semantic-router-system", "semantic-router"},
		{"istio-system", "istiod"},
		{ns, "llm-d-inference-scheduler-llama3-8b"},
		{ns, "llm-d-inference-scheduler-phi4-mini"},
		{ns, "vllm-llama3-8b-instruct"},
		{ns, "phi4-mini"},
	}
	for _, r := range required {
		if err := waitDeploymentReady(ctx, client, r.ns, r.dep, 1); err != nil {
			return err
		}
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	res, err := doLLMDChat(ctx, localPort, "llama3-8b", "ping", 30*time.Second)
	if err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"deployments":      len(required),
			"crds":             crdGVs,
			"latency_ms":       res.duration.Milliseconds(),
			"x-selected-model": getSelectedModel(res.headers),
			"x-inference-pod":  getInferencePod(res.headers),
		})
	}
	return nil
}

func waitDeploymentReady(ctx context.Context, client *kubernetes.Clientset, ns, name string, want int32) error {
	return wait.PollUntilContextTimeout(ctx, 2*time.Second, 5*time.Minute, true, func(ctx context.Context) (bool, error) {
		dep, err := client.AppsV1().Deployments(ns).Get(ctx, name, v1.GetOptions{})
		if err != nil {
			return false, err
		}
		return dep.Status.ReadyReplicas >= want, nil
	})
}

func ensureGroupVersion(ctx context.Context, client *kubernetes.Clientset, gv string) error {
	_, err := client.Discovery().ServerResourcesForGroupVersion(gv)
	if err != nil {
		return fmt.Errorf("CRD groupVersion %s not present: %w", gv, err)
	}
	return nil
}
