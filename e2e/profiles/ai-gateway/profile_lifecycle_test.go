package aigateway

import (
	"context"
	"errors"
	"strings"
	"testing"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
)

func TestProfileTeardownRefusesUnownedAliasAndStillCleansStack(t *testing.T) {
	const namespace = "router-system"
	stackCleanupErr := errors.New("stack cleanup failed")
	stack := &memoryGatewayStack{teardownErr: stackCleanupErr}
	services := &memoryLooperGatewayServices{service: looperGatewayAliasFixture(
		namespace,
		"generated-gateway.envoy-system.svc.cluster.local",
		false,
	)}
	profile := &Profile{
		stack: stack,
		servicesForNamespace: func(*kubernetes.Clientset, string) looperGatewayServices {
			return services
		},
	}

	err := profile.Teardown(context.Background(), &framework.TeardownOptions{
		KubeClient: &kubernetes.Clientset{},
	})
	if err == nil || !strings.Contains(err.Error(), "refusing to delete unowned") ||
		!errors.Is(err, stackCleanupErr) {
		t.Fatalf("Teardown() error = %v, want joined ownership refusal and stack error", err)
	}
	if services.deleteCalls != 0 || services.service == nil {
		t.Fatalf(
			"unowned alias state = delete calls:%d service:%v, want preserved",
			services.deleteCalls,
			services.service,
		)
	}
	if stack.teardownCalls != 1 {
		t.Fatalf("stack teardown calls = %d, want 1", stack.teardownCalls)
	}
}

func TestProfileSetupRollsBackStackWhenAliasResolutionFails(t *testing.T) {
	resolveErr := errors.New("gateway lookup failed")
	cleanupErr := errors.New("stack cleanup failed")
	stack := &memoryGatewayStack{
		serviceConfig: framework.ServiceConfig{
			Namespace:     "envoy-system",
			LabelSelector: "gateway=semantic-router",
		},
		teardownErr: cleanupErr,
	}
	services := &memoryLooperGatewayServices{}
	profile := &Profile{
		stack: stack,
		resolveGatewayService: func(
			context.Context,
			*kubernetes.Clientset,
			string,
			string,
			bool,
		) (string, error) {
			return "", resolveErr
		},
		servicesForNamespace: func(*kubernetes.Clientset, string) looperGatewayServices {
			return services
		},
	}
	opts := &framework.SetupOptions{
		KubeClient:  &kubernetes.Clientset{},
		KubeConfig:  "/tmp/kubeconfig",
		ClusterName: "rollback-cluster",
		Verbose:     true,
	}

	err := profile.Setup(context.Background(), opts)
	if !errors.Is(err, resolveErr) || !errors.Is(err, cleanupErr) {
		t.Fatalf("Setup() error = %v, want joined resolution and cleanup errors", err)
	}
	if stack.setupCalls != 1 || stack.teardownCalls != 1 {
		t.Fatalf(
			"stack calls = setup:%d teardown:%d, want 1 each",
			stack.setupCalls,
			stack.teardownCalls,
		)
	}
	if stack.teardownOpts == nil ||
		stack.teardownOpts.KubeClient != opts.KubeClient ||
		stack.teardownOpts.KubeConfig != opts.KubeConfig ||
		stack.teardownOpts.ClusterName != opts.ClusterName ||
		stack.teardownOpts.Verbose != opts.Verbose {
		t.Fatalf("rollback teardown options = %#v, want setup identity fields", stack.teardownOpts)
	}
}

func TestProfileSetupRollsBackStackWhenAliasCreateFails(t *testing.T) {
	createErr := errors.New("service create failed")
	stack := &memoryGatewayStack{serviceConfig: framework.ServiceConfig{
		Namespace:     "envoy-system",
		LabelSelector: "gateway=semantic-router",
	}}
	services := &memoryLooperGatewayServices{createErr: createErr}
	profile := &Profile{
		stack: stack,
		resolveGatewayService: func(
			context.Context,
			*kubernetes.Clientset,
			string,
			string,
			bool,
		) (string, error) {
			return "generated-gateway", nil
		},
		servicesForNamespace: func(*kubernetes.Clientset, string) looperGatewayServices {
			return services
		},
	}

	err := profile.Setup(context.Background(), &framework.SetupOptions{
		KubeClient: &kubernetes.Clientset{},
	})
	if !errors.Is(err, createErr) {
		t.Fatalf("Setup() error = %v, want alias create failure", err)
	}
	if stack.teardownCalls != 1 {
		t.Fatalf("stack teardown calls = %d, want 1", stack.teardownCalls)
	}
}
