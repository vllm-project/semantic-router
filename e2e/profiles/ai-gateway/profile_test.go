package aigateway

import (
	"context"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func TestProfileRegistersLooperSecurityContract(t *testing.T) {
	const testcaseName = "looper-secret-forgery"
	registered, err := pkgtestcases.ListByNames(testcaseName)
	if err != nil || len(registered) != 1 || registered[0].Fn == nil {
		t.Fatalf("registered Looper testcase = %#v, err = %v", registered, err)
	}

	found := false
	for _, name := range NewProfile().GetTestCases() {
		if name == testcaseName {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("baseline profile does not include registered testcase %q", testcaseName)
	}
}

func TestEnsureLooperGatewayAliasCreatesStableExternalName(t *testing.T) {
	services := &memoryLooperGatewayServices{}
	const (
		namespace    = "router-system"
		externalName = "generated-gateway.envoy-system.svc.cluster.local"
	)

	if err := ensureLooperGatewayAlias(
		context.Background(),
		services,
		namespace,
		externalName,
	); err != nil {
		t.Fatalf("ensureLooperGatewayAlias() error = %v", err)
	}

	service := services.service
	if service.Spec.Type != corev1.ServiceTypeExternalName {
		t.Fatalf("alias service type = %q, want ExternalName", service.Spec.Type)
	}
	if service.Spec.ExternalName != externalName {
		t.Fatalf("alias target = %q, want %q", service.Spec.ExternalName, externalName)
	}
	if len(service.Spec.Ports) != 1 || service.Spec.Ports[0].Port != 80 {
		t.Fatalf("alias ports = %#v, want one HTTP port 80", service.Spec.Ports)
	}
	if !isOwnedLooperGatewayAlias(service) {
		t.Fatalf("created alias labels = %#v, want E2E ownership identity", service.Labels)
	}

	if err := ensureLooperGatewayAlias(
		context.Background(),
		services,
		namespace,
		externalName,
	); err != nil {
		t.Fatalf("idempotent ensureLooperGatewayAlias() error = %v", err)
	}
}

func TestEnsureLooperGatewayAliasRejectsConflictingService(t *testing.T) {
	services := &memoryLooperGatewayServices{service: &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      looperGatewayAliasName,
			Namespace: "router-system",
		},
		Spec: corev1.ServiceSpec{
			Type:         corev1.ServiceTypeExternalName,
			ExternalName: "unexpected.example.test",
		},
	}}

	err := ensureLooperGatewayAlias(
		context.Background(),
		services,
		"router-system",
		"generated-gateway.envoy-system.svc.cluster.local",
	)
	if err == nil || !strings.Contains(err.Error(), "expected ownership and specification") {
		t.Fatalf("conflicting alias error = %v, want ownership/specification conflict", err)
	}
}

func TestEnsureLooperGatewayAliasRejectsUnownedMatchingService(t *testing.T) {
	const (
		namespace    = "router-system"
		externalName = "generated-gateway.envoy-system.svc.cluster.local"
	)
	services := &memoryLooperGatewayServices{service: looperGatewayAliasFixture(
		namespace,
		externalName,
		false,
	)}

	err := ensureLooperGatewayAlias(
		context.Background(),
		services,
		namespace,
		externalName,
	)
	if err == nil || !strings.Contains(err.Error(), "expected ownership") {
		t.Fatalf("unowned matching alias error = %v, want ownership rejection", err)
	}
	if services.deleteCalls != 0 {
		t.Fatalf("unowned matching alias delete calls = %d, want 0", services.deleteCalls)
	}
}

func TestEnsureLooperGatewayAliasRejectsOwnedServiceWithWrongPort(t *testing.T) {
	const (
		namespace    = "router-system"
		externalName = "generated-gateway.envoy-system.svc.cluster.local"
	)
	service := looperGatewayAliasFixture(namespace, externalName, true)
	service.Spec.Ports[0].Port = 8080
	services := &memoryLooperGatewayServices{service: service}

	err := ensureLooperGatewayAlias(
		context.Background(),
		services,
		namespace,
		externalName,
	)
	if err == nil || !strings.Contains(err.Error(), "expected ownership and specification") {
		t.Fatalf("wrong-port alias error = %v, want specification rejection", err)
	}
}

func TestDeleteOwnedLooperGatewayAliasRefusesUnownedService(t *testing.T) {
	const namespace = "router-system"
	services := &memoryLooperGatewayServices{service: looperGatewayAliasFixture(
		namespace,
		"generated-gateway.envoy-system.svc.cluster.local",
		false,
	)}

	err := deleteOwnedLooperGatewayAlias(
		context.Background(),
		services,
		namespace,
		fixedLooperGatewayExternalName("generated-gateway.envoy-system.svc.cluster.local"),
	)
	if err == nil || !strings.Contains(err.Error(), "refusing to delete unowned") {
		t.Fatalf("unowned alias deletion error = %v, want refusal", err)
	}
	if services.deleteCalls != 0 {
		t.Fatalf("unowned alias delete calls = %d, want 0", services.deleteCalls)
	}
	if services.service == nil {
		t.Fatal("unowned alias was deleted")
	}
}

func TestDeleteOwnedLooperGatewayAliasIsIdempotent(t *testing.T) {
	const namespace = "router-system"
	services := &memoryLooperGatewayServices{service: looperGatewayAliasFixture(
		namespace,
		"generated-gateway.envoy-system.svc.cluster.local",
		true,
	)}

	expectedExternalName := fixedLooperGatewayExternalName(
		"generated-gateway.envoy-system.svc.cluster.local",
	)
	if err := deleteOwnedLooperGatewayAlias(
		context.Background(),
		services,
		namespace,
		expectedExternalName,
	); err != nil {
		t.Fatalf("deleteOwnedLooperGatewayAlias() error = %v", err)
	}
	if err := deleteOwnedLooperGatewayAlias(
		context.Background(),
		services,
		namespace,
		expectedExternalName,
	); err != nil {
		t.Fatalf("idempotent deleteOwnedLooperGatewayAlias() error = %v", err)
	}
	if services.deleteCalls != 1 {
		t.Fatalf("owned alias delete calls = %d, want 1", services.deleteCalls)
	}
}

func TestDeleteOwnedLooperGatewayAliasRefusesMismatchedService(t *testing.T) {
	const namespace = "router-system"
	services := &memoryLooperGatewayServices{service: looperGatewayAliasFixture(
		namespace,
		"unexpected.envoy-system.svc.cluster.local",
		true,
	)}

	err := deleteOwnedLooperGatewayAlias(
		context.Background(),
		services,
		namespace,
		fixedLooperGatewayExternalName("generated-gateway.envoy-system.svc.cluster.local"),
	)
	if err == nil || !strings.Contains(err.Error(), "refusing to delete mismatched") {
		t.Fatalf("mismatched alias deletion error = %v, want refusal", err)
	}
	if services.deleteCalls != 0 || services.service == nil {
		t.Fatalf(
			"mismatched alias state = delete calls:%d service:%v, want preserved",
			services.deleteCalls,
			services.service,
		)
	}
}
