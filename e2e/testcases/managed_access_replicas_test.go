package testcases

import (
	"net/url"
	"strings"
	"testing"
)

func TestManagedAccessReplicaEnvoyConfigPinsExtProcAndDispatchToOneRouter(t *testing.T) {
	const routerPodIP = "10.244.0.17"
	config, err := managedAccessReplicaEnvoyConfig(routerPodIP)
	if err != nil {
		t.Fatal(err)
	}

	for _, expected := range []string{
		"address: " + routerPodIP + ", port_value: 50051",
		"address: " + routerPodIP + ", port_value: 8180",
	} {
		if !strings.Contains(config, expected) {
			t.Fatalf("Envoy config is missing %q:\n%s", expected, config)
		}
	}
	fixtureBackend, err := url.Parse(managedAccessFixtureBackendOrigin)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(config, fixtureBackend.Hostname()) ||
		strings.Contains(config, managedAccessFixtureBackendOrigin) {
		t.Fatalf("exact-replica proxy bypasses Router backend dispatch:\n%s", config)
	}
}

func TestManagedAccessReplicaEnvoyConfigRejectsInvalidRouterAddress(t *testing.T) {
	if _, err := managedAccessReplicaEnvoyConfig("router.example.test"); err == nil {
		t.Fatal("expected an invalid Router Pod address to be rejected")
	}
}
