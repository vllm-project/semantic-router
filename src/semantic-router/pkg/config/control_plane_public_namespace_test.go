package config

import "testing"

const testPublicNamespaceID = "11111111-1111-4111-8111-111111111111"

func TestManagedRoutingOnlyRequiresOneCanonicalPublicNamespace(t *testing.T) {
	cfg := DefaultGlobalConfig()
	configureValidManagedAccess(&cfg)
	cfg.Access = DefaultAccessServiceConfig()

	if err := cfg.ValidateControlPlaneBootstrap(); err == nil {
		t.Fatal("managed routing-only accepted a missing public namespace")
	}
	cfg.ControlPlane.PublicNamespaceID = testPublicNamespaceID
	if err := cfg.ValidateControlPlaneBootstrap(); err != nil {
		t.Fatalf("managed routing-only bootstrap error = %v", err)
	}

	for _, invalid := range []string{
		"{11111111-1111-4111-8111-111111111111}",
		"11111111-1111-4111-8111-111111111111 ",
		"00000000-0000-0000-0000-000000000000",
		"PUBLIC",
	} {
		cfg.ControlPlane.PublicNamespaceID = invalid
		if err := cfg.ValidateControlPlaneBootstrap(); err == nil {
			t.Fatalf("managed routing-only accepted public namespace %q", invalid)
		}
	}
}

func TestPublicNamespaceIsExclusiveToManagedRoutingOnly(t *testing.T) {
	managed := DefaultGlobalConfig()
	configureValidManagedAccess(&managed)
	managed.ControlPlane.PublicNamespaceID = testPublicNamespaceID
	if err := managed.ValidateControlPlaneBootstrap(); err == nil {
		t.Fatal("managed access accepted an operator-selected public namespace")
	}

	standalone := DefaultGlobalConfig()
	standalone.ControlPlane.PublicNamespaceID = testPublicNamespaceID
	if err := standalone.ValidateControlPlaneBootstrap(); err == nil {
		t.Fatal("standalone accepted a managed public namespace")
	}
}

func TestControlPlaneBootstrapValidatesBackendDispatch(t *testing.T) {
	cfg := DefaultGlobalConfig()
	cfg.BackendDispatch.Port = 0
	if err := cfg.ValidateControlPlaneBootstrap(); err == nil {
		t.Fatal("bootstrap accepted an invalid backend-dispatch listener")
	}
}
