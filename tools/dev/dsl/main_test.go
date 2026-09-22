package main

import "testing"

func TestValidateRunnerFactoryDisabledByDefault(t *testing.T) {
	if got := validateRunnerFactory(false); got != nil {
		t.Fatalf("validateRunnerFactory(false) = %v, want nil", got)
	}
}

func TestValidateRunnerFactoryEnablesNativeChecks(t *testing.T) {
	if got := validateRunnerFactory(true); got == nil {
		t.Fatal("validateRunnerFactory(true) returned nil")
	}
}
