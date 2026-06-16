/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package looper

import "testing"

func TestSecretIsStableAndValidates(t *testing.T) {
	s := Secret()
	if s == "" {
		t.Fatal("Secret() returned an empty secret")
	}
	if got := Secret(); got != s {
		t.Fatalf("Secret() not stable: first %q second %q", s, got)
	}
	if !ValidateSecret(s) {
		t.Fatal("ValidateSecret rejected the process secret")
	}
}

func TestValidateSecretRejectsBadInput(t *testing.T) {
	s := Secret()
	if ValidateSecret("") {
		t.Fatal("ValidateSecret accepted an empty secret")
	}
	if ValidateSecret(s + "tamper") {
		t.Fatal("ValidateSecret accepted a tampered secret")
	}
	if ValidateSecret("not-the-secret") {
		t.Fatal("ValidateSecret accepted an unrelated value")
	}
}

func TestLoadSecretUsesEnvOverride(t *testing.T) {
	t.Setenv(SecretEnv, "  shared-replica-secret  ")
	if got := loadSecret(); got != "shared-replica-secret" {
		t.Fatalf("loadSecret() = %q, want trimmed env value %q", got, "shared-replica-secret")
	}
}

func TestLoadSecretGeneratesRandomWhenEnvUnset(t *testing.T) {
	t.Setenv(SecretEnv, "")
	first := loadSecret()
	if len(first) != 64 { // 32 bytes hex-encoded
		t.Fatalf("loadSecret() length = %d, want 64 hex chars", len(first))
	}
	if second := loadSecret(); second == first {
		t.Fatal("loadSecret() produced identical random secrets across calls")
	}
}
