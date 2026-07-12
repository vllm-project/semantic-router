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

package extproc

import (
	"net/http"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func TestLooperRequestAuthenticatorFromEnvironmentIsSharedAcrossReplicas(t *testing.T) {
	t.Setenv(looperSharedSecretEnvironmentVariable, strings.Repeat("0123456789abcdef", 4))

	first, err := newLooperRequestAuthenticatorFromEnvironment()
	if err != nil {
		t.Fatalf("newLooperRequestAuthenticatorFromEnvironment() first error = %v", err)
	}
	second, err := newLooperRequestAuthenticatorFromEnvironment()
	if err != nil {
		t.Fatalf("newLooperRequestAuthenticatorFromEnvironment() second error = %v", err)
	}

	requestHeaders := make(http.Header)
	first.Apply(requestHeaders)
	if !second.Authenticate(
		requestHeaders.Get(headers.VSRLooperRequest),
		requestHeaders.Get(headers.VSRLooperSecret),
	) {
		t.Fatal("replica authenticator rejected a request signed with the shared deployment secret")
	}
}

func TestLooperRequestAuthenticatorFromEnvironmentRejectsInvalidSecret(t *testing.T) {
	invalidSecret := strings.Repeat("not-hex-", 8)
	t.Setenv(looperSharedSecretEnvironmentVariable, invalidSecret)

	_, err := newLooperRequestAuthenticatorFromEnvironment()
	if err == nil {
		t.Fatal("newLooperRequestAuthenticatorFromEnvironment() error = nil")
	}
	if strings.Contains(err.Error(), invalidSecret) {
		t.Fatal("startup error exposed the supplied shared secret")
	}
}
