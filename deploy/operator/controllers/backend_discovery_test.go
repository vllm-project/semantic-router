/*
Copyright 2026 vLLM Semantic Router Contributors.

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

package controllers

import (
	"os"
	"path/filepath"
	"testing"
)

func TestClusterDNSDomainDefaultsToClusterLocal(t *testing.T) {
	// A missing resolv.conf (or one without usable search entries) must fall
	// back to the default Kubernetes cluster domain.
	got := clusterDNSDomainFromFile(filepath.Join(t.TempDir(), "resolv.conf"))
	if got != "cluster.local" {
		t.Fatalf("expected default cluster domain %q, got %q", "cluster.local", got)
	}
}

func TestClusterDNSDomainFromSearchEntry(t *testing.T) {
	// Kubernetes renders search entries as "<namespace>.svc.<cluster-domain>";
	// the operator must pick the custom domain, not assume cluster.local.
	file := filepath.Join(t.TempDir(), "resolv.conf")
	contents := "nameserver 10.96.0.10\nsearch custom-ns.svc.my.corp my.corp\noptions ndots:5\n"
	if err := os.WriteFile(file, []byte(contents), 0o644); err != nil {
		t.Fatalf("failed to write resolv.conf fixture: %v", err)
	}
	if got := clusterDNSDomainFromFile(file); got != "my.corp" {
		t.Fatalf("expected custom domain %q, got %q", "my.corp", got)
	}
}

func TestBackendServiceFQDNIncludesSvcLabel(t *testing.T) {
	// Regression test: generated backend addresses must keep the ".svc." DNS
	// label for both the default and custom cluster domains.
	testCases := []struct {
		name      string
		service   string
		namespace string
		domain    string
		want      string
	}{
		{
			name:      "default domain",
			service:   "qwen3-svc",
			namespace: "default",
			domain:    "cluster.local",
			want:      "qwen3-svc.default.svc.cluster.local",
		},
		{
			name:      "custom domain",
			service:   "model-server",
			namespace: "custom-ns",
			domain:    "my.corp",
			want:      "model-server.custom-ns.svc.my.corp",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := backendServiceFQDN(tc.service, tc.namespace, tc.domain); got != tc.want {
				t.Fatalf("backendServiceFQDN(%q, %q, %q) = %q, want %q",
					tc.service, tc.namespace, tc.domain, got, tc.want)
			}
		})
	}
}
