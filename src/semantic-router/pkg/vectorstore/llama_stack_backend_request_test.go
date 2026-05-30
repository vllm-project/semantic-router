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

package vectorstore

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("LlamaStackBackend request helpers", func() {
	Context("doRequest (HTTP helper via public methods)", func() {
		It("should add Authorization header when auth token is set", func() {
			var receivedAuthHeader string

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				receivedAuthHeader = r.Header.Get("Authorization")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"id": "vs_gen_abc123"}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{
				Endpoint:  server.URL,
				AuthToken: "my-secret-token",
			})
			Expect(err).NotTo(HaveOccurred())

			_ = b.CreateCollection(context.Background(), "test", 768)
			Expect(receivedAuthHeader).To(Equal("Bearer my-secret-token"))
		})

		It("should not add Authorization header when auth token is empty", func() {
			var receivedAuthHeader string

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				receivedAuthHeader = r.Header.Get("Authorization")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"id": "vs_gen_abc123"}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())

			_ = b.CreateCollection(context.Background(), "test", 768)
			Expect(receivedAuthHeader).To(BeEmpty())
		})

		It("should set Content-Type to application/json", func() {
			var receivedContentType string

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				receivedContentType = r.Header.Get("Content-Type")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"id": "vs_gen_abc123"}`))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())

			_ = b.CreateCollection(context.Background(), "test", 768)
			Expect(receivedContentType).To(Equal("application/json"))
		})

		It("should truncate long error messages from server", func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = w.Write([]byte(strings.Repeat("x", 1000)))
			}))
			defer server.Close()

			b, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
			Expect(err).NotTo(HaveOccurred())

			err = b.CreateCollection(context.Background(), "test", 768)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("..."))
		})
	})
})
