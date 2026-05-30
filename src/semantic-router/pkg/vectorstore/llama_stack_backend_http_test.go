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

var _ = Describe("LlamaStackBackend HTTP safety", func() {
	It("rejects oversized successful responses", func() {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(strings.Repeat("x", llamaStackMaxResponseBodyBytes+1)))
		}))
		defer server.Close()

		backend, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
		Expect(err).NotTo(HaveOccurred())

		_, err = backend.doRequest(context.Background(), http.MethodGet, "/v1/vector_stores", nil)
		Expect(err).To(MatchError(ContainSubstring("response body exceeded")))
	})

	It("caps error responses before formatting diagnostics", func() {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(strings.Repeat("x", llamaStackMaxErrorBodyBytes+1)))
		}))
		defer server.Close()

		backend, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: server.URL})
		Expect(err).NotTo(HaveOccurred())

		_, err = backend.doRequest(context.Background(), http.MethodGet, "/v1/vector_stores", nil)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("returned status 500"))
		Expect(err.Error()).To(ContainSubstring(strings.Repeat("x", llamaStackErrorPreviewBytes)))
		Expect(err.Error()).To(ContainSubstring("..."))
		Expect(err.Error()).NotTo(ContainSubstring(strings.Repeat("x", llamaStackErrorPreviewBytes+1)))
	})
})
