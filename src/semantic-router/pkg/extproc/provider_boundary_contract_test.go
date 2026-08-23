package extproc

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestDataPlaneProviderBoundaryExcludesLegacyPhysicalRouting prevents ExtProc
// from regaining a second physical-backend control plane. Provider catalog
// compilation produces immutable runtime snapshots; only BackendInvoker may
// interpret their endpoint, credential, protocol, retry, and fallback fields.
func TestDataPlaneProviderBoundaryExcludesLegacyPhysicalRouting(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	forbidden := []string{
		"VLLMEndpoints",
		"PreferredEndpoints",
		"ImageGenBackends",
		"ResolveExternalModelID",
		"GetModelAPIFormat",
		"APIFormatAnthropic",
		"handleAnthropicRouting",
		"handleAnthropicStreamingResponseBody",
		"resolveARModelEndpoint",
		"resolveDiffusionBackend",
		"executeOmni",
		"executeBoth",
		"callARModel",
		"imagegen.CreateBackend",
		"ctx.SkipProcessing",
		"skipProcessingEnabled",
		"handleSkipProcessing",
		// ExtProc owns semantic routing state, never a canonical Chat-shaped
		// working body or a pair-specific translation sidecar. Wire DTOs are
		// confined to protocol codecs and private service adapters.
		"ResponseAPICtx",
		"ClientProtocol",
		"IRExtensions",
		"ChatCompletionNewParams",
		"OriginalRequestBody",
		"WorkingRequestBody",
		"workingRequestBody",
		"setWorkingRequestBody",
	}
	internalServiceEgress := map[string]struct{}{
		"req_filter_memory_rewrite.go": {},
		"req_filter_rag_external.go":   {},
	}
	directEgress := []string{
		"http.NewRequest(",
		"http.NewRequestWithContext(",
		"client.Do(",
		"RoundTrip(",
	}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		contents, readErr := os.ReadFile(filepath.Clean(name))
		if readErr != nil {
			t.Fatalf("read %s: %v", name, readErr)
		}
		for _, symbol := range forbidden {
			if strings.Contains(string(contents), symbol) {
				t.Errorf("%s contains forbidden legacy routing symbol %q", name, symbol)
			}
		}
		if _, allowed := internalServiceEgress[name]; allowed {
			continue
		}
		for _, call := range directEgress {
			if strings.Contains(string(contents), call) {
				t.Errorf("%s contains direct network egress %q outside an internal service adapter", name, call)
			}
		}
	}
}
