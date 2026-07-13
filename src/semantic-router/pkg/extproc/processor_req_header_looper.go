package extproc

import (
	"strconv"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

var looperInternalRequestHeaders = []string{
	headers.VSRLooperRequest,
	headers.VSRLooperSecret,
	headers.VSRLooperDecision,
	headers.VSRLooperIteration,
	headers.VSRFusionDepth,
}

type looperRequestMetadata struct {
	values map[string]string
	counts map[string]int
}

func (m *looperRequestMetadata) captureCanonical(name, value string) bool {
	if !isCanonicalLooperInternalRequestHeader(name) {
		return false
	}
	if m.values == nil {
		m.values = make(map[string]string, len(looperInternalRequestHeaders))
		m.counts = make(map[string]int, len(looperInternalRequestHeaders))
	}
	m.values[name] = value
	m.counts[name]++
	return true
}

func (m *looperRequestMetadata) present() bool {
	return m != nil && len(m.counts) > 0
}

func (m *looperRequestMetadata) authenticate(
	authenticator *looper.RequestAuthenticator,
) (bool, string, int) {
	if m == nil {
		return false, "", 0
	}
	metadataValid, iteration := validateLooperMetadata(m.values, m.counts)
	trusted := metadataValid &&
		m.counts[headers.VSRLooperRequest] == 1 &&
		m.counts[headers.VSRLooperSecret] == 1 &&
		authenticator.Authenticate(
			m.values[headers.VSRLooperRequest],
			m.values[headers.VSRLooperSecret],
		)
	if !trusted {
		return false, "", 0
	}
	return true, m.values[headers.VSRLooperDecision], iteration
}

func requestTraceHeaders(
	v *ext_proc.ProcessingRequest_RequestHeaders,
) map[string]string {
	headerMap := make(map[string]string, len(v.RequestHeaders.Headers.Headers))
	for _, header := range v.RequestHeaders.Headers.Headers {
		if isLooperInternalRequestHeader(header.Key) {
			continue
		}
		headerMap[header.Key] = extractHeaderValue(header)
	}
	return headerMap
}

func isLooperInternalRequestHeader(name string) bool {
	return isCanonicalLooperInternalRequestHeader(strings.ToLower(name))
}

func isCanonicalLooperInternalRequestHeader(name string) bool {
	return strings.HasPrefix(name, "x-vsr-looper-") ||
		name == headers.VSRFusionDepth
}

func isKnownLooperInternalRequestHeader(name string) bool {
	for _, knownName := range looperInternalRequestHeaders {
		if name == knownName {
			return true
		}
	}
	return false
}

func validateLooperMetadata(values map[string]string, counts map[string]int) (bool, int) {
	if !validateLooperMetadataShape(values, counts) {
		return false, 0
	}

	iteration, valid := parseOptionalPositiveLooperMetadata(
		values,
		counts,
		headers.VSRLooperIteration,
	)
	if !valid {
		return false, 0
	}
	if _, valid := parseOptionalPositiveLooperMetadata(
		values,
		counts,
		headers.VSRFusionDepth,
	); !valid {
		return false, 0
	}
	return true, iteration
}

func validateLooperMetadataShape(values map[string]string, counts map[string]int) bool {
	for name, count := range counts {
		if !isKnownLooperInternalRequestHeader(name) || count > 1 {
			return false
		}
	}
	for _, name := range []string{
		headers.VSRLooperDecision,
		headers.VSRLooperIteration,
		headers.VSRFusionDepth,
	} {
		if counts[name] == 1 && strings.TrimSpace(values[name]) == "" {
			return false
		}
	}
	return true
}

func parseOptionalPositiveLooperMetadata(
	values map[string]string,
	counts map[string]int,
	name string,
) (int, bool) {
	if counts[name] == 0 {
		return 0, true
	}
	parsed, err := strconv.Atoi(values[name])
	if err != nil || parsed < 1 {
		return 0, false
	}
	return parsed, true
}
