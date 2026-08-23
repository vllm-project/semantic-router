package http

import (
	"fmt"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// KeystoneHeaderOptions returns the public schema and response-path headers
// shared by every immediate response producer.
func KeystoneHeaderOptions(path string) []*core.HeaderValueOption {
	return []*core.HeaderValueOption{
		{Header: &core.HeaderValue{Key: headers.VSRSchemaVersion, RawValue: []byte(headers.SchemaVersionValue)}},
		{Header: &core.HeaderValue{Key: headers.VSRResponsePath, RawValue: []byte(path)}},
	}
}

// CreateCacheHitResponseWithBody wraps a response already encoded by the
// neutral protocol engine. This package owns only ExtProc transport metadata;
// it never parses or rewrites a protocol body.
func CreateCacheHitResponseWithBody(
	responseBody []byte,
	contentType string,
	category string,
	decisionName string,
	matchedKeywords []string,
	similarity ...float32,
) *ext_proc.ProcessingResponse {
	setHeaders := []*core.HeaderValueOption{
		{Header: &core.HeaderValue{Key: "content-type", RawValue: []byte(contentType)}},
		{Header: &core.HeaderValue{Key: headers.VSRCacheHit, RawValue: []byte("true")}},
		{Header: &core.HeaderValue{Key: headers.VSRSelectedDecision, RawValue: []byte(decisionName)}},
	}

	if category != "" {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{Key: headers.VSRSelectedCategory, RawValue: []byte(category)},
		})
	}
	if len(similarity) > 0 && similarity[0] > 0 {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{Key: "x-vsr-cache-similarity", RawValue: []byte(fmt.Sprintf("%.4f", similarity[0]))},
		})
	}
	if len(matchedKeywords) > 0 {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{Key: headers.VSRMatchedKeywords, RawValue: []byte(strings.Join(matchedKeywords, ","))},
		})
	}
	setHeaders = append(setHeaders, KeystoneHeaderOptions(headers.ResponsePathCache)...)

	return immediateOK(responseBody, setHeaders)
}

// CreateFastResponseWithBody wraps an already encoded public response. Body
// generation belongs to the neutral protocol engine.
func CreateFastResponseWithBody(
	responseBody []byte,
	contentType string,
	decisionName string,
) *ext_proc.ProcessingResponse {
	setHeaders := []*core.HeaderValueOption{
		{Header: &core.HeaderValue{Key: "content-type", RawValue: []byte(contentType)}},
		{Header: &core.HeaderValue{Key: headers.VSRSelectedDecision, RawValue: []byte(decisionName)}},
		{Header: &core.HeaderValue{Key: headers.VSRFastResponse, RawValue: []byte("true")}},
	}
	setHeaders = append(setHeaders, KeystoneHeaderOptions(headers.ResponsePathFastResponse)...)

	return immediateOK(responseBody, setHeaders)
}

func immediateOK(body []byte, setHeaders []*core.HeaderValueOption) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status:  &typev3.HttpStatus{Code: typev3.StatusCode_OK},
				Headers: &ext_proc.HeaderMutation{SetHeaders: setHeaders},
				Body:    body,
			},
		},
	}
}
