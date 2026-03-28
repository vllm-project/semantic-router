package extproc

import (
	"fmt"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

// createLooperResponse creates an ImmediateResponse from looper output.
func (r *OpenAIRouter) createLooperResponse(
	resp *looper.Response,
	reqCtx *RequestContext,
) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status: &typev3.HttpStatus{Code: typev3.StatusCode_OK},
				Headers: &ext_proc.HeaderMutation{
					SetHeaders: buildLooperResponseHeaders(resp, reqCtx),
				},
				Body: resp.Body,
			},
		},
	}
}

func buildLooperResponseHeaders(
	resp *looper.Response,
	reqCtx *RequestContext,
) []*core.HeaderValueOption {
	setHeaders := baseLooperResponseHeaders(resp)
	appendLooperSignalHeaders(&setHeaders, reqCtx)
	appendLooperDecisionHeaders(&setHeaders, reqCtx)
	return setHeaders
}

func baseLooperResponseHeaders(resp *looper.Response) []*core.HeaderValueOption {
	return []*core.HeaderValueOption{
		newHeaderValueOption("content-type", resp.ContentType),
		newHeaderValueOption(headers.VSRLooperModel, resp.Model),
		newHeaderValueOption(headers.VSRLooperModelsUsed, strings.Join(resp.ModelsUsed, ",")),
		newHeaderValueOption(headers.VSRLooperIterations, fmt.Sprintf("%d", resp.Iterations)),
		newHeaderValueOption(headers.VSRLooperAlgorithm, resp.AlgorithmType),
	}
}

func appendLooperSignalHeaders(
	setHeaders *[]*core.HeaderValueOption,
	reqCtx *RequestContext,
) {
	appendJoinedHeader(setHeaders, headers.VSRMatchedKeywords, reqCtx.VSRMatchedKeywords)
	appendJoinedHeader(setHeaders, headers.VSRMatchedEmbeddings, reqCtx.VSRMatchedEmbeddings)
	appendJoinedHeader(setHeaders, headers.VSRMatchedDomains, reqCtx.VSRMatchedDomains)
	appendJoinedHeader(setHeaders, headers.VSRMatchedFactCheck, reqCtx.VSRMatchedFactCheck)
	appendJoinedHeader(setHeaders, headers.VSRMatchedUserFeedback, reqCtx.VSRMatchedUserFeedback)
	appendJoinedHeader(setHeaders, headers.VSRMatchedPreference, reqCtx.VSRMatchedPreference)
	appendJoinedHeader(setHeaders, headers.VSRMatchedLanguage, reqCtx.VSRMatchedLanguage)
	appendJoinedHeader(setHeaders, headers.VSRMatchedContext, reqCtx.VSRMatchedContext)
	appendJoinedHeader(setHeaders, headers.VSRMatchedStructure, reqCtx.VSRMatchedStructure)
	appendJoinedHeader(setHeaders, headers.VSRMatchedComplexity, reqCtx.VSRMatchedComplexity)
	appendJoinedHeader(setHeaders, headers.VSRMatchedModality, reqCtx.VSRMatchedModality)
	appendJoinedHeader(setHeaders, headers.VSRMatchedAuthz, reqCtx.VSRMatchedAuthz)
	appendJoinedHeader(setHeaders, headers.VSRMatchedJailbreak, reqCtx.VSRMatchedJailbreak)
	appendJoinedHeader(setHeaders, headers.VSRMatchedPII, reqCtx.VSRMatchedPII)
	appendJoinedHeader(setHeaders, headers.VSRMatchedKB, reqCtx.VSRMatchedKB)
	appendJoinedHeader(setHeaders, headers.VSRMatchedProjection, reqCtx.VSRMatchedProjection)

	if reqCtx.VSRContextTokenCount > 0 {
		*setHeaders = append(
			*setHeaders,
			newHeaderValueOption(
				headers.VSRContextTokenCount,
				fmt.Sprintf("%d", reqCtx.VSRContextTokenCount),
			),
		)
	}
}

func appendLooperDecisionHeaders(
	setHeaders *[]*core.HeaderValueOption,
	reqCtx *RequestContext,
) {
	appendOptionalHeader(setHeaders, headers.VSRSelectedDecision, reqCtx.VSRSelectedDecisionName)
	appendOptionalHeader(setHeaders, headers.VSRSelectedCategory, reqCtx.VSRSelectedCategory)
}

func appendJoinedHeader(
	setHeaders *[]*core.HeaderValueOption,
	key string,
	values []string,
) {
	if len(values) == 0 {
		return
	}
	*setHeaders = append(*setHeaders, newHeaderValueOption(key, strings.Join(values, ",")))
}

func appendOptionalHeader(
	setHeaders *[]*core.HeaderValueOption,
	key string,
	value string,
) {
	if value == "" {
		return
	}
	*setHeaders = append(*setHeaders, newHeaderValueOption(key, value))
}

func newHeaderValueOption(key string, value string) *core.HeaderValueOption {
	return &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      key,
			RawValue: []byte(value),
		},
	}
}
