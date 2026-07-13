package extproc

import (
	"context"
	"net/http"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/imageurl"
)

const invalidRequestImageMessage = "image input must contain a decodable JPEG or PNG image within the supported limits"

// validateFastRequestImages closes the gap between the allocation-light fast
// extractor and full request processing. The extractor preserves image-part
// presence plus every inline data URI; this seam rejects malformed parts before
// default routing and performs bounded decode validation only while holding the
// process-wide native embedding admission lease. Remote HTTP(S) and Anthropic
// file-backed references are deliberately left to the full protocol parser and
// are never fetched by the router.
func (r *OpenAIRouter) validateFastRequestImages(
	fast *FastExtractResult,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if fast == nil || !fast.ImagePartPresent {
		return nil
	}
	if fast.InvalidImagePart {
		return r.invalidRequestImageResponse()
	}
	if len(fast.InlineImageURLs) == 0 {
		return nil
	}

	requestCtx := context.Background()
	if ctx != nil && ctx.TraceContext != nil {
		requestCtx = ctx.TraceContext
	}
	release, err := r.embeddingProcessAdmission().TryAcquire(requestCtx)
	if err != nil {
		return r.classificationEvaluationErrorResponse(err)
	}
	defer release()

	var budget imageurl.RequestImageBudget
	firstImageURL := ""
	for _, candidate := range fast.InlineImageURLs {
		validated, ok := imageurl.ValidateJPEGOrPNGDataURL(candidate, &budget)
		if !ok {
			return r.invalidRequestImageResponse()
		}
		if firstImageURL == "" {
			firstImageURL = validated.DataURL
		}
	}
	if ctx != nil {
		ctx.RequestImageURL = firstImageURL
	}
	fast.FirstImageURL = firstImageURL
	return nil
}

func (r *OpenAIRouter) invalidRequestImageResponse() *ext_proc.ProcessingResponse {
	return r.createNonCacheableEvaluationError(http.StatusBadRequest, invalidRequestImageMessage)
}
