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
	"encoding/base64"
	"net/http"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/handoff"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

const (
	maxHandoffHeaderBytes         = 6 * 1024
	defaultHandoffLedgerCapacity  = 10_000
	handoffStatusAccepted         = "accepted"
	handoffStatusPartial          = "partial"
	handoffStatusDuplicate        = "duplicate"
	handoffStatusRejected         = "rejected"
	handoffStatusExpired          = "expired"
	handoffStatusCancelled        = "cancelled"
	handoffStatusIgnored          = "ignored"
	handoffReasonAdmitted         = "admitted"
	handoffReasonSelectionPending = "selection_unconsumed"
)

// processHandoffLedger outlives router rebuilds on config reload so a
// cancellation or first sighting is not forgotten by a hot reload.
var processHandoffLedger = handoff.NewLedger(defaultHandoffLedgerCapacity)

var supportedHandoffPaths = map[string]struct{}{
	"/v1/chat/completions": {},
	"/v1/responses":        {},
	"/v1/messages":         {},
}

// handoffReceipt is the content-free outcome returned to the gateway. Version
// and ID are present only after the envelope parsed.
type handoffReceipt struct {
	Version string
	ID      string
	Status  string
	Reason  string
}

// admitHandoffEnvelope strips the carrier from request metadata, then
// validates and records a trusted envelope. Version 1 never alters routing, so
// the receipt is final once this returns; a non-nil response rejects the
// request.
func (r *OpenAIRouter) admitHandoffEnvelope(
	request *ext_proc.ProcessingRequest_RequestHeaders,
	method string,
	path string,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	values := handoffHeaderValues(request)
	removeHeaderValueCI(ctx, headers.VSRHandoffEnvelope)
	switch {
	case len(values) == 0:
		return nil
	case !r.handoffEnabled():
		return r.ignoreHandoff(ctx, "feature_disabled")
	case !supportedHandoffEndpoint(method, path):
		return r.ignoreHandoff(ctx, "unsupported_endpoint")
	case ctx.SkipProcessing:
		return r.ignoreHandoff(ctx, "skip_processing")
	case len(values) != 1:
		return r.rejectHandoff(ctx, nil, http.StatusBadRequest, handoffStatusRejected, "multiple_headers")
	case len(values[0]) > maxHandoffHeaderBytes:
		return r.rejectHandoff(ctx, nil, http.StatusRequestEntityTooLarge, handoffStatusRejected, string(handoff.CodePayloadTooLarge))
	}
	decoded, err := base64.RawURLEncoding.DecodeString(values[0])
	if err != nil {
		return r.rejectHandoff(ctx, nil, http.StatusBadRequest, handoffStatusRejected, "malformed_encoding")
	}

	now := r.currentHandoffTime()
	envelope, err := handoff.Parse(decoded, now)
	if err != nil {
		switch code := handoff.CodeOf(err); code {
		case handoff.CodePayloadTooLarge:
			return r.rejectHandoff(ctx, nil, http.StatusRequestEntityTooLarge, handoffStatusRejected, string(code))
		case handoff.CodeExpired:
			return r.rejectHandoff(ctx, nil, http.StatusUnprocessableEntity, handoffStatusExpired, string(code))
		default:
			return r.rejectHandoff(ctx, nil, http.StatusBadRequest, handoffStatusRejected, string(code))
		}
	}

	switch r.currentHandoffLedger().Admit(envelope, now) {
	case handoff.AdmissionConflict:
		return r.rejectHandoff(ctx, envelope, http.StatusConflict, handoffStatusRejected, "handoff_id_conflict")
	case handoff.AdmissionCancelRecorded:
		return r.rejectHandoff(ctx, envelope, http.StatusConflict, handoffStatusCancelled, "cancel_recorded")
	case handoff.AdmissionCancelled:
		return r.rejectHandoff(ctx, envelope, http.StatusConflict, handoffStatusCancelled, "handoff_cancelled")
	case handoff.AdmissionDuplicate:
		setHandoffReceipt(ctx, envelope, handoffStatusDuplicate, "idempotent_retry")
	default:
		if envelope.Selection != nil {
			setHandoffReceipt(ctx, envelope, handoffStatusPartial, handoffReasonSelectionPending)
		} else {
			setHandoffReceipt(ctx, envelope, handoffStatusAccepted, handoffReasonAdmitted)
		}
	}
	return nil
}

func handoffHeaderValues(request *ext_proc.ProcessingRequest_RequestHeaders) []string {
	if request == nil || request.RequestHeaders == nil || request.RequestHeaders.Headers == nil {
		return nil
	}
	var values []string
	for _, header := range request.RequestHeaders.Headers.Headers {
		if strings.EqualFold(header.Key, headers.VSRHandoffEnvelope) {
			values = append(values, extractHeaderValue(header))
		}
	}
	return values
}

func supportedHandoffEndpoint(method, path string) bool {
	path, _, _ = strings.Cut(path, "?")
	_, ok := supportedHandoffPaths[path]
	return ok && method == http.MethodPost
}

func (r *OpenAIRouter) handoffEnabled() bool {
	return r != nil && r.Config != nil && r.Config.Handoff.IsEnabled()
}

func (r *OpenAIRouter) currentHandoffTime() time.Time {
	if r != nil && r.handoffNow != nil {
		return r.handoffNow()
	}
	return time.Now()
}

func (r *OpenAIRouter) currentHandoffLedger() *handoff.Ledger {
	if r != nil && r.handoffLedger != nil {
		return r.handoffLedger
	}
	return processHandoffLedger
}

func (r *OpenAIRouter) ignoreHandoff(ctx *RequestContext, reason string) *ext_proc.ProcessingResponse {
	setHandoffReceipt(ctx, nil, handoffStatusIgnored, reason)
	return nil
}

func (r *OpenAIRouter) rejectHandoff(
	ctx *RequestContext,
	envelope *handoff.Envelope,
	statusCode int,
	status string,
	reason string,
) *ext_proc.ProcessingResponse {
	setHandoffReceipt(ctx, envelope, status, reason)
	return r.createErrorResponse(statusCode, "handoff envelope "+status+": "+reason)
}

func setHandoffReceipt(ctx *RequestContext, envelope *handoff.Envelope, status, reason string) {
	receipt := handoffReceipt{Status: status, Reason: reason}
	if envelope != nil {
		receipt.Version = envelope.Version
		receipt.ID = envelope.HandoffID
	}
	ctx.HandoffReceipt = receipt
}

func appendHandoffReceiptToImmediateResponse(response *ext_proc.ProcessingResponse, ctx *RequestContext) {
	if ctx == nil || ctx.HandoffReceipt.Status == "" {
		return
	}
	appendImmediateResponseHeader(response, headers.VSRHandoffVersion, ctx.HandoffReceipt.Version)
	appendImmediateResponseHeader(response, headers.VSRHandoffID, ctx.HandoffReceipt.ID)
	appendImmediateResponseHeader(response, headers.VSRHandoffStatus, ctx.HandoffReceipt.Status)
	appendImmediateResponseHeader(response, headers.VSRHandoffReason, ctx.HandoffReceipt.Reason)
}
