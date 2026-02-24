package ratelimit

import (
	"context"
	"fmt"
	"time"

	ratelimitv3common "github.com/envoyproxy/go-control-plane/envoy/extensions/common/ratelimit/v3"
	ratelimitv3 "github.com/envoyproxy/go-control-plane/envoy/service/ratelimit/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// EnvoyRLSProvider calls an external Envoy Rate Limit Service via gRPC.
// It sends descriptors (user_id, model, groups) and interprets the response
// to produce a Decision.
//
// This provider integrates with the standard envoyproxy/ratelimit reference
// implementation or any gRPC service implementing the
// envoy.service.ratelimit.v3.RateLimitService interface.
type EnvoyRLSProvider struct {
	client  ratelimitv3.RateLimitServiceClient
	conn    *grpc.ClientConn
	domain  string
	timeout time.Duration
}

// NewEnvoyRLSProvider connects to the external Envoy Rate Limit Service.
func NewEnvoyRLSProvider(address, domain string) (*EnvoyRLSProvider, error) {
	conn, err := grpc.NewClient(address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to envoy rate limit service at %s: %w", address, err)
	}

	client := ratelimitv3.NewRateLimitServiceClient(conn)
	logging.Infof("Connected to Envoy Rate Limit Service at %s (domain=%s)", address, domain)

	return &EnvoyRLSProvider{
		client:  client,
		conn:    conn,
		domain:  domain,
		timeout: 2 * time.Second,
	}, nil
}

func (p *EnvoyRLSProvider) Name() string {
	return "envoy-ratelimit"
}

// Check sends a ShouldRateLimit request to the external RLS.
// Descriptors are built from the request context: user_id, model, and each group.
func (p *EnvoyRLSProvider) Check(ctx Context) (*Decision, error) {
	descriptors := p.buildDescriptors(ctx)

	reqCtx, cancel := context.WithTimeout(context.Background(), p.timeout)
	defer cancel()

	resp, err := p.client.ShouldRateLimit(reqCtx, &ratelimitv3.RateLimitRequest{
		Domain:      p.domain,
		Descriptors: descriptors,
		HitsAddend:  1,
	})
	if err != nil {
		return nil, fmt.Errorf("envoy ratelimit gRPC call failed: %w", err)
	}

	decision := &Decision{
		Allowed:  resp.GetOverallCode() == ratelimitv3.RateLimitResponse_OK,
		Provider: p.Name(),
	}

	// Extract quota info from the first status (most descriptors return one)
	if statuses := resp.GetStatuses(); len(statuses) > 0 {
		st := statuses[0]
		decision.Remaining = int64(st.GetLimitRemaining())
		if cl := st.GetCurrentLimit(); cl != nil {
			decision.Limit = int64(cl.GetRequestsPerUnit())
		}
		if dur := st.GetDurationUntilReset(); dur != nil {
			resetDur := dur.AsDuration()
			decision.ResetAt = time.Now().Add(resetDur)
			if !decision.Allowed {
				decision.RetryAfter = resetDur
			}
		}
	}

	return decision, nil
}

// Report is a no-op for the Envoy RLS provider. The external RLS tracks
// usage via its own descriptor counting. Token-based reporting is handled
// by the local-limiter provider.
func (p *EnvoyRLSProvider) Report(_ Context, _ TokenUsage) error {
	return nil
}

// Close releases the gRPC connection.
func (p *EnvoyRLSProvider) Close() error {
	if p.conn != nil {
		return p.conn.Close()
	}
	return nil
}

func (p *EnvoyRLSProvider) buildDescriptors(ctx Context) []*ratelimitv3common.RateLimitDescriptor {
	entries := []*ratelimitv3common.RateLimitDescriptor_Entry{
		{Key: "user_id", Value: ctx.UserID},
	}
	if ctx.Model != "" {
		entries = append(entries, &ratelimitv3common.RateLimitDescriptor_Entry{
			Key: "model", Value: ctx.Model,
		})
	}

	descriptors := []*ratelimitv3common.RateLimitDescriptor{
		{Entries: entries},
	}

	// Add group-based descriptors for group-specific rate limits
	for _, group := range ctx.Groups {
		descriptors = append(descriptors, &ratelimitv3common.RateLimitDescriptor{
			Entries: []*ratelimitv3common.RateLimitDescriptor_Entry{
				{Key: "group", Value: group},
			},
		})
	}

	return descriptors
}
