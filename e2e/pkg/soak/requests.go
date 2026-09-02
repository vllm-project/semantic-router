package soak

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync/atomic"
	"time"
)

const sessionStoreCapEntries = 50_000

const requestTimeout = 120 * time.Second

// Client issues OpenAI-shaped chat completions against the gateway.
type Client struct {
	httpClient  *http.Client
	gatewayURL  string
	model       string
	highCardIDs int

	seq            atomic.Uint64
	hcSeq          atomic.Uint64
	hcSuccess      atomic.Uint64
	capCrossedUnix atomic.Int64
}

// NewClient builds a client whose connection pool is sized for the load
// generator's concurrency.
func NewClient(gatewayURL, model string, concurrency, highCardIDs int) *Client {
	if concurrency < 1 {
		concurrency = 1
	}
	if highCardIDs < 1 {
		highCardIDs = 1
	}
	transport := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   10 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConns:          concurrency * 2,
		MaxIdleConnsPerHost:   concurrency * 2,
		MaxConnsPerHost:       concurrency * 2,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: time.Second,
	}
	return &Client{
		httpClient:  &http.Client{Transport: transport, Timeout: requestTimeout},
		gatewayURL:  gatewayURL,
		model:       model,
		highCardIDs: highCardIDs,
	}
}

// Chat is the steady-state request function.
func (c *Client) Chat(ctx context.Context) error {
	n := c.seq.Add(1) - 1
	return c.do(ctx, prompts[n%uint64(len(prompts))], "")
}

// ChatHighCardinality cycles through highCardIDs distinct session identities.
func (c *Client) ChatHighCardinality(ctx context.Context) error {
	n := c.hcSeq.Add(1) - 1
	id := fmt.Sprintf("soak-hc-%06d", n%uint64(c.highCardIDs))
	if err := c.do(ctx, prompts[n%uint64(len(prompts))], id); err != nil {
		return err
	}
	if issued := c.hcSuccess.Add(1); min(issued, uint64(c.highCardIDs)) >= sessionStoreCapEntries {
		c.capCrossedUnix.CompareAndSwap(0, time.Now().Unix())
	}
	return nil
}

// UniqueIDsIssued reports how many distinct session IDs the router accepted,
// approximated as the number of successful high-cardinality requests capped at
// the pool size.
func (c *Client) UniqueIDsIssued() uint64 {
	return min(c.hcSuccess.Load(), uint64(c.highCardIDs))
}

// CapCrossed reports whether enough unique IDs were accepted to overflow the
// router's per-store 50k cap.
func (c *Client) CapCrossed() bool { return c.capCrossedUnix.Load() != 0 }

// CapCrossedAt returns when the cap was crossed, or the zero time.
func (c *Client) CapCrossedAt() time.Time {
	if v := c.capCrossedUnix.Load(); v != 0 {
		return time.Unix(v, 0).UTC()
	}
	return time.Time{}
}

func (c *Client) do(ctx context.Context, prompt, sessionID string) error {
	content := prompt
	if sessionID != "" {
		content = fmt.Sprintf("[session %s / user %s] %s", sessionID, sessionID, prompt)
	}
	payload := map[string]any{
		"model":  c.model,
		"stream": false,
		"messages": []map[string]string{
			{"role": "user", "content": content},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	reqCtx, cancel := context.WithTimeout(ctx, requestTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, c.gatewayURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if sessionID != "" {
		req.Header.Set("x-session-id", sessionID)
		req.Header.Set("x-authz-user-id", sessionID)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status %d", resp.StatusCode)
	}
	return nil
}
