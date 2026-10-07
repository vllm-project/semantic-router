package looper

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/graph"
)

// Headers that label a test hop for the fake model server it reaches.
const (
	testHopIteration = "X-Test-Hop-Iteration"
	testHopDecision  = "X-Test-Hop-Decision"
)

// hopsTo returns a hop client whose calls reach the fake model server at url,
// each labeled with its hop's iteration and decision, so a fixture can tell
// the calls apart.
func hopsTo(cfg *config.LooperConfig, url string) *Client {
	return NewHopClient(cfg, labeledHops{url: url})
}

type labeledHops struct{ url string }

func (h labeledHops) Call(ctx context.Context, req *graph.HopRequest) (*graph.HopResponse, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, h.url, bytes.NewReader(req.Request.Body))
	if err != nil {
		return nil, err
	}
	for _, field := range req.Request.Header {
		if !strings.HasPrefix(field.Name, ":") && field.Name != "content-length" {
			httpReq.Header.Add(field.Name, field.Value)
		}
	}
	httpReq.Header.Set(testHopIteration, strconv.Itoa(req.Hop.Iteration))
	httpReq.Header.Set(testHopDecision, req.Hop.Decision)
	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	return &graph.HopResponse{Status: resp.StatusCode, Header: routing.Header{{Name: "content-type", Value: resp.Header.Get("Content-Type")}}, Body: body}, err
}

// fusionHopsTo is a Fusion Looper whose calls reach the fake server at url
// as labeled hops.
func fusionHopsTo(url string) *FusionLooper {
	cfg := &config.LooperConfig{}
	return newFusionLooper(cfg, borrowClient(hopsTo(cfg, url)))
}

// workflowsHopsTo is a Workflows Looper whose calls reach the fake server at
// url as labeled hops.
func workflowsHopsTo(url string) *WorkflowsLooper {
	cfg := &config.LooperConfig{}
	return newWorkflowsLooper(cfg, borrowClient(hopsTo(cfg, url)))
}
