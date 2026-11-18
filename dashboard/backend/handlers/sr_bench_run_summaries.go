package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"sync"
	"time"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

// The run list carries identity and progress, not always scores. For finished
// live runs without a summary the proxy reads each report once and attaches a
// compact summary, so the run table can show a best score and recorded cost
// without the browser reading every report. A finished run's report is fixed
// for a given updated_at, which keys the cache.
const (
	maxSummaryReadsPerList  = 20
	summaryReadConcurrency  = 4
	summaryReadBudget       = 3 * time.Second
	maxCachedRunSummaries   = 2000
	maxSummaryResponseBytes = 8 << 20
)

var summaryTargetFields = []string{
	"id", "total", "completed", "failed", "scored", "pending", "complete",
	"accuracy", "macro_accuracy", "cost_usd", "total_spend_usd", "evaluation_cost_usd",
}

var finishedRunStatuses = map[string]bool{
	"completed": true, "failed": true, "cancelled": true, "interrupted": true,
}

type cachedRunSummary struct {
	updatedAt string
	summary   json.RawMessage
}

type runSummaryCache struct {
	mu      sync.Mutex
	entries map[string]cachedRunSummary
}

func (c *runSummaryCache) get(id, updatedAt string) (json.RawMessage, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	entry, ok := c.entries[id]
	if !ok || entry.updatedAt != updatedAt {
		return nil, false
	}
	return entry.summary, true
}

func (c *runSummaryCache) put(id, updatedAt string, summary json.RawMessage) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.entries == nil || len(c.entries) >= maxCachedRunSummaries {
		c.entries = map[string]cachedRunSummary{}
	}
	c.entries[id] = cachedRunSummary{updatedAt: updatedAt, summary: summary}
}

type listedRun struct {
	fields    map[string]json.RawMessage
	id        string
	updatedAt string
}

// enrichRunList returns data unchanged unless at least one summary was attached.
// Report failures never fail the list; the run simply keeps no summary.
func (h *SRBenchHandler) enrichRunList(ctx context.Context, actor dashboardauth.AuthContext, data []byte) []byte {
	var document map[string]json.RawMessage
	if json.Unmarshal(data, &document) != nil {
		return data
	}
	var runs []map[string]json.RawMessage
	if json.Unmarshal(document["runs"], &runs) != nil {
		return data
	}
	var missing []listedRun
	changed := false
	for _, fields := range runs {
		run, ok := summaryCandidate(fields)
		if !ok {
			continue
		}
		if summary, cached := h.summaries.get(run.id, run.updatedAt); cached {
			fields["summary"] = summary
			changed = true
			continue
		}
		if len(missing) < maxSummaryReadsPerList {
			missing = append(missing, run)
		}
	}
	if h.readSummaries(ctx, actor, missing) > 0 {
		changed = true
	}
	if !changed {
		return data
	}
	encodedRuns, err := json.Marshal(runs)
	if err != nil {
		return data
	}
	document["runs"] = encodedRuns
	encoded, err := json.Marshal(document)
	if err != nil {
		return data
	}
	return encoded
}

func summaryCandidate(fields map[string]json.RawMessage) (listedRun, bool) {
	if summary, present := fields["summary"]; present && !bytes.Equal(bytes.TrimSpace(summary), []byte("null")) {
		return listedRun{}, false
	}
	var id, status, updatedAt string
	var manifest struct {
		Mode string `json:"mode"`
	}
	if json.Unmarshal(fields["id"], &id) != nil || !validSRBenchRunID(id) ||
		json.Unmarshal(fields["status"], &status) != nil || !finishedRunStatuses[status] ||
		json.Unmarshal(fields["updated_at"], &updatedAt) != nil ||
		json.Unmarshal(fields["manifest"], &manifest) != nil || manifest.Mode != "live" {
		return listedRun{}, false
	}
	return listedRun{fields: fields, id: id, updatedAt: updatedAt}, true
}

func (h *SRBenchHandler) readSummaries(ctx context.Context, actor dashboardauth.AuthContext, runs []listedRun) int {
	if len(runs) == 0 {
		return 0
	}
	ctx, cancel := context.WithTimeout(ctx, summaryReadBudget)
	defer cancel()
	results := make([]json.RawMessage, len(runs))
	slots := make(chan struct{}, summaryReadConcurrency)
	var wait sync.WaitGroup
	for index, run := range runs {
		wait.Add(1)
		go func(index int, run listedRun) {
			defer wait.Done()
			slots <- struct{}{}
			defer func() { <-slots }()
			results[index] = h.readSummary(ctx, actor, run.id)
		}(index, run)
	}
	wait.Wait()
	attached := 0
	for index, summary := range results {
		if summary == nil {
			continue
		}
		h.summaries.put(runs[index].id, runs[index].updatedAt, summary)
		runs[index].fields["summary"] = summary
		attached++
	}
	return attached
}

func (h *SRBenchHandler) readSummary(ctx context.Context, actor dashboardauth.AuthContext, id string) json.RawMessage {
	upstream := *h.origin
	upstream.Path = SRBenchAPIPath + "/runs/" + url.PathEscape(id) + "/report"
	upstream.RawQuery = ""
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, upstream.String(), nil)
	if err != nil {
		return nil
	}
	request.Header.Set("Authorization", "Bearer "+h.token)
	request.Header.Set("Accept", "application/json")
	request.Header.Set("X-SR-Bench-Actor-ID", actor.UserID)
	request.Header.Set("X-SR-Bench-Actor-Role", actor.Role)
	response, err := h.client.Do(request)
	if err != nil {
		return nil
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, maxSummaryResponseBytes+1))
	if err != nil || len(data) > maxSummaryResponseBytes || response.StatusCode != http.StatusOK {
		return nil
	}
	return compactSummary(data)
}

// compactSummary keeps only the per-target outcome counts, scores and recorded
// costs the run table reads; usage, routing and provenance stay in the report.
func compactSummary(report []byte) json.RawMessage {
	var parsed struct {
		Summary struct {
			Targets   []map[string]json.RawMessage `json:"targets"`
			WallTimeS json.RawMessage              `json:"wall_time_s"`
		} `json:"summary"`
	}
	if json.Unmarshal(report, &parsed) != nil || len(parsed.Summary.Targets) == 0 {
		return nil
	}
	targets := make([]map[string]json.RawMessage, 0, len(parsed.Summary.Targets))
	for _, target := range parsed.Summary.Targets {
		compact := map[string]json.RawMessage{}
		for _, field := range summaryTargetFields {
			if value, ok := target[field]; ok {
				compact[field] = value
			}
		}
		targets = append(targets, compact)
	}
	summary := map[string]any{"targets": targets}
	if len(parsed.Summary.WallTimeS) > 0 {
		summary["wall_time_s"] = parsed.Summary.WallTimeS
	}
	encoded, err := json.Marshal(summary)
	if err != nil {
		return nil
	}
	return encoded
}
