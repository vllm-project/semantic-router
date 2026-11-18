package handlers

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"strconv"
	"strings"
)

const (
	logSpoolRootEnv           = "VLLM_SR_LOG_SPOOL_DIR"
	defaultLogTailLines       = 100
	maxLogTailLines           = 1000
	maxLogsResponseBytes      = 1 << 20
	logsResponseHeadroom      = 4 << 10
	maxRuntimeLogLineBytes    = 64 << 10
	runtimeLogTruncatedMarker = " ... [log record truncated]"
	logsUnsupportedNotice     = "Runtime logs are unavailable: no bounded log spool is mounted, and the Dashboard is not running in Kubernetes. Start the stack with vllm-sr serve, or deploy the Dashboard in Kubernetes beside the Router."
)

var logSpoolComponents = []string{"router", "envoy", "dashboard"}

// LogEntry represents one centrally redacted line from a log source.
type LogEntry struct {
	Line    string `json:"line"`
	Service string `json:"service,omitempty"`
	// Pod and Timestamp are set for lines read from Kubernetes pods.
	Pod       string `json:"pod,omitempty"`
	Timestamp string `json:"timestamp,omitempty"`
}

// LogsResponse is the bounded Dashboard log API contract.
type LogsResponse struct {
	// DeploymentType names the source: local-spool, kubernetes or unsupported.
	DeploymentType string `json:"deployment_type"`
	Service        string `json:"service"`
	Supported      bool   `json:"supported"`
	// Components lists the components this source can read.
	Components []string `json:"components"`
	Namespace  string   `json:"namespace,omitempty"`
	// Pods lists the pods read for this answer.
	Pods    []string   `json:"pods,omitempty"`
	Logs    []LogEntry `json:"logs"`
	Count   int        `json:"count"`
	Error   string     `json:"error,omitempty"`
	Message string     `json:"message,omitempty"`
}

// LogsHandler retains the historical constructor shape while deliberately
// ignoring the Router URL. Logs are read from the bounded shared spool, or,
// without one, from the Kubernetes API when the Dashboard runs in a cluster.
// This handler never invokes a container runtime or mounts its control socket.
func LogsHandler(_ string) http.HandlerFunc {
	return newLogsHandler(os.Getenv(logSpoolRootEnv), newInClusterPodLogSource())
}

// newLogsHandler serves the spool at spoolRoot and falls back to pods when no
// spool is mounted; pods may be nil outside a cluster.
func newLogsHandler(spoolRoot string, pods *podLogSource) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			writeLogsError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only GET and HEAD are supported.")
			return
		}

		component := strings.TrimSpace(r.URL.Query().Get("component"))
		if component == "" {
			component = "router"
		}
		if !validLogSpoolComponent(component) {
			writeLogsError(w, http.StatusBadRequest, "invalid_component", "Component must be router, envoy, dashboard, or all.")
			return
		}
		lines := requestedLogTailLines(r.URL.Query().Get("lines"))

		reader, err := openLogSpoolReader(spoolRoot)
		if errors.Is(err, errLogSpoolUnsupported) {
			if pods != nil {
				writeLogsResponse(w, pods.read(r.Context(), component, lines))
				return
			}
			writeLogsResponse(w, LogsResponse{
				DeploymentType: "unsupported",
				Service:        component,
				Components:     []string{},
				Logs:           []LogEntry{},
				Message:        logsUnsupportedNotice,
			})
			return
		}
		response := LogsResponse{
			DeploymentType: "local-spool",
			Service:        component,
			Supported:      true,
			Components:     logSpoolComponents,
			Logs:           []LogEntry{},
		}
		if err != nil {
			response.Error = "The bounded log spool is unavailable or unsafe."
			writeLogsResponse(w, response)
			return
		}
		defer reader.Close()

		entries, readErr := readRequestedLogEntries(reader, component, lines)
		if readErr != nil {
			response.Error = "The bounded log spool could not be read safely."
			writeLogsResponse(w, response)
			return
		}
		response.Logs = entries
		if len(entries) == 0 {
			response.Message = "No recent log entries are available for this service."
		}
		writeLogsResponse(w, response)
	}
}

func validLogSpoolComponent(component string) bool {
	if component == "all" {
		return true
	}
	for _, candidate := range logSpoolComponents {
		if component == candidate {
			return true
		}
	}
	return false
}

func requestedLogTailLines(raw string) int {
	if raw == "" {
		return defaultLogTailLines
	}
	lines, err := strconv.Atoi(raw)
	if err != nil || lines < 1 || lines > maxLogTailLines {
		return defaultLogTailLines
	}
	return lines
}

func readRequestedLogEntries(reader *logSpoolReader, component string, lines int) ([]LogEntry, error) {
	if component != "all" {
		return readComponentLogEntries(reader, component, lines)
	}

	entries := make([]LogEntry, 0, lines)
	base := lines / len(logSpoolComponents)
	remainder := lines % len(logSpoolComponents)
	for index, candidate := range logSpoolComponents {
		allocation := base
		if index < remainder {
			allocation++
		}
		if allocation == 0 {
			continue
		}
		componentEntries, err := readComponentLogEntries(reader, candidate, allocation)
		if err != nil {
			return nil, err
		}
		entries = append(entries, componentEntries...)
	}
	return entries, nil
}

func readComponentLogEntries(reader *logSpoolReader, component string, lines int) ([]LogEntry, error) {
	rawLines, err := reader.Tail(component, lines)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	redacted := redactRuntimeLogLines(rawLines)
	entries := make([]LogEntry, 0, len(redacted))
	for _, line := range redacted {
		if line == "" {
			continue
		}
		entries = append(entries, LogEntry{Line: line, Service: component})
	}
	return entries, nil
}

// boundRuntimeLogLine applies the per-record size limit every log source
// shares.
func boundRuntimeLogLine(record []byte) string {
	if len(record) > maxRuntimeLogLineBytes {
		prefixBytes := maxRuntimeLogLineBytes - len(runtimeLogTruncatedMarker)
		record = append(append([]byte{}, record[:prefixBytes]...), runtimeLogTruncatedMarker...)
	}
	return strings.ToValidUTF8(string(bytes.TrimSuffix(record, []byte{'\r'})), "\uFFFD")
}

func limitLogEntriesByJSONBytes(entries []LogEntry, budget int) []LogEntry {
	if budget <= 0 {
		return []LogEntry{}
	}
	selected := make([]LogEntry, 0, len(entries))
	used := 0
	for index := len(entries) - 1; index >= 0; index-- {
		encoded, err := json.Marshal(entries[index])
		if err != nil {
			continue
		}
		if used+len(encoded)+1 > budget {
			break
		}
		used += len(encoded) + 1
		selected = append(selected, entries[index])
	}
	for left, right := 0, len(selected)-1; left < right; left, right = left+1, right-1 {
		selected[left], selected[right] = selected[right], selected[left]
	}
	return selected
}

// writeLogsResponse keeps the newest entries that fit the response limit
// beside the rest of the answer.
func writeLogsResponse(w http.ResponseWriter, response LogsResponse) {
	entries := response.Logs
	response.Logs = []LogEntry{}
	envelope, err := json.Marshal(response)
	if err != nil {
		writeLogsError(w, http.StatusInternalServerError, "encoding_failed", "The log response could not be encoded safely.")
		return
	}
	response.Logs = limitLogEntriesByJSONBytes(entries, maxLogsResponseBytes-len(envelope)-logsResponseHeadroom)
	response.Count = len(response.Logs)
	data, err := json.Marshal(response)
	if err != nil || len(data)+1 > maxLogsResponseBytes {
		writeLogsError(w, http.StatusInternalServerError, "encoding_failed", "The log response could not be encoded safely.")
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(append(data, '\n'))
}

func writeLogsError(w http.ResponseWriter, status int, code, message string) {
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{
		"error":   code,
		"message": message,
	})
}
