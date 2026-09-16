package handlers

import (
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"strconv"
	"strings"
)

const (
	logSpoolRootEnv       = "VLLM_SR_LOG_SPOOL_DIR"
	defaultLogTailLines   = 100
	maxLogTailLines       = 1000
	maxLogsResponseBytes  = 1 << 20
	logsResponseHeadroom  = 4 << 10
	logsUnsupportedNotice = "Runtime logs are unsupported for this deployment because the bounded shared log spool is not mounted."
)

var logSpoolComponents = []string{"router", "envoy", "dashboard"}

// LogEntry represents one centrally redacted line from a fixed runtime spool.
type LogEntry struct {
	Line    string `json:"line"`
	Service string `json:"service,omitempty"`
}

// LogsResponse is the bounded Dashboard log API contract.
type LogsResponse struct {
	DeploymentType string     `json:"deployment_type"`
	Service        string     `json:"service"`
	Supported      bool       `json:"supported"`
	Logs           []LogEntry `json:"logs"`
	Count          int        `json:"count"`
	Error          string     `json:"error,omitempty"`
	Message        string     `json:"message,omitempty"`
}

// LogsHandler retains the historical constructor shape while deliberately
// ignoring the Router URL. Logs are read only from the bounded shared spool;
// this handler never invokes a container runtime or mounts its control socket.
func LogsHandler(_ string) http.HandlerFunc {
	return newLogsHandler(os.Getenv(logSpoolRootEnv))
}

func newLogsHandler(spoolRoot string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			writeLogsError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only GET is supported.")
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
		response := LogsResponse{
			DeploymentType: "local-spool",
			Service:        component,
			Supported:      true,
			Logs:           []LogEntry{},
		}

		reader, err := openLogSpoolReader(spoolRoot)
		if errors.Is(err, errLogSpoolUnsupported) {
			response.DeploymentType = "unsupported"
			response.Supported = false
			response.Message = logsUnsupportedNotice
			writeLogsResponse(w, response)
			return
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
		response.Logs = limitLogEntriesByJSONBytes(
			entries,
			maxLogsResponseBytes-logsResponseHeadroom,
		)
		response.Count = len(response.Logs)
		if response.Count == 0 {
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
		return readComponentLogEntries(reader, component, lines, false)
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
		componentEntries, err := readComponentLogEntries(reader, candidate, allocation, true)
		if err != nil {
			return nil, err
		}
		entries = append(entries, componentEntries...)
	}
	return entries, nil
}

func readComponentLogEntries(
	reader *logSpoolReader,
	component string,
	lines int,
	prefixService bool,
) ([]LogEntry, error) {
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
		if prefixService {
			line = "[" + component + "] " + line
		}
		entries = append(entries, LogEntry{Line: line, Service: component})
	}
	return entries, nil
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

func writeLogsResponse(w http.ResponseWriter, response LogsResponse) {
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
