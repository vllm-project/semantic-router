// Generate label-free judge requests with the exact serving prompt builder.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

type sourceCandidate struct {
	Stage string          `json:"stage"`
	Model string          `json:"model"`
	Body  json.RawMessage `json:"body"`
}

type sourceRecord struct {
	RecordID   string            `json:"record_id"`
	GroupID    string            `json:"group_id"`
	Cohort     string            `json:"cohort"`
	Split      string            `json:"split"`
	Request    json.RawMessage   `json:"request"`
	Candidates []sourceCandidate `json:"candidates"`
}

type outputRecord struct {
	RecordID string          `json:"record_id"`
	GroupID  string          `json:"group_id"`
	Cohort   string          `json:"cohort"`
	Split    string          `json:"split"`
	Request  json.RawMessage `json:"request"`
}

func generate(input io.Reader, output io.Writer, model string) error {
	if model == "" {
		return fmt.Errorf("an explicit judge model is required")
	}
	decoder := json.NewDecoder(input)
	decoder.DisallowUnknownFields()
	encoder := json.NewEncoder(output)
	seen := map[string]bool{}
	for {
		var record sourceRecord
		if err := decoder.Decode(&record); err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("decode source record: %w", err)
		}
		if record.RecordID == "" || record.GroupID == "" || seen[record.RecordID] {
			return fmt.Errorf("source record needs a unique ID and source group")
		}
		seen[record.RecordID] = true
		request, err := systemone.ParseNativeRequest(record.Request)
		if err != nil {
			return fmt.Errorf("invalid native request %s: %w", record.RecordID, err)
		}
		candidates := make([]systemone.Candidate, len(record.Candidates))
		for i, candidate := range record.Candidates {
			candidates[i] = systemone.Candidate{Stage: candidate.Stage, Model: candidate.Model, Body: candidate.Body}
		}
		stage := config.CascadeStage{Model: model, Generation: &config.NativeGenerationConfig{MaxOutputTokens: 128}}
		body, err := systemone.JudgeRequest(request, stage, candidates)
		if err != nil {
			return fmt.Errorf("judge request %s: %w", record.RecordID, err)
		}
		result := outputRecord{record.RecordID, record.GroupID, record.Cohort, record.Split, body}
		if err := encoder.Encode(result); err != nil {
			return err
		}
	}
}

func main() {
	model := flag.String("model", "", "Concrete served judge model name")
	flag.Parse()
	if err := generate(os.Stdin, os.Stdout, *model); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
