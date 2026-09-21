package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
)

type testCase struct {
	ID       string  `json:"id"`
	Group    string  `json:"group"`
	State    string  `json:"state"`
	Expected *string `json:"expected"`
}

type runOptions struct {
	Inputs, Question, Output, Location, Revision string
	Live                                         bool
	Timeout                                      time.Duration
	MaxCases                                     int
}

type record struct {
	Schema            string  `json:"schema"`
	Contract          string  `json:"contract"`
	ID                string  `json:"id"`
	Group             string  `json:"group"`
	Expected          *string `json:"expected"`
	Timestamp         string  `json:"timestamp"`
	Location          string  `json:"location"`
	Revision          string  `json:"revision"`
	DatasetSHA256     string  `json:"dataset_sha256"`
	QuestionSHA256    string  `json:"question_sha256"`
	Request           request `json:"request"`
	RawResponse       string  `json:"raw_response"`
	ElapsedMS         float64 `json:"elapsed_ms"`
	TimeoutMS         int64   `json:"timeout_ms"`
	Attempts          int     `json:"attempts"`
	Status            int     `json:"http_status"`
	Valid             bool    `json:"contract_valid"`
	Correct           *bool   `json:"correct,omitempty"`
	Error             string  `json:"error,omitempty"`
	ErrorKind         string  `json:"error_kind,omitempty"`
	ResponseTruncated bool    `json:"response_truncated,omitempty"`
}

func loadCases(data []byte, labels map[string]string, limit int) ([]testCase, error) {
	if limit <= 0 {
		return nil, fmt.Errorf("max-cases must be positive")
	}
	var cases []testCase
	seen := map[string]bool{}
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 4096), 1<<20)
	for scanner.Scan() {
		if strings.TrimSpace(scanner.Text()) == "" {
			continue
		}
		var c testCase
		if err := json.Unmarshal(scanner.Bytes(), &c); err != nil {
			return nil, fmt.Errorf("invalid case JSON: %w", err)
		}
		if c.ID == "" || c.Group == "" || strings.TrimSpace(c.State) == "" || seen[c.ID] {
			return nil, fmt.Errorf("case IDs must be unique and id/group/state must be nonempty")
		}
		if c.Expected != nil {
			if _, ok := labels[*c.Expected]; !ok {
				return nil, fmt.Errorf("expected label is not configured")
			}
		}
		seen[c.ID] = true
		cases = append(cases, c)
		if len(cases) > limit {
			return nil, fmt.Errorf("dataset exceeds max-cases; no calls made")
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(cases) == 0 {
		return nil, fmt.Errorf("dataset is empty")
	}
	return cases, nil
}

func run(ctx context.Context, opts runOptions) error {
	if !opts.Live {
		return fmt.Errorf("no calls made: --live is required for paid evaluation")
	}
	if opts.Inputs == "" || opts.Question == "" || opts.Output == "" || strings.TrimSpace(opts.Location) == "" || strings.TrimSpace(opts.Revision) == "" {
		return fmt.Errorf("inputs, question, output, location, and revision are required")
	}
	key := strings.TrimSpace(os.Getenv("TYPESAFE_API_KEY"))
	if key == "" {
		return fmt.Errorf("TYPESAFE_API_KEY is required; never pass it as a CLI argument")
	}
	data, err := os.ReadFile(opts.Inputs)
	if err != nil {
		return err
	}
	qdata, err := os.ReadFile(opts.Question)
	if err != nil {
		return err
	}
	var q question
	if err = json.Unmarshal(qdata, &q); err != nil {
		return err
	}
	if err = validateQuestion(q); err != nil {
		return err
	}
	cases, err := loadCases(data, q.Criteria, opts.MaxCases)
	if err != nil {
		return err
	}
	a, err := newAdapter("https://api.typesafe.ai", key, q, opts.Timeout)
	if err != nil {
		return err
	}
	defer a.client.Close()
	file, err := os.OpenFile(opts.Output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	runErr := collect(ctx, a, cases, opts, fmt.Sprintf("%x", sha256.Sum256(data)), fmt.Sprintf("%x", sha256.Sum256(qdata)), json.NewEncoder(file))
	return errors.Join(runErr, file.Close())
}

func collect(ctx context.Context, a *adapter, cases []testCase, opts runOptions, datasetDigest, questionDigest string, encoder *json.Encoder) error {
	for _, c := range cases {
		if err := ctx.Err(); err != nil {
			return err
		}
		r := record{
			Schema: "jev-research-record.v1", Contract: contract, ID: c.ID, Group: c.Group, Expected: c.Expected,
			Timestamp: time.Now().UTC().Format(time.RFC3339Nano), Location: opts.Location, Revision: opts.Revision,
			DatasetSHA256: datasetDigest, QuestionSHA256: questionDigest, TimeoutMS: opts.Timeout.Milliseconds(),
		}
		start := time.Now()
		req, wire, out, err := a.evaluate(ctx, c.State)
		r.ElapsedMS = float64(time.Since(start).Microseconds()) / 1000
		r.Request, r.RawResponse, r.Status, r.Attempts = req, string(wire.Body), wire.StatusCode, wire.Attempts
		r.Valid = err == nil
		if err != nil {
			r.ErrorKind, r.Error = "contract", err.Error()
			var transport *connector.Error
			if errors.As(err, &transport) {
				r.ErrorKind, r.Status, r.Attempts = string(transport.Kind), transport.StatusCode, transport.Attempt
				body, truncated := transport.ResponseBody()
				r.RawResponse, r.ResponseTruncated = string(body), truncated
				// Do not serialize arbitrary transport causes or credential headers.
				r.Error = "shared connector rejected or failed the request"
			}
		} else if c.Expected != nil {
			correct := out.Answers["intent"].Choice == *c.Expected
			r.Correct = &correct
		}
		if writeErr := encoder.Encode(r); writeErr != nil {
			return writeErr
		}
		// Preserve the failure record and stop: never spend through failures or
		// silently drop invalid cases from a later comparison.
		if err != nil {
			return fmt.Errorf("probe stopped after recorded failure for case %s", c.ID)
		}
	}
	return nil
}
