package store

import (
	"encoding/json"
	"errors"
	"fmt"
	"unicode/utf8"
)

// MaxRecordBytes caps one serialized replay record.
const MaxRecordBytes = 1 << 20

var ErrRecordTooLarge = errors.New("router replay record exceeds the size limit")

type sizeCut struct {
	text *string
	mark func()
}

// encodeRecord marshals record, cutting text until it fits in limit bytes.
// Raw tool fields are cut first.
func encodeRecord(record *Record, limit int) ([]byte, error) {
	cloned := false
	for {
		data, err := json.Marshal(record)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal record: %w", err)
		}
		excess := len(data) - limit
		if excess <= 0 {
			return data, nil
		}
		if !cloned {
			record.ToolTrace = cloneToolTrace(record.ToolTrace)
			cloned = true
		}
		// Truncation flags add a few bytes, so measure again.
		if !cutRecord(record, excess) {
			return nil, fmt.Errorf("%w: %d bytes over %d", ErrRecordTooLarge, excess, limit)
		}
	}
}

// JSON escaping never shortens a string, so cutting n bytes of text saves at
// least n bytes.
func cutRecord(record *Record, excess int) bool {
	for _, cut := range recordSizeCuts(record) {
		if removed := cutText(cut.text, excess); removed > 0 {
			excess -= removed
			cut.mark()
		}
		if excess <= 0 {
			return true
		}
	}
	return false
}

func recordSizeCuts(r *Record) []sizeCut {
	var cuts []sizeCut
	if r.ToolTrace != nil {
		steps := r.ToolTrace.Steps
		for i := range steps {
			step := &steps[i]
			mark := func() { step.Truncated = true }
			cuts = append(cuts, sizeCut{&step.RawOutput, mark}, sizeCut{&step.RawArguments, mark})
		}
		for i := range steps {
			step := &steps[i]
			mark := func() { step.Truncated = true }
			cuts = append(cuts, sizeCut{&step.Output, mark}, sizeCut{&step.Arguments, mark}, sizeCut{&step.Text, mark})
		}
	}
	return append(cuts,
		sizeCut{&r.ResponseBody, func() { r.ResponseBodyTruncated = true }},
		sizeCut{&r.RequestBody, func() { r.RequestBodyTruncated = true }},
		sizeCut{&r.Prompt, func() { r.PromptTruncated = true }},
		sizeCut{&r.ToolDefinitions, func() { r.ToolDefinitionsTruncated = true }},
	)
}

func cutText(text *string, excess int) int {
	keep := max(len(*text)-excess, 0)
	for keep > 0 && !utf8.RuneStart((*text)[keep]) {
		keep--
	}
	removed := len(*text) - keep
	*text = (*text)[:keep]
	return removed
}

// fitToolTrace applies the record limit to a trace written on its own.
func fitToolTrace(trace ToolTrace) (ToolTrace, error) {
	holder := Record{ToolTrace: &trace}
	if _, err := encodeRecord(&holder, MaxRecordBytes); err != nil {
		return ToolTrace{}, err
	}
	return *holder.ToolTrace, nil
}
