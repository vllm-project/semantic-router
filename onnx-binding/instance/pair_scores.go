//go:build !windows && cgo && (amd64 || arm64)

package instance

/*
#include "ort_instance.h"
*/
import "C"

import (
	"encoding/json"
	"unicode/utf8"
)

type PairScorer struct{ *owner }

func LoadPairScorer(options Options, selection PairScorerSelection) (*PairScorer, error) {
	if selection.Layer < 0 || selection.Dimension < 0 {
		return nil, &Error{Kind: "configuration", Message: "negative pair scorer selection"}
	}
	payload, err := json.Marshal(selection)
	if err != nil {
		return nil, err
	}
	var result *owner
	err = withText(string(payload), func(selection *C.char) error {
		var loadErr error
		result, loadErr = load(options, func(options *C.char) C.OrtInstanceResult { return C.ort_instance_load_pair_scorer(options, selection) })
		return loadErr
	})
	if err != nil {
		return nil, err
	}
	return &PairScorer{result}, nil
}

func (m *PairScorer) Clone() (*PairScorer, error) {
	if m == nil {
		return nil, &Error{Kind: "closed", Message: "nil pair scorer"}
	}
	cloned, err := m.owner.clone()
	if err != nil {
		return nil, err
	}
	return &PairScorer{cloned}, nil
}

func (m *PairScorer) ScorePairs(pairs []TextPair) (PairScores, error) {
	var result PairScores
	if m == nil {
		return result, &Error{Kind: "closed", Message: "nil pair scorer"}
	}
	for _, pair := range pairs {
		if !utf8.ValidString(pair.Query) || !utf8.ValidString(pair.Document) {
			return result, &Error{Kind: "configuration", Message: "pair input must be valid UTF-8"}
		}
	}
	payload, err := json.Marshal(pairs)
	if err != nil {
		return result, err
	}
	err = m.withHandle(func(handle C.uint64_t) error {
		return withText(string(payload), func(pairs *C.char) error { return decode(C.ort_instance_score_pairs(handle, pairs), &result) })
	})
	return result, err
}
