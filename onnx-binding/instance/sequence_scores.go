//go:build !windows && cgo && (amd64 || arm64)

package instance

/*
#include "ort_instance.h"
*/
import "C"

type LabelScorer struct{ *owner }

func LoadLabelScorer(options Options) (*LabelScorer, error) {
	o, err := load(options, func(p *C.char) C.OrtInstanceResult { return C.ort_instance_load_label_scores(p) })
	if err != nil {
		return nil, err
	}
	return &LabelScorer{o}, nil
}

func (m *LabelScorer) Clone() (*LabelScorer, error) {
	o, err := m.owner.clone()
	if err != nil {
		return nil, err
	}
	return &LabelScorer{o}, nil
}

func (m *LabelScorer) Score(text string) (LabelScores, error) {
	var output LabelScores
	err := m.withHandle(func(handle C.uint64_t) error {
		return withText(text, func(text *C.char) error { return decode(C.ort_instance_score(handle, text), &output) })
	})
	return output, err
}

func (m *SequenceClassifier) ClassifyWindows(text string, options SequenceWindowOptions) (WindowedClassificationOutput, error) {
	var output WindowedClassificationOutput
	if err := validSequenceWindow(options); err != nil {
		return output, err
	}
	err := m.withHandle(func(handle C.uint64_t) error {
		return withText(text, func(text *C.char) error {
			return decode(C.ort_instance_classify_windows(handle, text, C.size_t(options.Size), C.size_t(options.Overlap)), &output)
		})
	})
	return output, err
}

func (m *LabelScorer) ScoreWindows(text string, options SequenceWindowOptions) (WindowedLabelScoresOutput, error) {
	var output WindowedLabelScoresOutput
	if err := validSequenceWindow(options); err != nil {
		return output, err
	}
	err := m.withHandle(func(handle C.uint64_t) error {
		return withText(text, func(text *C.char) error {
			return decode(C.ort_instance_score_windows(handle, text, C.size_t(options.Size), C.size_t(options.Overlap)), &output)
		})
	})
	return output, err
}

func (m *TokenClassifier) DetectWindows(text string, options SequenceWindowOptions) (WindowedTokenOutput, error) {
	var output WindowedTokenOutput
	if err := validSequenceWindow(options); err != nil {
		return output, err
	}
	err := m.withHandle(func(handle C.uint64_t) error {
		return withText(text, func(text *C.char) error {
			return decode(C.ort_instance_token_windows(handle, text, C.size_t(options.Size), C.size_t(options.Overlap)), &output)
		})
	})
	return output, err
}
