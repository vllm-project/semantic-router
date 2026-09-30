//go:build !windows && cgo && (amd64 || arm64)

package instance

/*
#include "ort_instance.h"
*/
import "C"

type GroundedClassifier struct{ *owner }

func LoadGroundedClassifier(options Options) (*GroundedClassifier, error) {
	model, err := load(options, func(p *C.char) C.OrtInstanceResult { return C.ort_instance_load_grounded(p) })
	if err != nil {
		return nil, err
	}
	return &GroundedClassifier{model}, nil
}

func (m *GroundedClassifier) Detect(context, question, answer string) (TokenSpans, error) {
	var result TokenSpans
	if m == nil {
		return result, &Error{Kind: "closed", Message: "nil grounding classifier"}
	}
	err := m.withHandle(func(handle C.uint64_t) error {
		return withText(context, func(context *C.char) error {
			return withText(question, func(question *C.char) error {
				return withText(answer, func(answer *C.char) error {
					return decode(C.ort_instance_grounded(handle, context, question, answer), &result)
				})
			})
		})
	})
	return result, err
}
