package candle_binding

import (
	"errors"
	"fmt"
	"strings"
)

var errEmbeddedNULByte = errors.New("input contains an embedded NUL byte")

type cStringInput struct {
	field string
	value string
}

func validateCStringInputs(inputs ...cStringInput) error {
	for _, input := range inputs {
		if strings.IndexByte(input.value, 0) >= 0 {
			return fmt.Errorf("%w: %s", errEmbeddedNULByte, input.field)
		}
	}
	return nil
}
