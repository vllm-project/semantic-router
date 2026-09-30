package benchmarks

import (
	"errors"
	"io/fs"
)

func missingBenchModels(err error) bool {
	return err != nil && errors.Is(err, fs.ErrNotExist)
}
