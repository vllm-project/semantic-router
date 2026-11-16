package services

import (
	"errors"
	"io"
)

// classifierAPIResources are what a standalone service's published
// classifiers hold: the classifiers and the model lease that serves them.
type classifierAPIResources struct {
	classifiers io.Closer
	lease       io.Closer
}

// Close retires the generation's bindings before the lease that serves them.
func (r classifierAPIResources) Close() error {
	return errors.Join(r.classifiers.Close(), r.lease.Close())
}
