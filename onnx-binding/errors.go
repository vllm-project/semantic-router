package onnx_binding

import "errors"

// ErrBackendUnavailable identifies an inference operation the linked backend
// cannot provide. Consumers can preserve this cause with errors.Is.
var ErrBackendUnavailable = errors.New("onnx: requested native inference operation unavailable")
