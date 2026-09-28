//go:build !onnx

package config

// ROCm images override this build default with -X while retaining one binary.
var defaultModelProvider = "candle"
