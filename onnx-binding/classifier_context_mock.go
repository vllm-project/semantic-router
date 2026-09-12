//go:build windows || !cgo || (!amd64 && !arm64)

package onnx_binding

func InitMmBert32KIntentClassifierWithMaxSequenceLength(modelPath string, useCPU bool, maxSequenceLength int) error {
	return InitMmBert32KIntentClassifier(modelPath, useCPU)
}
func InitMmBert32KFactcheckClassifierWithMaxSequenceLength(modelPath string, useCPU bool, maxSequenceLength int) error {
	return InitMmBert32KFactcheckClassifier(modelPath, useCPU)
}
func InitMmBert32KJailbreakClassifierWithMaxSequenceLength(modelPath string, useCPU bool, maxSequenceLength int) error {
	return InitMmBert32KJailbreakClassifier(modelPath, useCPU)
}
func InitMmBert32KFeedbackClassifierWithMaxSequenceLength(modelPath string, useCPU bool, maxSequenceLength int) error {
	return InitMmBert32KFeedbackClassifier(modelPath, useCPU)
}
func InitMmBert32KPIIClassifierWithMaxSequenceLength(modelPath string, useCPU bool, maxSequenceLength int) error {
	return InitMmBert32KPIIClassifier(modelPath, useCPU)
}
func InitMmBert32KModalityClassifierWithMaxSequenceLength(modelPath string, useCPU bool, maxSequenceLength int) error {
	return InitMmBert32KModalityClassifier(modelPath, useCPU)
}
