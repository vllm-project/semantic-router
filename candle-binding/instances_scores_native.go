//go:build !windows && cgo && (amd64 || arm64 || riscv64)

package candle_binding

/*
#include <stdint.h>
#include <stddef.h>
extern char* candle_instance_score(uint64_t, const char*);
extern char* candle_instance_token_windows(uint64_t, const char*, size_t, size_t);
extern char* candle_instance_classify_windows(uint64_t, const char*, size_t, size_t);
extern char* candle_instance_score_windows(uint64_t, const char*, size_t, size_t);
*/
import "C"

func nativeInstanceScore(handle uint64, text string) (LabelScoresOutput, error) {
	args, free, err := instanceStrings(text)
	if err != nil {
		return LabelScoresOutput{}, err
	}
	defer free()
	return decodeInstanceResult[LabelScoresOutput](C.candle_instance_score(C.uint64_t(handle), args[0]))
}

func nativeInstanceClassifyWindows(handle uint64, text string, options SequenceWindowOptions) (WindowedClassificationOutput, error) {
	args, free, err := instanceStrings(text)
	if err != nil {
		return WindowedClassificationOutput{}, err
	}
	defer free()
	return decodeInstanceResult[WindowedClassificationOutput](C.candle_instance_classify_windows(C.uint64_t(handle), args[0], C.size_t(options.Size), C.size_t(options.Overlap)))
}

func nativeInstanceScoreWindows(handle uint64, text string, options SequenceWindowOptions) (WindowedLabelScoresOutput, error) {
	args, free, err := instanceStrings(text)
	if err != nil {
		return WindowedLabelScoresOutput{}, err
	}
	defer free()
	return decodeInstanceResult[WindowedLabelScoresOutput](C.candle_instance_score_windows(C.uint64_t(handle), args[0], C.size_t(options.Size), C.size_t(options.Overlap)))
}

func nativeInstanceTokenWindows(handle uint64, text string, options SequenceWindowOptions) (WindowedTokenOutput, error) {
	args, free, err := instanceStrings(text)
	if err != nil {
		return WindowedTokenOutput{}, err
	}
	defer free()
	return decodeInstanceResult[WindowedTokenOutput](C.candle_instance_token_windows(C.uint64_t(handle), args[0], C.size_t(options.Size), C.size_t(options.Overlap)))
}
