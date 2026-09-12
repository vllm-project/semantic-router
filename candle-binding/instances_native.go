//go:build !windows && cgo && (amd64 || arm64)

package candle_binding

/*
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
extern char* candle_instance_load_guard(const char*);
extern char* candle_instance_load_generative(const char*);
extern char* candle_instance_guard(uint64_t, const char*, const char*);
extern char* candle_instance_generative(uint64_t, const char*, const char*, const char*, bool);
extern char* candle_instance_load_backbone(const char*);
extern char* candle_instance_load_sequence(const char*);
extern char* candle_instance_load_token(const char*);
extern char* candle_instance_load_nli(const char*);
extern char* candle_instance_load_hallucination(const char*);
extern char* candle_instance_load_embedding(const char*);
extern char* candle_instance_clone(uint64_t);
extern char* candle_instance_close(uint64_t);
extern char* candle_instance_info(uint64_t);
extern char* candle_instance_bind_head(uint64_t, const char*, const char*);
extern char* candle_instance_sequence(uint64_t, const char*);
extern char* candle_instance_tokens(uint64_t, const char*);
extern char* candle_instance_nli(uint64_t, const char*, const char*);
extern char* candle_instance_hallucination(uint64_t, const char*, const char*, const char*, float);
extern char* candle_instance_embedding(uint64_t, const char*, size_t, size_t);
extern char* candle_instance_image(uint64_t, const uint8_t*, size_t, size_t);
extern char* candle_instance_audio(uint64_t, const float*, size_t, size_t, size_t, size_t);
extern void candle_instance_free_string(char*);
extern char* candle_instance_text_windows(unsigned long long handle, const char* text, size_t max_tokens);
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"strings"
	"unsafe"
)

func decodeInstanceResult[T any](ptr *C.char) (T, error) {
	var envelope struct {
		Value T      `json:"value"`
		Error string `json:"error"`
	}
	if ptr == nil {
		return envelope.Value, fmt.Errorf("candle returned a null instance response")
	}
	defer C.candle_instance_free_string(ptr)
	if err := json.Unmarshal([]byte(C.GoString(ptr)), &envelope); err != nil {
		return envelope.Value, err
	}
	if envelope.Error != "" {
		return envelope.Value, instanceError(envelope.Error)
	}
	return envelope.Value, nil
}

func instanceStrings(values ...string) ([]*C.char, func(), error) {
	for _, value := range values {
		if strings.ContainsRune(value, 0) {
			return nil, nil, &InstanceError{Code: "configuration", Message: "embedded NUL is not supported by the native string ABI"}
		}
	}
	ptrs := make([]*C.char, len(values))
	for i, value := range values {
		ptrs[i] = C.CString(value)
	}
	return ptrs, func() {
		for _, ptr := range ptrs {
			C.free(unsafe.Pointer(ptr))
		}
	}, nil
}

func nativeInstanceLoad(options InstanceOptions, task string) (uint64, error) {
	payload, err := json.Marshal(options)
	if err != nil {
		return 0, err
	}
	args, free, err := instanceStrings(string(payload))
	if err != nil {
		return 0, err
	}
	defer free()
	var result *C.char
	switch task {
	case "backbone":
		result = C.candle_instance_load_backbone(args[0])
	case "guard":
		result = C.candle_instance_load_guard(args[0])
	case "generative":
		result = C.candle_instance_load_generative(args[0])
	case "sequence":
		result = C.candle_instance_load_sequence(args[0])
	case "token":
		result = C.candle_instance_load_token(args[0])
	case "nli":
		result = C.candle_instance_load_nli(args[0])
	case "hallucination":
		result = C.candle_instance_load_hallucination(args[0])
	case "embedding":
		result = C.candle_instance_load_embedding(args[0])
	default:
		return 0, fmt.Errorf("unsupported instance task %q", task)
	}
	return decodeInstanceResult[uint64](result)
}

func nativeInstanceClose(handle uint64) error {
	_, err := decodeInstanceResult[any](C.candle_instance_close(C.uint64_t(handle)))
	return err
}

func nativeInstanceClone(handle uint64) (uint64, error) {
	return decodeInstanceResult[uint64](C.candle_instance_clone(C.uint64_t(handle)))
}

func nativeInstanceInfo(handle uint64) (InstanceInfo, error) {
	return decodeInstanceResult[InstanceInfo](C.candle_instance_info(C.uint64_t(handle)))
}

func nativeInstanceBindHead(handle uint64, path, task string) (uint64, error) {
	args, free, err := instanceStrings(path, task)
	if err != nil {
		return 0, err
	}
	defer free()
	return decodeInstanceResult[uint64](C.candle_instance_bind_head(C.uint64_t(handle), args[0], args[1]))
}

func nativeInstanceSequence(handle uint64, text string) (DistributionOutput, error) {
	args, free, err := instanceStrings(text)
	if err != nil {
		return DistributionOutput{}, err
	}
	defer free()
	return decodeInstanceResult[DistributionOutput](C.candle_instance_sequence(C.uint64_t(handle), args[0]))
}

func nativeInstanceTokens(handle uint64, text string) (TokenOutput, error) {
	args, free, err := instanceStrings(text)
	if err != nil {
		return TokenOutput{}, err
	}
	defer free()
	return decodeInstanceResult[TokenOutput](C.candle_instance_tokens(C.uint64_t(handle), args[0]))
}

func nativeInstanceNLI(handle uint64, premise, hypothesis string) (DistributionOutput, error) {
	args, free, err := instanceStrings(premise, hypothesis)
	if err != nil {
		return DistributionOutput{}, err
	}
	defer free()
	return decodeInstanceResult[DistributionOutput](C.candle_instance_nli(C.uint64_t(handle), args[0], args[1]))
}

func nativeInstanceHallucination(handle uint64, context, question, answer string, threshold float32) (HallucinationOutput, error) {
	args, free, err := instanceStrings(context, question, answer)
	if err != nil {
		return HallucinationOutput{}, err
	}
	defer free()
	return decodeInstanceResult[HallucinationOutput](C.candle_instance_hallucination(C.uint64_t(handle), args[0], args[1], args[2], C.float(threshold)))
}

func nativeInstanceEmbedding(handle uint64, text string, dimension, layer int) (InstanceEmbeddingOutput, error) {
	args, free, err := instanceStrings(text)
	if err != nil {
		return InstanceEmbeddingOutput{}, err
	}
	defer free()
	return decodeInstanceResult[InstanceEmbeddingOutput](C.candle_instance_embedding(C.uint64_t(handle), args[0], C.size_t(dimension), C.size_t(layer)))
}

func nativeInstanceImage(handle uint64, encoded []byte, dimension int) (InstanceEmbeddingOutput, error) {
	return decodeInstanceResult[InstanceEmbeddingOutput](C.candle_instance_image(C.uint64_t(handle), (*C.uint8_t)(unsafe.Pointer(&encoded[0])), C.size_t(len(encoded)), C.size_t(dimension)))
}

func nativeInstanceAudio(handle uint64, mel []float32, melBins, frames, dimension int) (InstanceEmbeddingOutput, error) {
	return decodeInstanceResult[InstanceEmbeddingOutput](C.candle_instance_audio(C.uint64_t(handle), (*C.float)(unsafe.Pointer(&mel[0])), C.size_t(len(mel)), C.size_t(melBins), C.size_t(frames), C.size_t(dimension)))
}

func nativeInstanceGuard(handle uint64, text, mode string) (GuardOutput, error) {
	args, free, err := instanceStrings(text, mode)
	if err != nil {
		return GuardOutput{}, err
	}
	defer free()
	return decodeInstanceResult[GuardOutput](C.candle_instance_guard(C.uint64_t(handle), args[0], args[1]))
}

func nativeInstanceGenerative(handle uint64, text, adapter string, categories []string, multiToken bool) (GenerativeOutput, error) {
	if categories == nil {
		categories = []string{}
	}
	payload, err := json.Marshal(categories)
	if err != nil {
		return GenerativeOutput{}, err
	}
	args, free, err := instanceStrings(text, adapter, string(payload))
	if err != nil {
		return GenerativeOutput{}, err
	}
	defer free()
	return decodeInstanceResult[GenerativeOutput](C.candle_instance_generative(C.uint64_t(handle), args[0], args[1], args[2], C.bool(multiToken)))
}

func nativeInstanceTextWindows(handle uint64, text string, limit int) ([][2]int, error) {
	if strings.ContainsRune(text, 0) {
		return nil, fmt.Errorf("text contains NUL")
	}
	value := C.CString(text)
	defer C.free(unsafe.Pointer(value))
	return decodeInstanceResult[[][2]int](C.candle_instance_text_windows(C.ulonglong(handle), value, C.size_t(limit)))
}
