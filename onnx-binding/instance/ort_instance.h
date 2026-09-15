#ifndef ORT_INSTANCE_H
#define ORT_INSTANCE_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t handle;
    char *payload;
    char *error;
    char *error_kind;
} OrtInstanceResult;

OrtInstanceResult ort_instance_load_sequence(const char *options);
OrtInstanceResult ort_instance_load_token(const char *options);
OrtInstanceResult ort_instance_load_embedding(const char *options);
OrtInstanceResult ort_instance_load_multimodal(const char *options);
OrtInstanceResult ort_instance_clone(uint64_t handle);
void ort_instance_close(uint64_t handle);
OrtInstanceResult ort_instance_info(uint64_t handle);
OrtInstanceResult ort_instance_finish_profiling(uint64_t handle);
OrtInstanceResult ort_instance_classify(uint64_t handle, const char *text);
OrtInstanceResult ort_instance_detect_tokens(uint64_t handle, const char *text);
OrtInstanceResult ort_instance_text_windows(uint64_t handle, const char *text, size_t max_tokens);
OrtInstanceResult ort_instance_encode_text(uint64_t handle, const char *text, size_t layer, size_t dimension);
OrtInstanceResult ort_instance_encode_image(uint64_t handle, const float *pixels, size_t length, size_t height, size_t width, size_t dimension);
OrtInstanceResult ort_instance_encode_image_bytes(uint64_t handle, const uint8_t *bytes, size_t length, size_t dimension);
OrtInstanceResult ort_instance_encode_audio(uint64_t handle, const float *mel, size_t length, size_t n_mels, size_t frames, size_t dimension);
void ort_instance_result_free(OrtInstanceResult result);
#endif
