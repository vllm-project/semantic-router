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

OrtInstanceResult ort_instance_load_pair_scorer(const char* options, const char* selection);
OrtInstanceResult ort_instance_score_pairs(uint64_t handle, const char* pairs);
OrtInstanceResult ort_instance_load_sequence(const char *options);
OrtInstanceResult ort_instance_load_label_scores(const char *options);
OrtInstanceResult ort_instance_load_token(const char *options);
OrtInstanceResult ort_instance_load_grounded(const char *options);
OrtInstanceResult ort_instance_grounded(uint64_t handle, const char *context, const char *question, const char *answer);
OrtInstanceResult ort_instance_load_embedding(const char *options);
OrtInstanceResult ort_instance_load_multimodal(const char *options);
OrtInstanceResult ort_instance_load_omni(const char *options);
OrtInstanceResult ort_instance_encode_audio_pcm(uint64_t handle, const float *pcm, size_t length, size_t sample_rate, size_t channels, size_t dimension);
OrtInstanceResult ort_instance_clone(uint64_t handle);
void ort_instance_close(uint64_t handle);
OrtInstanceResult ort_instance_info(uint64_t handle);
OrtInstanceResult ort_instance_finish_profiling(uint64_t handle);
OrtInstanceResult ort_instance_classify(uint64_t handle, const char *text);
OrtInstanceResult ort_instance_detect_tokens(uint64_t handle, const char *text);
OrtInstanceResult ort_instance_text_windows(uint64_t handle, const char *text, size_t max_tokens);
OrtInstanceResult ort_instance_encode_text(uint64_t handle, const char *text, size_t layer, size_t dimension);
OrtInstanceResult ort_instance_embedding_descriptor(uint64_t handle, size_t layer, size_t dimension);
OrtInstanceResult ort_instance_encode_image(uint64_t handle, const float *pixels, size_t length, size_t height, size_t width, size_t dimension);
OrtInstanceResult ort_instance_encode_image_bytes(uint64_t handle, const uint8_t *bytes, size_t length, size_t dimension);
OrtInstanceResult ort_instance_encode_audio(uint64_t handle, const float *mel, size_t length, size_t n_mels, size_t frames, size_t dimension);
void ort_instance_result_free(OrtInstanceResult result);
OrtInstanceResult ort_instance_score(uint64_t handle, const char *text);
OrtInstanceResult ort_instance_classify_windows(uint64_t handle, const char *text, size_t size, size_t overlap);
OrtInstanceResult ort_instance_score_windows(uint64_t handle, const char *text, size_t size, size_t overlap);
OrtInstanceResult ort_instance_token_windows(uint64_t handle, const char* text, size_t size, size_t overlap);
#endif
