#ifndef OPENVINO_SR_OWNED_MODEL_H
#define OPENVINO_SR_OWNED_MODEL_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Each handle owns its model, tokenizer and inference requests. Callers must
// finish inference before closing a handle; handles never use the legacy globals.
typedef struct OVEmbeddingHandle OVEmbeddingHandle;
typedef struct OVClassifierHandle OVClassifierHandle;

typedef struct {
    float* values;
    int length;
    int original_tokens;
    int processed_tokens;
    // 0: success, 1: input budget exceeded, 2: inference failure.
    int status;
} OVOwnedResult;

OVEmbeddingHandle* ov_embedding_open(const char* path, const char* device);
OVClassifierHandle* ov_classifier_open(const char* path, const char* device, int classes);
OVOwnedResult ov_embedding_run(OVEmbeddingHandle* handle, const char* text,
                               int max_tokens, bool reject_overflow);
OVOwnedResult ov_classifier_run(OVClassifierHandle* handle, const char* text,
                                int max_tokens, bool reject_overflow);
void ov_embedding_close(OVEmbeddingHandle* handle);
void ov_classifier_close(OVClassifierHandle* handle);
void ov_owned_result_free(OVOwnedResult result);

#ifdef __cplusplus
}
#endif
#endif
