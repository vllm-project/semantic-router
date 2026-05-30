//go:build !windows && cgo

package classification

/*
#include <stdlib.h>
#include <stdbool.h>

bool init_lora_unified_classifier(const char* intent_model_path, const char* pii_model_path,
                                  const char* security_model_path, const char* architecture, bool use_cpu);
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (uc *UnifiedClassifier) classificationMode() (bool, error) {
	uc.mu.Lock()
	defer uc.mu.Unlock()

	if !uc.initialized {
		return false, fmt.Errorf("unified classifier not initialized")
	}
	capabilities := CurrentNativeBackendCapabilities()
	if uc.useLoRA && !capabilities.LoRABatchClassification {
		return false, fmt.Errorf("native backend %q does not support LoRA unified batch classification", capabilities.Name)
	}
	if !uc.useLoRA && !capabilities.UnifiedBatchClassification {
		return false, fmt.Errorf("native backend %q does not support unified batch classification", capabilities.Name)
	}

	return uc.useLoRA, nil
}

func (uc *UnifiedClassifier) ensureLoRAInitialized() error {
	uc.mu.Lock()
	if uc.loraInitialized {
		uc.mu.Unlock()
		return nil
	}
	uc.mu.Unlock()

	uc.mu.Lock()
	defer uc.mu.Unlock()

	if uc.loraInitialized {
		return nil
	}

	if err := uc.initializeLoRABindings(); err != nil {
		return err
	}

	uc.loraInitialized = true
	return nil
}

// initializeLoRABindings initializes the LoRA C bindings lazily.
func (uc *UnifiedClassifier) initializeLoRABindings() error {
	if uc.testInitializeLoRA != nil {
		return uc.testInitializeLoRA()
	}

	if !CurrentNativeBackendCapabilities().LoRABatchClassification {
		return fmt.Errorf("native backend %q does not support LoRA unified batch classification", CurrentNativeBackendCapabilities().Name)
	}

	if uc.loraModelPaths == nil {
		return fmt.Errorf("loRA model paths not configured")
	}

	logging.ComponentDebugEvent("classifier", "lora_bindings_init_started", map[string]interface{}{
		"intent_model_ref":   uc.loraModelPaths.IntentPath,
		"pii_model_ref":      uc.loraModelPaths.PIIPath,
		"security_model_ref": uc.loraModelPaths.SecurityPath,
		"architecture":       uc.loraModelPaths.Architecture,
		"use_cpu":            true,
	})

	cIntentPath := C.CString(uc.loraModelPaths.IntentPath)
	defer C.free(unsafe.Pointer(cIntentPath))

	cPIIPath := C.CString(uc.loraModelPaths.PIIPath)
	defer C.free(unsafe.Pointer(cPIIPath))

	cSecurityPath := C.CString(uc.loraModelPaths.SecurityPath)
	defer C.free(unsafe.Pointer(cSecurityPath))

	cArch := C.CString(uc.loraModelPaths.Architecture)
	defer C.free(unsafe.Pointer(cArch))

	success := C.init_lora_unified_classifier(
		cIntentPath,
		cPIIPath,
		cSecurityPath,
		cArch,
		C.bool(true),
	)

	if !success {
		return fmt.Errorf("c.init_lora_unified_classifier failed")
	}

	logging.ComponentEvent("classifier", "lora_bindings_initialized", map[string]interface{}{
		"intent_model_ref":   uc.loraModelPaths.IntentPath,
		"pii_model_ref":      uc.loraModelPaths.PIIPath,
		"security_model_ref": uc.loraModelPaths.SecurityPath,
		"architecture":       uc.loraModelPaths.Architecture,
		"use_cpu":            true,
	})
	return nil
}
