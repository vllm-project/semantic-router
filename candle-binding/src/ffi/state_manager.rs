//! Global State Manager

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, RwLock};

// Import all necessary types
use crate::classifiers::lora::parallel_engine::ParallelLoRAEngine;
use crate::classifiers::lora::token_lora::LoRATokenClassifier;
use crate::classifiers::unified::DualPathUnifiedClassifier;
use crate::core::similarity::BertSimilarity;
use crate::model_architectures::traditional::bert::TraditionalBertClassifier;

use crate::registry::REGISTRY;

/// System state for the global state manager
#[derive(Debug, Clone, PartialEq)]
pub enum SystemState {
    /// System is not initialized
    Uninitialized,
    /// System is being initialized
    Initializing,
    /// System is ready for operation
    Ready,
    /// System is shutting down
    ShuttingDown,
    /// System encountered an error
    Error(String),
}

/// Global state manager for unified FFI state management
pub struct GlobalStateManager {
    // System state tracking
    system_state: RwLock<SystemState>,

    // Initialization synchronization
    initialization_lock: Mutex<()>,
}

impl GlobalStateManager {
    /// Create a new global state manager
    fn new() -> Self {
        Self {
            system_state: RwLock::new(SystemState::Uninitialized),
            initialization_lock: Mutex::new(()),
        }
    }

    /// Get the global instance (singleton pattern)
    pub fn instance() -> &'static GlobalStateManager {
        &GLOBAL_STATE_MANAGER
    }

    // Unified Classifier Management

    /// Initialize the unified classifier
    pub fn init_unified_classifier(
        &self,
        classifier: DualPathUnifiedClassifier,
    ) -> Result<(), String> {
        let _lock = self
            .initialization_lock
            .lock()
            .map_err(|e| format!("Failed to acquire initialization lock: {}", e))?;

        // Update system state
        *self
            .system_state
            .write()
            .map_err(|e| format!("Failed to update system state: {}", e))? =
            SystemState::Initializing;

        REGISTRY.register("unified_classifier", classifier)?;

        // Update system state to ready
        *self
            .system_state
            .write()
            .map_err(|e| format!("Failed to update system state: {}", e))? = SystemState::Ready;

        Ok(())
    }

    /// Get the unified classifier
    pub fn get_unified_classifier(&self) -> Option<Arc<DualPathUnifiedClassifier>> {
        REGISTRY.get::<DualPathUnifiedClassifier>("unified_classifier")
    }

    /// Check if unified classifier is initialized
    pub fn is_unified_classifier_initialized(&self) -> bool {
        self.get_unified_classifier().is_some()
    }

    // LoRA Components Management

    /// Initialize the parallel LoRA engine
    pub fn init_parallel_lora_engine(&self, engine: ParallelLoRAEngine) -> Result<(), String> {
        REGISTRY.register("parallel_lora_engine", engine)
    }

    /// Get the parallel LoRA engine
    pub fn get_parallel_lora_engine(&self) -> Option<Arc<ParallelLoRAEngine>> {
        REGISTRY.get::<ParallelLoRAEngine>("parallel_lora_engine")
    }

    /// Initialize the LoRA token classifier
    pub fn init_lora_token_classifier(
        &self,
        classifier: LoRATokenClassifier,
    ) -> Result<(), String> {
        REGISTRY.register("lora_token_classifier", classifier)
    }

    /// Get the LoRA token classifier
    pub fn get_lora_token_classifier(&self) -> Option<Arc<LoRATokenClassifier>> {
        REGISTRY.get::<LoRATokenClassifier>("lora_token_classifier")
    }

    // Similarity Engine Management

    /// Initialize the BERT similarity engine
    pub fn init_bert_similarity(&self, similarity: BertSimilarity) -> Result<(), String> {
        REGISTRY.register("bert_similarity", similarity)
    }

    /// Get the BERT similarity engine
    pub fn get_bert_similarity(&self) -> Option<Arc<BertSimilarity>> {
        REGISTRY.get::<BertSimilarity>("bert_similarity")
    }

    // Legacy Classifier Management

    /// Initialize a legacy BERT classifier
    pub fn init_legacy_bert_classifier(
        &self,
        classifier: TraditionalBertClassifier,
    ) -> Result<(), String> {
        REGISTRY.register("legacy_bert", classifier)
    }

    /// Initialize a legacy BERT PII classifier
    pub fn init_legacy_bert_pii_classifier(
        &self,
        classifier: TraditionalBertClassifier,
    ) -> Result<(), String> {
        REGISTRY.register("legacy_bert_pii", classifier)
    }

    /// Initialize a legacy BERT jailbreak classifier
    pub fn init_legacy_bert_jailbreak_classifier(
        &self,
        classifier: TraditionalBertClassifier,
    ) -> Result<(), String> {
        REGISTRY.register("legacy_bert_jailbreak", classifier)
    }

    /// Get a legacy classifier by name
    pub fn get_legacy_classifier(&self, name: &str) -> Option<Arc<TraditionalBertClassifier>> {
        let key = format!("legacy_{}", name);
        REGISTRY.get::<TraditionalBertClassifier>(&key)
    }

    // System State Management

    /// Get the current system state
    pub fn get_system_state(&self) -> SystemState {
        self.system_state
            .read()
            .map(|s| s.clone())
            .unwrap_or(SystemState::Error(
                "Failed to read system state".to_string(),
            ))
    }

    /// Check if the system is ready for operation
    pub fn is_ready(&self) -> bool {
        matches!(self.get_system_state(), SystemState::Ready)
    }

    /// Check if the system is initialized (any component)
    pub fn is_any_initialized(&self) -> bool {
        self.is_unified_classifier_initialized()
            || self.get_parallel_lora_engine().is_some()
            || self.get_bert_similarity().is_some()
            || self.get_legacy_classifier("bert").is_some()
    }

    /// Cleanup all resources
    pub fn cleanup(&self) {
        let _lock = self.initialization_lock.lock();

        // Update system state
        if let Ok(mut state) = self.system_state.write() {
            *state = SystemState::ShuttingDown;
        }

        // Clear all components
        let _ = REGISTRY.unregister("unified_classifier");
        let _ = REGISTRY.unregister("parallel_lora_engine");
        let _ = REGISTRY.unregister("lora_token_classifier");
        let _ = REGISTRY.unregister("bert_similarity");
        let _ = REGISTRY.unregister("legacy_bert");
        let _ = REGISTRY.unregister("legacy_bert_pii");
        let _ = REGISTRY.unregister("legacy_bert_jailbreak");

        // Update system state
        if let Ok(mut state) = self.system_state.write() {
            *state = SystemState::Uninitialized;
        }
    }

    /// Get system statistics
    pub fn get_stats(&self) -> GlobalStateStats {
        GlobalStateStats {
            unified_classifier_initialized: self.is_unified_classifier_initialized(),
            parallel_lora_engine_initialized: self.get_parallel_lora_engine().is_some(),
            lora_token_classifier_initialized: self.get_lora_token_classifier().is_some(),
            bert_similarity_initialized: self.get_bert_similarity().is_some(),
            legacy_classifiers_count: 0, // Simplified for now
            system_state: self.get_system_state(),
        }
    }
}

/// Statistics about the global state
#[derive(Debug, Clone)]
pub struct GlobalStateStats {
    pub unified_classifier_initialized: bool,
    pub parallel_lora_engine_initialized: bool,
    pub lora_token_classifier_initialized: bool,
    pub bert_similarity_initialized: bool,
    pub legacy_classifiers_count: usize,
    pub system_state: SystemState,
}

// Global singleton instance using LazyLock
static GLOBAL_STATE_MANAGER: LazyLock<GlobalStateManager> = LazyLock::new(GlobalStateManager::new);

/// Convenience functions for backward compatibility

/// Get the global state manager instance
pub fn get_global_state_manager() -> &'static GlobalStateManager {
    GlobalStateManager::instance()
}

/// Check if any component is initialized
pub fn is_any_component_initialized() -> bool {
    GlobalStateManager::instance().is_any_initialized()
}

/// Get system statistics
pub fn get_system_stats() -> GlobalStateStats {
    GlobalStateManager::instance().get_stats()
}

/// Cleanup all global state
pub fn cleanup_global_state() {
    GlobalStateManager::instance().cleanup();
}
