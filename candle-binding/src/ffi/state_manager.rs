//! Global State Manager

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, RwLock};

// Import all necessary types
use crate::classifiers::lora::parallel_engine::ParallelLoRAEngine;
use crate::classifiers::lora::token_lora::LoRATokenClassifier;
use crate::classifiers::unified::DualPathUnifiedClassifier;
use crate::core::similarity::BertSimilarity;
use crate::model_architectures::traditional::bert::TraditionalBertClassifier;

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
    // Core dual-path classifier (wrapped in Arc to avoid Clone requirement)
    unified_classifier: RwLock<Option<Arc<DualPathUnifiedClassifier>>>,

    // LoRA-specific components (wrapped in Arc)
    parallel_lora_engine: RwLock<Option<Arc<ParallelLoRAEngine>>>,
    lora_token_classifier: RwLock<Option<Arc<LoRATokenClassifier>>>,

    // Similarity engine (wrapped in Arc)
    bert_similarity: RwLock<Option<Arc<BertSimilarity>>>,

    // Legacy classifiers for backward compatibility (wrapped in Arc)
    legacy_classifiers: RwLock<HashMap<String, Arc<TraditionalBertClassifier>>>,

    // System state tracking
    system_state: RwLock<SystemState>,

    // Initialization synchronization
    initialization_lock: Mutex<()>,
}

impl GlobalStateManager {
    /// Create a new global state manager
    fn new() -> Self {
        Self {
            unified_classifier: RwLock::new(None),
            parallel_lora_engine: RwLock::new(None),
            lora_token_classifier: RwLock::new(None),
            bert_similarity: RwLock::new(None),
            legacy_classifiers: RwLock::new(HashMap::new()),
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

        // Set the classifier (wrapped in Arc)
        *self
            .unified_classifier
            .write()
            .map_err(|e| format!("Failed to set unified classifier: {}", e))? =
            Some(Arc::new(classifier));

        // Update system state to ready
        *self
            .system_state
            .write()
            .map_err(|e| format!("Failed to update system state: {}", e))? = SystemState::Ready;

        Ok(())
    }

    /// Get the unified classifier
    pub fn get_unified_classifier(&self) -> Option<Arc<DualPathUnifiedClassifier>> {
        self.unified_classifier.read().ok()?.clone()
    }

    /// Check if unified classifier is initialized
    pub fn is_unified_classifier_initialized(&self) -> bool {
        self.unified_classifier
            .read()
            .map(|c| c.is_some())
            .unwrap_or(false)
    }

    // LoRA Components Management

    /// Initialize the parallel LoRA engine
    pub fn init_parallel_lora_engine(&self, engine: ParallelLoRAEngine) -> Result<(), String> {
        *self
            .parallel_lora_engine
            .write()
            .map_err(|e| format!("Failed to set LoRA engine: {}", e))? = Some(Arc::new(engine));
        Ok(())
    }

    /// Get the parallel LoRA engine
    pub fn get_parallel_lora_engine(&self) -> Option<Arc<ParallelLoRAEngine>> {
        self.parallel_lora_engine.read().ok()?.clone()
    }

    /// Initialize the LoRA token classifier
    pub fn init_lora_token_classifier(
        &self,
        classifier: LoRATokenClassifier,
    ) -> Result<(), String> {
        *self
            .lora_token_classifier
            .write()
            .map_err(|e| format!("Failed to set LoRA token classifier: {}", e))? =
            Some(Arc::new(classifier));
        Ok(())
    }

    /// Get the LoRA token classifier
    pub fn get_lora_token_classifier(&self) -> Option<Arc<LoRATokenClassifier>> {
        self.lora_token_classifier.read().ok()?.clone()
    }

    // Similarity Engine Management

    /// Initialize the BERT similarity engine
    pub fn init_bert_similarity(&self, similarity: BertSimilarity) -> Result<(), String> {
        *self
            .bert_similarity
            .write()
            .map_err(|e| format!("Failed to set BERT similarity: {}", e))? =
            Some(Arc::new(similarity));
        Ok(())
    }

    /// Get the BERT similarity engine
    pub fn get_bert_similarity(&self) -> Option<Arc<BertSimilarity>> {
        self.bert_similarity.read().ok()?.clone()
    }

    // Legacy Classifier Management

    /// Initialize a legacy BERT classifier
    pub fn init_legacy_bert_classifier(
        &self,
        classifier: TraditionalBertClassifier,
    ) -> Result<(), String> {
        let mut classifiers = self
            .legacy_classifiers
            .write()
            .map_err(|e| format!("Failed to access legacy classifiers: {}", e))?;
        classifiers.insert("bert".to_string(), Arc::new(classifier));
        Ok(())
    }

    /// Initialize a legacy BERT PII classifier
    pub fn init_legacy_bert_pii_classifier(
        &self,
        classifier: TraditionalBertClassifier,
    ) -> Result<(), String> {
        let mut classifiers = self
            .legacy_classifiers
            .write()
            .map_err(|e| format!("Failed to access legacy classifiers: {}", e))?;
        classifiers.insert("bert_pii".to_string(), Arc::new(classifier));
        Ok(())
    }

    /// Initialize a legacy BERT jailbreak classifier
    pub fn init_legacy_bert_jailbreak_classifier(
        &self,
        classifier: TraditionalBertClassifier,
    ) -> Result<(), String> {
        let mut classifiers = self
            .legacy_classifiers
            .write()
            .map_err(|e| format!("Failed to access legacy classifiers: {}", e))?;
        classifiers.insert("bert_jailbreak".to_string(), Arc::new(classifier));
        Ok(())
    }

    /// Get a legacy classifier by name
    pub fn get_legacy_classifier(&self, name: &str) -> Option<Arc<TraditionalBertClassifier>> {
        let classifiers = self.legacy_classifiers.read().ok()?;
        classifiers.get(name).cloned()
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
            || self
                .parallel_lora_engine
                .read()
                .map(|e| e.is_some())
                .unwrap_or(false)
            || self
                .bert_similarity
                .read()
                .map(|s| s.is_some())
                .unwrap_or(false)
            || !self
                .legacy_classifiers
                .read()
                .map(|c| c.is_empty())
                .unwrap_or(true)
    }

    /// Cleanup all resources
    pub fn cleanup(&self) {
        let _lock = self.initialization_lock.lock();

        // Update system state
        if let Ok(mut state) = self.system_state.write() {
            *state = SystemState::ShuttingDown;
        }

        // Clear all components
        if let Ok(mut classifier) = self.unified_classifier.write() {
            *classifier = None;
        }

        if let Ok(mut engine) = self.parallel_lora_engine.write() {
            *engine = None;
        }

        if let Ok(mut classifier) = self.lora_token_classifier.write() {
            *classifier = None;
        }

        if let Ok(mut similarity) = self.bert_similarity.write() {
            *similarity = None;
        }

        if let Ok(mut classifiers) = self.legacy_classifiers.write() {
            classifiers.clear();
        }

        // Update system state
        if let Ok(mut state) = self.system_state.write() {
            *state = SystemState::Uninitialized;
        }
    }

    /// Get system statistics
    pub fn get_stats(&self) -> GlobalStateStats {
        GlobalStateStats {
            unified_classifier_initialized: self.is_unified_classifier_initialized(),
            parallel_lora_engine_initialized: self
                .parallel_lora_engine
                .read()
                .map(|e| e.is_some())
                .unwrap_or(false),
            lora_token_classifier_initialized: self
                .lora_token_classifier
                .read()
                .map(|c| c.is_some())
                .unwrap_or(false),
            bert_similarity_initialized: self
                .bert_similarity
                .read()
                .map(|s| s.is_some())
                .unwrap_or(false),
            legacy_classifiers_count: self.legacy_classifiers.read().map(|c| c.len()).unwrap_or(0),
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
