//! Global State Manager

use std::sync::{Arc, Mutex, RwLock, OnceLock};

// Import all necessary types
use crate::classifiers::lora::parallel_engine::ParallelLoRAEngine;
use crate::classifiers::lora::token_lora::LoRATokenClassifier;
use crate::classifiers::unified::DualPathUnifiedClassifier;
use crate::core::similarity::BertSimilarity;
use crate::model_architectures::traditional::bert::TraditionalBertClassifier;

use crate::registry::get_registry;

/// System state for the global state manager
#[derive(Debug, Clone, PartialEq)]
pub enum SystemState {
    Uninitialized,
    Initializing,
    Ready,
    ShuttingDown,
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
    fn new() -> Self {
        Self {
            system_state: RwLock::new(SystemState::Uninitialized),
            initialization_lock: Mutex::new(()),
        }
    }

    pub fn instance() -> &'static GlobalStateManager {
        GLOBAL_STATE_MANAGER_ONCE.get_or_init(GlobalStateManager::new)
    }

    // Unified Classifier Management
    pub fn init_unified_classifier(
        &self,
        classifier: DualPathUnifiedClassifier,
    ) -> Result<(), String> {
        let _lock = self.initialization_lock.lock().map_err(|e| e.to_string())?;
        *self.system_state.write().unwrap() = SystemState::Initializing;
        get_registry().register("unified_classifier", classifier)?;
        *self.system_state.write().unwrap() = SystemState::Ready;
        Ok(())
    }

    pub fn get_unified_classifier(&self) -> Option<Arc<DualPathUnifiedClassifier>> {
        get_registry().get::<DualPathUnifiedClassifier>("unified_classifier")
    }

    pub fn is_unified_classifier_initialized(&self) -> bool {
        self.get_unified_classifier().is_some()
    }

    // LoRA Components Management
    pub fn init_parallel_lora_engine(&self, engine: ParallelLoRAEngine) -> Result<(), String> {
        get_registry().register("parallel_lora_engine", engine)
    }

    pub fn get_parallel_lora_engine(&self) -> Option<Arc<ParallelLoRAEngine>> {
        get_registry().get::<ParallelLoRAEngine>("parallel_lora_engine")
    }

    pub fn init_lora_token_classifier(
        &self,
        classifier: LoRATokenClassifier,
    ) -> Result<(), String> {
        get_registry().register("lora_token_classifier", classifier)
    }

    pub fn get_lora_token_classifier(&self) -> Option<Arc<LoRATokenClassifier>> {
        get_registry().get::<LoRATokenClassifier>("lora_token_classifier")
    }

    // Similarity Engine Management
    pub fn init_bert_similarity(&self, similarity: BertSimilarity) -> Result<(), String> {
        get_registry().register("bert_similarity", similarity)
    }

    pub fn get_bert_similarity(&self) -> Option<Arc<BertSimilarity>> {
        get_registry().get::<BertSimilarity>("bert_similarity")
    }

    // Legacy Classifier Management
    pub fn init_legacy_bert_classifier(
        &self,
        classifier: TraditionalBertClassifier,
    ) -> Result<(), String> {
        get_registry().register("legacy_bert", classifier)
    }

    pub fn init_legacy_bert_pii_classifier(
        &self,
        classifier: TraditionalBertClassifier,
    ) -> Result<(), String> {
        get_registry().register("legacy_bert_pii", classifier)
    }

    pub fn init_legacy_bert_jailbreak_classifier(
        &self,
        classifier: TraditionalBertClassifier,
    ) -> Result<(), String> {
        get_registry().register("legacy_bert_jailbreak", classifier)
    }

    pub fn get_legacy_classifier(&self, name: &str) -> Option<Arc<TraditionalBertClassifier>> {
        let key = format!("legacy_{}", name);
        get_registry().get::<TraditionalBertClassifier>(&key)
    }

    // System State Management
    pub fn get_system_state(&self) -> SystemState {
        self.system_state.read().unwrap().clone()
    }

    pub fn is_ready(&self) -> bool {
        matches!(self.get_system_state(), SystemState::Ready)
    }

    pub fn is_any_initialized(&self) -> bool {
        self.is_unified_classifier_initialized()
            || self.get_parallel_lora_engine().is_some()
            || self.get_bert_similarity().is_some()
            || self.get_legacy_classifier("bert").is_some()
    }

    pub fn cleanup(&self) {
        let _lock = self.initialization_lock.lock();
        if let Ok(mut state) = self.system_state.write() {
            *state = SystemState::ShuttingDown;
        }

        let _ = get_registry().unregister("unified_classifier");
        let _ = get_registry().unregister("parallel_lora_engine");
        let _ = get_registry().unregister("lora_token_classifier");
        let _ = get_registry().unregister("bert_similarity");
        let _ = get_registry().unregister("legacy_bert");
        let _ = get_registry().unregister("legacy_bert_pii");
        let _ = get_registry().unregister("legacy_bert_jailbreak");

        if let Ok(mut state) = self.system_state.write() {
            *state = SystemState::Uninitialized;
        }
    }

    pub fn get_stats(&self) -> GlobalStateStats {
        GlobalStateStats {
            unified_classifier_initialized: self.is_unified_classifier_initialized(),
            parallel_lora_engine_initialized: self.get_parallel_lora_engine().is_some(),
            lora_token_classifier_initialized: self.get_lora_token_classifier().is_some(),
            bert_similarity_initialized: self.get_bert_similarity().is_some(),
            legacy_classifiers_count: 0,
            system_state: self.get_system_state(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GlobalStateStats {
    pub unified_classifier_initialized: bool,
    pub parallel_lora_engine_initialized: bool,
    pub lora_token_classifier_initialized: bool,
    pub bert_similarity_initialized: bool,
    pub legacy_classifiers_count: usize,
    pub system_state: SystemState,
}

static GLOBAL_STATE_MANAGER_ONCE: OnceLock<GlobalStateManager> = OnceLock::new();

pub fn get_global_state_manager() -> &'static GlobalStateManager {
    GlobalStateManager::instance()
}

pub fn is_any_component_initialized() -> bool {
    GlobalStateManager::instance().is_any_initialized()
}

pub fn get_system_stats() -> GlobalStateStats {
    GlobalStateManager::instance().get_stats()
}

pub fn cleanup_global_state() {
    GlobalStateManager::instance().cleanup();
}
