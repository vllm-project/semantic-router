use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// Central model registry to replace role-specific OnceLock sprawl
pub struct ModelRegistry {
    models: RwLock<HashMap<String, Arc<dyn std::any::Any + Send + Sync>>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
        }
    }

    pub fn register<T: std::any::Any + Send + Sync>(&self, id: &str, model: T) -> Result<(), String> {
        let mut map = self.models.write().map_err(|e| e.to_string())?;
        map.insert(id.to_string(), Arc::new(model));
        Ok(())
    }

    pub fn get<T: std::any::Any + Send + Sync>(&self, id: &str) -> Option<Arc<T>> {
        let map = self.models.read().ok()?;
        let any_model = map.get(id)?;
        any_model.clone().downcast::<T>().ok()
    }

    pub fn unregister(&self, id: &str) -> Result<(), String> {
        let mut map = self.models.write().map_err(|e| e.to_string())?;
        map.remove(id);
        Ok(())
    }
}

static REGISTRY_ONCE: std::sync::OnceLock<ModelRegistry> = std::sync::OnceLock::new();

pub fn get_registry() -> &'static ModelRegistry {
    REGISTRY_ONCE.get_or_init(ModelRegistry::new)
}
