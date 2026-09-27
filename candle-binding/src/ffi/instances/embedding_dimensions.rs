//! Output views declared by the concrete, successfully loaded embedding adapter.
use super::Model;
use crate::model_architectures::traits::LongContextEmbeddingCapable;

pub(super) fn for_model(model: &Model) -> Vec<usize> {
    let Model::Embedding(factory) = model else {
        return Vec::new();
    };
    if let Some(model) = factory.get_mmbert_model() {
        return bounded(
            model.get_embedding_dimension(),
            model.get_matryoshka_dimensions(),
        );
    }
    if let Some(model) = factory.get_qwen3_model() {
        return bounded(
            model.get_embedding_dimension(),
            model.get_matryoshka_dimensions(),
        );
    }
    if let Some(model) = factory.get_multimodal_model() {
        return bounded(
            model.get_embedding_dimension(),
            model.get_matryoshka_dimensions(),
        );
    }
    if let Some(model) = factory.get_gemma_model() {
        return model.available_dimensions();
    }
    Vec::new()
}

fn bounded(full: usize, mut declared: Vec<usize>) -> Vec<usize> {
    declared.retain(|dimension| *dimension > 0 && *dimension <= full);
    declared.push(full);
    declared.sort_unstable();
    declared.dedup();
    declared
}
