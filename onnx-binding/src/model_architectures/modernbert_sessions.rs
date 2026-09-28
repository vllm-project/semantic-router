//! One physical session for an immutable ModernBERT classifier.
//! Logical token limits remain with the task; declared fixed dimensions govern
//! padding without changing which input tokens or entity offsets are retained.

use super::modernbert_inputs;
use crate::core::{
    instance_options::{InstanceOptions, PreparedSession, Provider},
    unified_error::{errors, UnifiedResult},
};
use ort::session::{Session, SessionOutputs};
use std::path::Path;

pub(crate) struct ClassifierSession {
    prepared: PreparedSession,
    batch: Option<usize>,
    sequence: Option<usize>,
}

fn invalid(message: &str) -> crate::core::unified_error::UnifiedError {
    errors::config_error("classifier_session", message)
}

impl ClassifierSession {
    pub(crate) fn legacy(session: Session) -> Self {
        Self {
            prepared: PreparedSession {
                session,
                cache_lease: None,
                artifacts: vec![],
            },
            batch: None,
            sequence: None,
        }
    }

    pub(crate) fn prepare(
        options: &InstanceOptions,
        graph: &Path,
        maximum: usize,
    ) -> UnifiedResult<Self> {
        let prepared = modernbert_inputs::prepare_session(options, graph, maximum)?;
        let [batch, sequence] = modernbert_inputs::fixed_dimensions(&prepared.session.inputs)?;
        if sequence.is_some_and(|length| length < maximum) {
            return Err(errors::config_error(
                "execution_max_input_tokens",
                "fixed graph capacity is below the configured execution limit",
            ));
        }
        let migraphx = options.provider == Provider::Migraphx;
        Ok(Self {
            prepared,
            batch: if migraphx { Some(1) } else { batch },
            sequence: if migraphx { Some(maximum) } else { sequence },
        })
    }

    fn covers_batch(&self, batch: usize) -> bool {
        batch > 0 && self.batch.is_none_or(|fixed| fixed == batch)
    }

    pub(crate) fn execution_length(&self, batch: usize, actual: usize) -> UnifiedResult<usize> {
        if actual == 0 || !self.covers_batch(batch) {
            return Err(invalid("input shape does not match the prepared session"));
        }
        match self.sequence {
            None => Ok(actual),
            Some(length) if actual <= length => Ok(length),
            _ => Err(invalid("complete input exceeds the prepared session shape")),
        }
    }

    pub(crate) fn run(
        &mut self,
        ids: Vec<i64>,
        mask: Vec<i64>,
        batch: usize,
        sequence: usize,
    ) -> UnifiedResult<SessionOutputs<'_>> {
        if !self.covers_batch(batch) || self.sequence.is_some_and(|fixed| fixed != sequence) {
            return Err(invalid("input shape does not match the prepared session"));
        }
        modernbert_inputs::run_cached(
            &mut self.prepared.session,
            &mut self.prepared.cache_lease,
            ids,
            mask,
            batch,
            sequence,
        )
    }

    pub(crate) fn finish_profiling(&mut self) -> UnifiedResult<Vec<String>> {
        self.prepared
            .session
            .end_profiling()
            .map(|path| vec![path])
            .map_err(|e| errors::ort_error(&e.to_string()))
    }
}
