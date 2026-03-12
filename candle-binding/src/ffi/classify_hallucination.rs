//! Hallucination detection and NLI (Natural Language Inference) FFI functions.

use crate::ffi::memory::{allocate_c_string, free_hallucination_detection_result};
use crate::ffi::types::{
    EnhancedHallucinationDetectionResult, EnhancedHallucinationSpan, HallucinationDetectionResult,
    HallucinationSpan, NLILabel, NLIResult,
};
use std::ffi::{c_char, CStr, CString};

/// Return type for collect_hallucinated_spans
type HallucinatedSpansResult = (Vec<(String, i32, i32, f32, String)>, usize, usize, f32);

/// Parse a C string to String, returning error result on failure
fn parse_c_str_or_error(
    ptr: *const c_char,
    error_msg: &str,
) -> Result<String, HallucinationDetectionResult> {
    let s = unsafe {
        match CStr::from_ptr(ptr).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                return Err(HallucinationDetectionResult {
                    error: true,
                    error_message: allocate_c_string(error_msg),
                    ..Default::default()
                });
            }
        }
    };
    Ok(s)
}

/// Extract span text from answer using character offsets
fn extract_span_text(answer: &str, start_pos: i32, end_pos: i32) -> String {
    if start_pos >= 0 && end_pos > start_pos && (end_pos as usize) <= answer.len() {
        answer[start_pos as usize..end_pos as usize].to_string()
    } else {
        String::new()
    }
}

/// Push current span to hallucinated_spans if valid
fn push_span_if_valid(
    hallucinated_spans: &mut Vec<(String, i32, i32, f32, String)>,
    answer: &str,
    start_pos: i32,
    end_pos: i32,
    confidence: f32,
) {
    let span_text = extract_span_text(answer, start_pos, end_pos);
    if !span_text.is_empty() {
        hallucinated_spans.push((
            span_text,
            start_pos,
            end_pos,
            confidence,
            "HALLUCINATED".to_string(),
        ));
    }
}

/// Collect hallucinated spans from token classification results
fn collect_hallucinated_spans(
    token_results: &[(String, usize, f32, usize, usize)],
    answer: &str,
    answer_start_pos: usize,
    effective_threshold: f32,
) -> HallucinatedSpansResult {
    let mut hallucinated_spans = Vec::new();
    let mut current_span_start: Option<i32> = None;
    let mut current_span_end: Option<i32> = None;
    let mut current_span_confidence: f32 = 0.0;
    let mut hallucination_token_count = 0;
    let mut total_answer_tokens = 0;
    let mut max_hallucination_confidence: f32 = 0.0;

    for (_token, class_idx, confidence, start, end) in token_results {
        let token_start = *start;
        if token_start < answer_start_pos {
            continue;
        }

        total_answer_tokens += 1;

        if *class_idx == 1 && *confidence >= effective_threshold {
            hallucination_token_count += 1;
            if *confidence > max_hallucination_confidence {
                max_hallucination_confidence = *confidence;
            }

            let token_offset_in_answer = (*start as i32) - answer_start_pos as i32;
            let token_end_in_answer = (*end as i32) - answer_start_pos as i32;

            if current_span_start.is_none() {
                current_span_start = Some(token_offset_in_answer);
                current_span_end = Some(token_end_in_answer);
                current_span_confidence = *confidence;
            } else {
                current_span_end = Some(token_end_in_answer);
            }

            if *confidence > current_span_confidence {
                current_span_confidence = *confidence;
            }
        } else if let (Some(start_pos), Some(end_pos)) = (current_span_start, current_span_end) {
            push_span_if_valid(
                &mut hallucinated_spans,
                answer,
                start_pos,
                end_pos,
                current_span_confidence,
            );
            current_span_start = None;
            current_span_end = None;
            current_span_confidence = 0.0;
        }
    }

    if let (Some(start_pos), Some(end_pos)) = (current_span_start, current_span_end) {
        push_span_if_valid(
            &mut hallucinated_spans,
            answer,
            start_pos,
            end_pos,
            current_span_confidence,
        );
    }

    (
        hallucinated_spans,
        hallucination_token_count,
        total_answer_tokens,
        max_hallucination_confidence,
    )
}

/// Allocate hallucination span array for FFI
/// # Safety
/// Caller must free the memory using free_hallucination_detection_result
unsafe fn allocate_hallucination_span_array(
    spans: &[(String, i32, i32, f32, String)],
) -> *mut HallucinationSpan {
    if spans.is_empty() {
        return std::ptr::null_mut();
    }

    let layout = std::alloc::Layout::array::<HallucinationSpan>(spans.len()).unwrap();
    let ptr = std::alloc::alloc(layout) as *mut HallucinationSpan;

    for (i, (text, start, end, confidence, label)) in spans.iter().enumerate() {
        let span = HallucinationSpan {
            text: allocate_c_string(text),
            start: *start,
            end: *end,
            confidence: *confidence,
            label: allocate_c_string(label),
        };
        std::ptr::write(ptr.add(i), span);
    }

    ptr
}

/// Detect hallucinations in an LLM answer given context
///
/// This is a token-level classifier that determines if each token in the answer
/// is SUPPORTED (grounded in context) or HALLUCINATED (not grounded).
///
/// # Safety
/// - `context`, `question`, `answer` must be valid null-terminated C strings
#[no_mangle]
pub unsafe extern "C" fn detect_hallucinations(
    context: *const c_char,
    question: *const c_char,
    answer: *const c_char,
    threshold: f32,
) -> HallucinationDetectionResult {
    let context = match parse_c_str_or_error(context, "Invalid context string") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let question = match parse_c_str_or_error(question, "Invalid question string") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let answer = match parse_c_str_or_error(answer, "Invalid answer string") {
        Ok(s) => s,
        Err(e) => return e,
    };

    let classifier = match crate::ffi::init::HALLUCINATION_CLASSIFIER.get() {
        Some(c) => c.clone(),
        None => {
            return HallucinationDetectionResult {
                error: true,
                error_message: unsafe {
                    allocate_c_string("Hallucination detection model not initialized")
                },
                ..Default::default()
            };
        }
    };

    let full_context = if question.is_empty() {
        context
    } else {
        format!("{} Question: {}", context, question)
    };
    let formatted_input = format!("{} [SEP] {}", full_context, answer);
    let answer_char_start = full_context.len() + " [SEP] ".len();

    match classifier.classify_tokens(&formatted_input) {
        Ok(token_results) => {
            let effective_threshold = if threshold > 0.0 && threshold <= 1.0 {
                threshold
            } else {
                0.5
            };

            let (
                hallucinated_spans,
                hallucination_token_count,
                total_answer_tokens,
                max_hallucination_confidence,
            ) = collect_hallucinated_spans(
                &token_results,
                &answer,
                answer_char_start,
                effective_threshold,
            );

            let has_hallucination = !hallucinated_spans.is_empty();
            let overall_confidence = if has_hallucination {
                max_hallucination_confidence
            } else if total_answer_tokens > 0 {
                1.0 - (hallucination_token_count as f32 / total_answer_tokens as f32)
            } else {
                1.0
            };

            let num_spans = hallucinated_spans.len() as i32;
            let spans_ptr = if num_spans > 0 {
                unsafe { allocate_hallucination_span_array(&hallucinated_spans) }
            } else {
                std::ptr::null_mut()
            };

            HallucinationDetectionResult {
                has_hallucination,
                confidence: overall_confidence,
                spans: spans_ptr,
                num_spans,
                error: false,
                error_message: std::ptr::null_mut(),
            }
        }
        Err(e) => HallucinationDetectionResult {
            error: true,
            error_message: unsafe { allocate_c_string(&format!("Classification failed: {}", e)) },
            ..Default::default()
        },
    }
}

/// Get NLI-based severity and explanation for a span
fn get_nli_severity_and_explanation(nli_result: &NLIResult) -> (i32, String) {
    match nli_result.label {
        NLILabel::Contradiction => (
            4,
            format!(
                "CONTRADICTION: This claim directly conflicts with the provided context (confidence: {:.1}%)",
                nli_result.confidence * 100.0
            ),
        ),
        NLILabel::Neutral => (
            2,
            format!(
                "FABRICATION: This claim is not supported by the provided context (confidence: {:.1}%)",
                nli_result.confidence * 100.0
            ),
        ),
        NLILabel::Entailment => (
            1,
            format!(
                "UNCERTAIN: Hallucination detector flagged this but NLI suggests it may be supported (confidence: {:.1}%)",
                nli_result.confidence * 100.0
            ),
        ),
        NLILabel::Error => (2, "Unable to determine relationship with context".to_string()),
    }
}

/// Process a single span with NLI and build EnhancedHallucinationSpan
fn process_span_with_nli(
    span: &HallucinationSpan,
    context_str: &str,
    question_str: &str,
    nli_available: bool,
) -> Option<EnhancedHallucinationSpan> {
    let span_text = if !span.text.is_null() {
        unsafe { CStr::from_ptr(span.text).to_str().unwrap_or("") }
    } else {
        ""
    };

    if span_text.is_empty() {
        return None;
    }

    let (nli_label, nli_confidence, severity, explanation) = if nli_available {
        let nli_input_premise = format!("{} {}", context_str, question_str);
        let premise_cstr = CString::new(nli_input_premise).unwrap();
        let hypothesis_cstr = CString::new(span_text).unwrap();
        let nli_result = unsafe { classify_nli(premise_cstr.as_ptr(), hypothesis_cstr.as_ptr()) };

        if nli_result.error {
            (
                NLILabel::Neutral,
                0.0,
                2,
                "NLI classification failed, based on hallucination detector only".to_string(),
            )
        } else {
            let (sev, expl) = get_nli_severity_and_explanation(&nli_result);
            (nli_result.label, nli_result.confidence, sev, expl)
        }
    } else {
        let sev = if span.confidence > 0.8 { 3 } else { 2 };
        (
            NLILabel::Neutral,
            0.0,
            sev,
            format!(
                "Unsupported claim detected (confidence: {:.1}%)",
                span.confidence * 100.0
            ),
        )
    };

    Some(EnhancedHallucinationSpan {
        text: unsafe { allocate_c_string(span_text) },
        start: span.start,
        end: span.end,
        hallucination_confidence: span.confidence,
        nli_label,
        nli_confidence,
        severity,
        explanation: unsafe { allocate_c_string(&explanation) },
    })
}

/// Classify NLI (Natural Language Inference) for a premise-hypothesis pair
///
/// # Safety
/// - `premise` and `hypothesis` must be valid null-terminated C strings
#[no_mangle]
pub unsafe extern "C" fn classify_nli(premise: *const c_char, hypothesis: *const c_char) -> NLIResult {
    let premise = unsafe {
        match CStr::from_ptr(premise).to_str() {
            Ok(s) => s,
            Err(_) => {
                return NLIResult {
                    error: true,
                    error_message: allocate_c_string("Invalid premise string"),
                    ..Default::default()
                }
            }
        }
    };

    let hypothesis = unsafe {
        match CStr::from_ptr(hypothesis).to_str() {
            Ok(s) => s,
            Err(_) => {
                return NLIResult {
                    error: true,
                    error_message: allocate_c_string("Invalid hypothesis string"),
                    ..Default::default()
                }
            }
        }
    };

    let classifier = match crate::ffi::init::NLI_CLASSIFIER.get() {
        Some(c) => c.clone(),
        None => {
            return NLIResult {
                error: true,
                error_message: unsafe { allocate_c_string("NLI model not initialized") },
                ..Default::default()
            };
        }
    };

    let nli_input = format!("{} [SEP] {}", premise, hypothesis);

    match classifier.classify_text(&nli_input) {
        Ok((class_idx, confidence)) => {
            let label = match class_idx {
                0 => NLILabel::Entailment,
                1 => NLILabel::Neutral,
                2 => NLILabel::Contradiction,
                _ => NLILabel::Error,
            };

            let (entailment_prob, neutral_prob, contradiction_prob) = match class_idx {
                0 => (
                    confidence,
                    (1.0 - confidence) / 2.0,
                    (1.0 - confidence) / 2.0,
                ),
                1 => (
                    (1.0 - confidence) / 2.0,
                    confidence,
                    (1.0 - confidence) / 2.0,
                ),
                2 => (
                    (1.0 - confidence) / 2.0,
                    (1.0 - confidence) / 2.0,
                    confidence,
                ),
                _ => (0.0, 0.0, 0.0),
            };

            NLIResult {
                label,
                confidence,
                entailment_prob,
                neutral_prob,
                contradiction_prob,
                error: false,
                error_message: std::ptr::null_mut(),
            }
        }
        Err(e) => NLIResult {
            error: true,
            error_message: unsafe {
                allocate_c_string(&format!("NLI classification failed: {}", e))
            },
            ..Default::default()
        },
    }
}

/// Detect hallucinations with NLI explanations (enhanced pipeline)
///
/// # Safety
/// - `context`, `question`, `answer` must be valid null-terminated C strings
#[no_mangle]
fn enhanced_error(msg: &str) -> EnhancedHallucinationDetectionResult {
    EnhancedHallucinationDetectionResult {
        error: true,
        error_message: unsafe { allocate_c_string(msg) },
        ..Default::default()
    }
}

fn build_enhanced_result(
    enhanced_spans: Vec<EnhancedHallucinationSpan>,
) -> EnhancedHallucinationDetectionResult {
    let num_spans = enhanced_spans.len() as i32;
    let has_hallucination = num_spans > 0;
    let overall_confidence = enhanced_spans
        .iter()
        .map(|s| s.hallucination_confidence.max(s.nli_confidence))
        .fold(0.0f32, |acc, c| acc.max(c));

    let spans_ptr = if num_spans > 0 {
        let layout =
            std::alloc::Layout::array::<EnhancedHallucinationSpan>(num_spans as usize).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) as *mut EnhancedHallucinationSpan };
        for (i, span) in enhanced_spans.into_iter().enumerate() {
            unsafe { std::ptr::write(ptr.add(i), span) };
        }
        ptr
    } else {
        std::ptr::null_mut()
    };

    EnhancedHallucinationDetectionResult {
        has_hallucination,
        confidence: overall_confidence,
        spans: spans_ptr,
        num_spans,
        error: false,
        error_message: std::ptr::null_mut(),
    }
}

/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn detect_hallucinations_with_nli(
    context: *const c_char,
    question: *const c_char,
    answer: *const c_char,
    threshold: f32,
) -> EnhancedHallucinationDetectionResult {
    let context_str = unsafe {
        match CStr::from_ptr(context).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return enhanced_error("Invalid context string"),
        }
    };
    let question_str = unsafe {
        match CStr::from_ptr(question).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return enhanced_error("Invalid question string"),
        }
    };
    let _answer_str = unsafe {
        match CStr::from_ptr(answer).to_str() {
            Ok(s) => s,
            Err(_) => return enhanced_error("Invalid answer string"),
        }
    };

    let hallucination_result = detect_hallucinations(context, question, answer, threshold);

    if hallucination_result.error {
        return EnhancedHallucinationDetectionResult {
            error: true,
            error_message: hallucination_result.error_message,
            ..Default::default()
        };
    }

    if !hallucination_result.has_hallucination || hallucination_result.num_spans == 0 {
        free_hallucination_detection_result(hallucination_result);
        return build_enhanced_result(Vec::new());
    }

    let nli_available = crate::ffi::init::NLI_CLASSIFIER.get().is_some();
    let mut enhanced_spans: Vec<EnhancedHallucinationSpan> = Vec::new();

    unsafe {
        let spans_slice = std::slice::from_raw_parts(
            hallucination_result.spans,
            hallucination_result.num_spans as usize,
        );
        for span in spans_slice {
            if let Some(enhanced) =
                process_span_with_nli(span, &context_str, &question_str, nli_available)
            {
                enhanced_spans.push(enhanced);
            }
        }
        free_hallucination_detection_result(hallucination_result);
    }

    build_enhanced_result(enhanced_spans)
}
