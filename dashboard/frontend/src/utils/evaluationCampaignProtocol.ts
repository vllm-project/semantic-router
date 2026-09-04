import { EVALUATION_CAMPAIGN_COHORT_SCHEMA_VERSION } from '../types/evaluationPlane'
import {
  hasOnlyEvaluationFields,
  isEvaluationRecord,
  isNonNegativeInteger,
  type EvaluationRecord,
} from './evaluationContractValidation'

function hasOwn(value: EvaluationRecord, field: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, field)
}

export function hasValidCampaignProtocol(suite: EvaluationRecord): boolean {
  if (!hasOwn(suite, 'campaign_protocol')) return true
  const protocol = suite.campaign_protocol
  return (
    isEvaluationRecord(protocol) &&
    hasOnlyEvaluationFields(protocol, ['schema_version', 'minimum_cases']) &&
    protocol.schema_version === EVALUATION_CAMPAIGN_COHORT_SCHEMA_VERSION &&
    isNonNegativeInteger(protocol.minimum_cases) &&
    protocol.minimum_cases > 0 &&
    isNonNegativeInteger(suite.case_count) &&
    suite.case_count > 0 &&
    protocol.minimum_cases <= suite.case_count
  )
}
