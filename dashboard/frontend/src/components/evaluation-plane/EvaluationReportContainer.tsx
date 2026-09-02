import useEvaluationReportDiagnostics from '../../hooks/useEvaluationReportDiagnostics'
import type { EvaluationReport } from '../../types/evaluationReport'
import EvaluationReportView from './EvaluationReportView'

export default function EvaluationReportContainer({ report }: { report: EvaluationReport }) {
  const diagnostics = useEvaluationReportDiagnostics(report)
  return <EvaluationReportView report={report} diagnostics={diagnostics} />
}
