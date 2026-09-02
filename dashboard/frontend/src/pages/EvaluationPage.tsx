import EvaluationPageContent from './EvaluationPageContent'
import { useEvaluationPageController } from './useEvaluationPageController'

export function EvaluationPage() {
  const controller = useEvaluationPageController()
  return <EvaluationPageContent controller={controller} />
}

export default EvaluationPage
