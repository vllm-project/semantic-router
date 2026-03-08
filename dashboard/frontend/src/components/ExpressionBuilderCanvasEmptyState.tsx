import styles from './ExpressionBuilder.module.css'
import { BUILDER_TEMPLATES, OPERATOR_META, type BuilderTemplate } from './ExpressionBuilderNodes'

interface ExpressionBuilderCanvasEmptyStateProps {
  onApplyTemplate: (template: BuilderTemplate) => void
}

export default function ExpressionBuilderCanvasEmptyState({
  onApplyTemplate,
}: ExpressionBuilderCanvasEmptyStateProps) {
  return (
    <div className={styles.canvasEmpty}>
      <div className={styles.canvasEmptyIcon}>⊕</div>
      <div>Drag signals or operators here</div>
      <div className={styles.canvasEmptyHint}>or start with a template</div>
      <div className={styles.canvasTemplates}>
        {BUILDER_TEMPLATES.map(template => {
          const GateShape = OPERATOR_META[template.op].GateShape

          return (
            <div
              key={template.name}
              className={styles.canvasTemplateCard}
              onClick={event => {
                event.stopPropagation()
                onApplyTemplate(template)
              }}
            >
              <span className={styles.canvasTemplateGate}>
                <GateShape color="rgba(99, 102, 241, 0.6)" />
              </span>
              <span className={styles.canvasTemplateName}>{template.name}</span>
              <span className={styles.canvasTemplateDesc}>{template.desc}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
