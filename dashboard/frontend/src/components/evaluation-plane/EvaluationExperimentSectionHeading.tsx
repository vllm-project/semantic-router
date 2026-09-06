import styles from './EvaluationExperimentSection.module.css'

interface EvaluationExperimentSectionHeadingProps {
  index: string
  title: string
  description: string
}

export default function EvaluationExperimentSectionHeading({
  index,
  title,
  description,
}: EvaluationExperimentSectionHeadingProps) {
  return (
    <div className={styles.sectionHeading}>
      <span>{index}</span>
      <div>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  )
}
