import React from 'react'
import Layout from '@theme/Layout'
import Translate from '@docusaurus/Translate'
import styles from './work-groups.module.css'

interface WorkGroup {
  name: string
  description: string
  label: string
  icon: string
  skills: string[]
  needs: string[]
}

const workingGroups: WorkGroup[] = [
  // First column - Core areas
  {
    name: 'RouterCore',
    description: 'Using embedded SLM, implement advanced routing algorithm like classify, security detection, auto reasoning etc.',
    label: 'area/core',
    icon: '🧠',
    skills: ['Machine learning', 'BERT models', 'Classification algorithms'],
    needs: ['Model optimization', 'Algorithm improvements', 'Reasoning logic'],
  },
  {
    name: 'Research',
    description: 'Explore the frontier of SLM (Small Language Model) in vLLM Semantic Router, improving latency and context of SLM.',
    label: 'area/research',
    icon: '🔬',
    skills: ['Model Training', 'Model Fine-Tuning', 'Deep Learning'],
    needs: ['SLM research', 'Latency optimization', 'Context improvement'],
  },
  {
    name: 'Networking',
    description: 'Envoy ExtProc, Traffic Management, Networks Optimization',
    label: 'area/networking',
    icon: '🌐',
    skills: ['Envoy proxy', 'Network protocols', 'Performance optimization'],
    needs: ['Load balancing', 'Traffic routing', 'Network security'],
  },
  // Second column - Operations and monitoring
  {
    name: 'Observability',
    description: 'Metrics collection, distributed tracing, monitoring dashboards, and structured logging for production visibility',
    label: 'area/observability',
    icon: '📈',
    skills: ['Prometheus/Grafana', 'OpenTelemetry', 'Log aggregation', 'Monitoring systems'],
    needs: ['Metrics implementation', 'Tracing integration', 'Dashboard creation', 'Log standardization'],
  },
  {
    name: 'Bench',
    description: 'Reasoning Benchmark Framework, Performance Optimization',
    label: 'area/benchmark',
    icon: '📊',
    skills: ['Performance testing', 'Benchmarking tools', 'Data analysis'],
    needs: ['Benchmark frameworks', 'Performance metrics', 'Testing automation'],
  },
  {
    name: 'Environment',
    description: 'Docker Compose, Kubernetes, Local support, Cloud Foundry Integration',
    label: 'area/environment',
    icon: '🐳',
    skills: ['Docker', 'Kubernetes', 'Cloud platforms', 'DevOps'],
    needs: ['Helm charts', 'Deployment automation', 'Cloud integrations'],
  },
  // Third column - Development and user experience
  {
    name: 'Test and Release',
    description: 'CI/CD, Build, test, release',
    label: 'area/tooling, area/ci',
    icon: '🔧',
    skills: ['CI/CD tools', 'Build automation', 'Release processes'],
    needs: ['Test automation', 'Release pipelines', 'Quality assurance'],
  },
  {
    name: 'User Experience',
    description: 'User experience across vLLM Semantic Router, API, Configuration and CLI guidelines and support',
    label: 'area/user-experience',
    icon: '👥',
    skills: ['API design', 'UX/UI', 'Developer experience'],
    needs: ['API improvements', 'CLI enhancements', 'User feedback integration'],
  },
  {
    name: 'Docs',
    description: 'User docs, information architecture, infrastructure',
    label: 'area/document',
    icon: '📚',
    skills: ['Technical writing', 'Documentation tools', 'User experience design'],
    needs: ['API documentation', 'Tutorials', 'Deployment guides'],
  },
]

interface WorkGroupCardProps {
  group: WorkGroup
}

const WorkGroupCard: React.FC<WorkGroupCardProps> = ({ group }) => {
  return (
    <div className={styles.workGroupCard}>
      <div className={styles.cardHeader}>
        <span className={styles.icon}>{group.icon}</span>
        <h3 className={styles.groupName}>{group.name}</h3>
        <span className={styles.label}>{group.label}</span>
      </div>
      <p className={styles.description}>{group.description}</p>

      <div className={styles.skillsSection}>
        <h4>Skills Needed:</h4>
        <ul className={styles.skillsList}>
          {group.skills && group.skills.map((skill, index) => (
            <li key={index}>{skill}</li>
          ))}
        </ul>
      </div>

      <div className={styles.needsSection}>
        <h4>Current Needs:</h4>
        <ul className={styles.needsList}>
          {group.needs && group.needs.map((need, index) => (
            <li key={index}>{need}</li>
          ))}
        </ul>
      </div>
    </div>
  )
}

const WorkGroups: React.FC = () => {
  return (
    <Layout
      title="Work Groups"
      description="vLLM Semantic Router Community Working Groups"
    >
      <div className={styles.container}>
        <header className={styles.header}>
          <h1><Translate id="workGroups.title">vLLM Semantic Router Work Groups 👋</Translate></h1>
        </header>

        <main className={styles.main}>
          <section className={styles.intro}>
            <h2>
              🌍
              <Translate id="workGroups.init.title">WG Initialization</Translate>
            </h2>
            <p>
              We are looking for interests around vLLM Semantic Router project and separate it into different WGs.
            </p>
            <p>
              Please comment on
              {' '}
              <a
                href="https://github.com/vllm-project/semantic-router/issues/15"
                target="_blank"
                rel="noopener noreferrer"
                className={styles.link}
              >
                GitHub Issue #15
              </a>
              {' '}
              if you are interested in one or more.
            </p>
          </section>

          <section className={styles.workingGroupsSection}>
            <h2>
              ⛰️
              <Translate id="workGroups.community.title">vLLM Semantic Router Community WG</Translate>
            </h2>
            <p>This section is about setting WG around this project, to gather focus on specify areas.</p>

            <div className={styles.workGroupsGrid}>
              {workingGroups.map((group, index) => (
                <WorkGroupCard key={index} group={group} />
              ))}
            </div>
          </section>

          <section className={styles.promotion}>
            <h2>
              🔝
              <Translate id="workGroups.promotion.title">Community Promotion</Translate>
            </h2>
            <p>
              We are grateful for any contributions, and if you show consistent contributions to the above specify area,
              you will be promoting as its maintainer after votes from maintainer team, and you will be invited to
              semantic-router-maintainer group, and granted WRITE access to this repo.
            </p>
          </section>

          <section className={styles.getInvolved}>
            <h2><Translate id="workGroups.getInvolved.title">How to Get Involved</Translate></h2>
            <ol className={styles.stepsList}>
              <li>
                <strong>Choose Your Interest Area:</strong>
                {' '}
                Review the working groups above and identify which areas align with your skills and interests
              </li>
              <li>
                <strong>Join the Discussion:</strong>
                {' '}
                Comment on
                {' '}
                <a href="https://github.com/vllm-project/semantic-router/issues/15" target="_blank" rel="noopener noreferrer">GitHub Issue #15</a>
                {' '}
                to express your interest
              </li>
              <li>
                <strong>Start Contributing:</strong>
                {' '}
                Look for issues labeled with the corresponding area tags (e.g.,
                {' '}
                <code>area/document</code>
                ,
                {' '}
                <code>area/core</code>
                )
              </li>
              <li>
                <strong>Collaborate:</strong>
                {' '}
                Connect with other community members working in the same areas
              </li>
            </ol>
          </section>

          <section className={styles.contact}>
            <h2><Translate id="workGroups.contact.title">Contact</Translate></h2>
            <p>For questions about working groups or to get involved:</p>
            <ul>
              <li>
                Open an issue on
                <a href="https://github.com/vllm-project/semantic-router/issues" target="_blank" rel="noopener noreferrer"> GitHub</a>
              </li>
              <li>
                Join the discussion on
                <a href="https://github.com/vllm-project/semantic-router/issues/15" target="_blank" rel="noopener noreferrer"> Issue #15</a>
              </li>
              <li>
                Check out our
                <a href="/docs/intro"> documentation</a>
                {' '}
                to get started
              </li>
            </ul>
          </section>
        </main>
      </div>
    </Layout>
  )
}

export default WorkGroups
