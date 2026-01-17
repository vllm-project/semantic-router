import React from 'react'
import Layout from '@theme/Layout'
import Translate from '@docusaurus/Translate'
import Link from '@docusaurus/Link'
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
  const groupId = group.name.replace(/\s+/g, '').toLowerCase()

  return (
    <div className={styles.workGroupCard}>
      <div className={styles.cardHeader}>
        <span className={styles.icon}>{group.icon}</span>
        <h3 className={styles.groupName}>
          <Translate id={`workGroups.group.${groupId}.name`}>{group.name}</Translate>
        </h3>
        <span className={styles.label}>{group.label}</span>
      </div>
      <p className={styles.description}>
        <Translate id={`workGroups.group.${groupId}.description`}>{group.description}</Translate>
      </p>

      <div className={styles.skillsSection}>
        <h4><Translate id="workGroups.card.skillsNeeded">Skills Needed:</Translate></h4>
        <ul className={styles.skillsList}>
          {group.skills && group.skills.map((skill, index) => (
            <li key={index}>
              <Translate id={`workGroups.group.${groupId}.skills.${index}`}>{skill}</Translate>
            </li>
          ))}
        </ul>
      </div>

      <div className={styles.needsSection}>
        <h4><Translate id="workGroups.card.currentNeeds">Current Needs:</Translate></h4>
        <ul className={styles.needsList}>
          {group.needs && group.needs.map((need, index) => (
            <li key={index}>
              <Translate id={`workGroups.group.${groupId}.needs.${index}`}>{need}</Translate>
            </li>
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
              <Translate id="workGroups.init.description">
                We are looking for interests around vLLM Semantic Router project and separate it into different WGs.
              </Translate>
            </p>
            <p>
              <Translate id="workGroups.init.comment.prefix">Please comment on</Translate>
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
              <Translate id="workGroups.init.comment.suffix">if you are interested in one or more.</Translate>
            </p>
          </section>

          <section className={styles.workingGroupsSection}>
            <h2>
              ⛰️
              <Translate id="workGroups.community.title">vLLM Semantic Router Community WG</Translate>
            </h2>
            <p>
              <Translate id="workGroups.community.description">
                This section is about setting WG around this project, to gather focus on specify areas.
              </Translate>
            </p>

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
              <Translate id="workGroups.promotion.description">
                We are grateful for any contributions, and if you show consistent contributions to the above specify area,
                you will be promoting as its maintainer after votes from maintainer team, and you will be invited to
                semantic-router-maintainer group, and granted WRITE access to this repo.
              </Translate>
            </p>
          </section>

          <section className={styles.getInvolved}>
            <h2><Translate id="workGroups.getInvolved.title">How to Get Involved</Translate></h2>
            <ol className={styles.stepsList}>
              <li>
                <strong><Translate id="workGroups.step1.title">Choose Your Interest Area:</Translate></strong>
                {' '}
                <Translate id="workGroups.step1.desc">Review the working groups above and identify which areas align with your skills and interests</Translate>
              </li>
              <li>
                <strong><Translate id="workGroups.step2.title">Join the Discussion:</Translate></strong>
                {' '}
                <Translate id="workGroups.step2.desc.prefix">Comment on</Translate>
                {' '}
                <a href="https://github.com/vllm-project/semantic-router/issues/15" target="_blank" rel="noopener noreferrer">GitHub Issue #15</a>
                {' '}
                <Translate id="workGroups.step2.desc.suffix">to express your interest</Translate>
              </li>
              <li>
                <strong><Translate id="workGroups.step3.title">Start Contributing:</Translate></strong>
                {' '}
                <Translate id="workGroups.step3.desc">Look for issues labeled with the corresponding area tags (e.g.,</Translate>
                {' '}
                <code>area/document</code>
                ,
                {' '}
                <code>area/core</code>
                )
              </li>
              <li>
                <strong><Translate id="workGroups.step4.title">Collaborate:</Translate></strong>
                {' '}
                <Translate id="workGroups.step4.desc">Connect with other community members working in the same areas</Translate>
              </li>
            </ol>
          </section>

          <section className={styles.contact}>
            <h2><Translate id="workGroups.contact.title">Contact</Translate></h2>
            <p><Translate id="workGroups.contact.desc">For questions about working groups or to get involved:</Translate></p>
            <ul>
              <li>
                <Translate id="workGroups.contact.issue">Open an issue on</Translate>
                <a href="https://github.com/vllm-project/semantic-router/issues" target="_blank" rel="noopener noreferrer"> GitHub</a>
              </li>
              <li>
                <Translate id="workGroups.contact.discussion">Join the discussion on</Translate>
                <a href="https://github.com/vllm-project/semantic-router/issues/15" target="_blank" rel="noopener noreferrer"> Issue #15</a>
              </li>
              <li>
                <Translate id="workGroups.contact.docs">Check out our</Translate>
                <Link to="/docs/intro"> documentation</Link>
                {' '}
                <Translate id="workGroups.contact.start">to get started</Translate>
              </li>
            </ul>
          </section>
        </main>
      </div>
    </Layout>
  )
}

export default WorkGroups
