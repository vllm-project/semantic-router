# Supported Categories

vLLM Semantic Router supports 14 predefined categories for automatic query classification. Each category represents a distinct domain that can be configured with custom routing rules, reasoning settings, and model preferences based on your specific needs.

## Category Overview

| Category | Domain | Typical Use Cases |
|----------|--------|-------------------|
| [Math](#math) | Mathematics | Calculations, equations, proofs, statistics |
| [Computer Science](#computer-science) | Programming & Technology | Coding, algorithms, software engineering |
| [Physics](#physics) | Physical Sciences | Mechanics, thermodynamics, quantum physics |
| [Chemistry](#chemistry) | Chemical Sciences | Reactions, molecular structures, formulas |
| [Biology](#biology) | Life Sciences | Genetics, ecology, anatomy, physiology |
| [Engineering](#engineering) | Applied Sciences | Design, systems analysis, problem-solving |
| [Business](#business) | Commerce & Management | Strategy, marketing, finance, operations |
| [Law](#law) | Legal Domain | Regulations, jurisprudence, legal procedures |
| [Economics](#economics) | Economic Sciences | Markets, theory, macroeconomics, finance |
| [Health](#health) | Medical & Wellness | Healthcare, anatomy, medical information |
| [Psychology](#psychology) | Behavioral Sciences | Mental health, cognition, therapy |
| [Philosophy](#philosophy) | Philosophical Inquiry | Ethics, logic, metaphysics, reasoning |
| [History](#history) | Historical Studies | Events, civilizations, historical analysis |
| [Other](#other) | General Purpose | Miscellaneous queries, general knowledge |

## Detailed Category Descriptions

### Math

**Domain**: Mathematics, calculations, equations, proofs, statistical analysis

**Description**: Handles mathematical queries ranging from basic arithmetic to advanced calculus, statistics, and mathematical proofs. This category is ideal for computational problems that benefit from step-by-step reasoning.

**Typical Queries**:

- "Solve the quadratic equation x² + 5x + 6 = 0"
- "Calculate the derivative of f(x) = x³ + 2x² - 5x + 1"
- "What is the probability of getting two heads in three coin flips?"
- "Prove that the square root of 2 is irrational"

**Configuration Considerations**:

- Often benefits from reasoning mode for complex problem-solving
- May require models with strong mathematical capabilities
- Consider higher reasoning effort for advanced mathematical proofs

### Computer Science

**Domain**: Programming, algorithms, data structures, software engineering

**Description**: Covers programming languages, software development, algorithms, system design, and technical computing concepts. Suitable for both theoretical computer science and practical programming tasks.

**Typical Queries**:

- "Implement a binary search algorithm in Python"
- "Explain the time complexity of quicksort"
- "How do I optimize this SQL query?"
- "What's the difference between REST and GraphQL APIs?"

**Configuration Considerations**:

- Benefits from reasoning mode for algorithmic problem-solving
- Requires models with strong coding and technical knowledge
- May need specialized code generation capabilities

### Physics

**Domain**: Physical concepts, mechanics, thermodynamics, electromagnetism

**Description**: Encompasses all areas of physics from classical mechanics to quantum physics and relativity. Handles both theoretical concepts and practical calculations involving physical phenomena.

**Typical Queries**:

- "Calculate the force needed to accelerate a 10kg mass at 5m/s²"
- "Explain Newton's laws of motion"
- "What is the relationship between voltage, current, and resistance?"
- "How does quantum entanglement work?"

**Configuration Considerations**:

- May benefit from reasoning mode for complex physics problems
- Requires models with strong scientific and mathematical knowledge
- Consider specialized physics-trained models for advanced topics

### Chemistry

**Domain**: Chemical reactions, molecular structures, organic/inorganic chemistry

**Description**: Covers chemical processes, molecular interactions, reaction mechanisms, and chemical analysis. Suitable for both theoretical chemistry concepts and practical laboratory applications.

**Typical Queries**:

- "Balance the equation: C₆H₁₂O₆ + O₂ → CO₂ + H₂O"
- "Explain the mechanism of SN2 reactions"
- "What is the molecular geometry of SF₆?"
- "How do catalysts affect reaction rates?"

**Configuration Considerations**:

- Often benefits from reasoning mode for reaction mechanisms
- Requires models with strong chemistry and scientific knowledge
- May need specialized chemical notation and formula handling

### Biology

**Domain**: Life sciences, genetics, ecology, anatomy, physiology

**Description**: Encompasses all biological sciences including molecular biology, genetics, ecology, evolution, and human biology. Handles both descriptive biological concepts and analytical processes.

**Typical Queries**:

- "Explain the process of photosynthesis"
- "How does DNA replication work?"
- "What are the stages of mitosis?"
- "Describe the structure and function of ribosomes"

**Configuration Considerations**:

- May benefit from reasoning mode for complex biological processes
- Requires models with comprehensive biological knowledge
- Consider models trained on scientific literature for accuracy

### Engineering

**Domain**: Technical problem-solving, design, systems analysis

**Description**: Covers various engineering disciplines including mechanical, electrical, civil, and software engineering. Focuses on practical problem-solving and system design.

**Typical Queries**:

- "Design a load-bearing beam for a 20-foot span"
- "How do I calculate the efficiency of this heat exchanger?"
- "What are the trade-offs between different sorting algorithms?"
- "Explain the principles of feedback control systems"

**Configuration Considerations**:

- Often benefits from reasoning mode for design problems
- Requires models with technical and mathematical capabilities
- May need specialized engineering knowledge and calculations

### Business

**Domain**: Business strategy, management, marketing, finance, entrepreneurship

**Description**: Covers business operations, strategic planning, management practices, marketing, finance, and entrepreneurship. Suitable for both theoretical business concepts and practical business advice.

**Typical Queries**:

- "What are the key components of a business plan?"
- "How do I improve team productivity?"
- "Explain different marketing strategies for startups"
- "What is the difference between B2B and B2C sales?"

**Configuration Considerations**:

- Typically conversational, may not require reasoning mode
- Benefits from models with business and management knowledge
- Consider models trained on business literature and case studies

### Law

**Domain**: Legal concepts, regulations, jurisprudence, legal procedures

**Description**: Encompasses legal principles, regulations, court procedures, and jurisprudence across various legal domains. Handles both general legal concepts and specific legal questions.

**Typical Queries**:

- "What are the elements of a valid contract?"
- "Explain the difference between civil and criminal law"
- "What is intellectual property protection?"
- "How does the appeals process work?"

**Configuration Considerations**:

- Usually explanatory, may not require reasoning mode
- Requires models with comprehensive legal knowledge
- Important: Ensure disclaimers about not providing legal advice

### Economics

**Domain**: Economic theory, markets, macroeconomics, microeconomics

**Description**: Covers economic principles, market analysis, fiscal policy, and economic theory. Handles both theoretical economic concepts and practical economic analysis.

**Typical Queries**:

- "Explain supply and demand curves"
- "What causes inflation and how is it measured?"
- "How do interest rates affect the economy?"
- "What is the difference between GDP and GNP?"

**Configuration Considerations**:

- Usually explanatory, may not require reasoning mode
- Benefits from models with strong economic and mathematical knowledge
- Consider models trained on economic literature and data

### Health

**Domain**: Medical information, wellness, healthcare, anatomy

**Description**: Encompasses medical knowledge, health information, anatomy, physiology, and wellness topics. Covers both general health information and specific medical concepts.

**Typical Queries**:

- "What are the symptoms of diabetes?"
- "How does the immune system work?"
- "What are the benefits of regular exercise?"
- "Explain the structure of the human heart"

**Configuration Considerations**:

- Typically informational, may not require reasoning mode
- Requires models with medical and health knowledge
- Important: Ensure disclaimers about not providing medical advice

### Psychology

**Domain**: Mental health, behavior, cognitive science, therapy

**Description**: Covers psychological concepts, mental health topics, cognitive processes, and therapeutic approaches. Handles both theoretical psychology and practical mental health information.

**Typical Queries**:

- "What are the stages of grief?"
- "Explain cognitive behavioral therapy techniques"
- "How does memory formation work?"
- "What is the difference between anxiety and depression?"

**Configuration Considerations**:

- Usually explanatory, may not require reasoning mode
- Benefits from models with psychology and mental health knowledge
- Important: Ensure disclaimers about not providing therapeutic advice

### Philosophy

**Domain**: Philosophical discussions, ethics, logic, metaphysics

**Description**: Encompasses philosophical inquiry, ethical discussions, logical reasoning, and metaphysical concepts. Covers both historical philosophical thought and contemporary philosophical issues.

**Typical Queries**:

- "What is the meaning of life according to different philosophers?"
- "Explain the trolley problem in ethics"
- "What are the main arguments for and against free will?"
- "How do different cultures approach moral reasoning?"

**Configuration Considerations**:

- Typically conversational and exploratory
- May benefit from reasoning mode for complex philosophical arguments
- Requires models with broad philosophical knowledge

### History

**Domain**: Historical events, narratives, civilizations, timelines

**Description**: Covers historical events, civilizations, cultural developments, and historical analysis across all time periods and regions. Handles both factual historical information and historical interpretation.

**Typical Queries**:

- "What were the causes of World War I?"
- "Explain the rise and fall of the Roman Empire"
- "How did the Industrial Revolution change society?"
- "What was the significance of the Renaissance?"

**Configuration Considerations**:

- Usually narrative-based, may not require reasoning mode
- Benefits from models with comprehensive historical knowledge
- Consider models trained on historical texts and sources

### Other

**Domain**: General queries, miscellaneous topics, unclassified content

**Description**: Serves as a catch-all category for queries that don't fit into specific domains. Handles general knowledge questions, casual conversations, and miscellaneous topics.

**Typical Queries**:

- "What's the weather like today?"
- "How do I cook pasta?"
- "What are some good book recommendations?"
- "Tell me a joke"

**Configuration Considerations**:

- Usually doesn't require reasoning mode
- Benefits from models with broad general knowledge
- Often used as fallback when classification confidence is low

## Configuration Guidelines

### Reasoning Configuration

Each category can be configured to enable or disable reasoning based on your needs:

- **STEM Categories** (Math, Physics, Chemistry, Biology, Computer Science, Engineering): Often benefit from reasoning mode for complex problem-solving
- **Professional Categories** (Business, Law, Economics): May or may not require reasoning depending on query complexity
- **Informational Categories** (Health, Psychology, Philosophy, History): Typically explanatory, but reasoning can help with complex analysis
- **General Category** (Other): Usually doesn't require reasoning mode

### Model Selection Strategy

Consider these factors when configuring model preferences:

- **Domain Expertise**: Choose models with strong knowledge in specific domains
- **Reasoning Capability**: Some models excel at step-by-step reasoning
- **Performance Requirements**: Balance accuracy with latency needs
- **Cost Considerations**: Optimize model selection based on computational costs

### Best Practices

1. **Start Simple**: Begin with basic configurations and iterate based on performance
2. **Test Thoroughly**: Validate category classification accuracy with your specific queries
3. **Monitor Performance**: Track classification confidence and routing decisions
4. **Customize Gradually**: Adjust reasoning settings and model scores based on usage patterns

## Performance Considerations

### Classification Accuracy

The category classifier performance varies by domain complexity:

- **STEM Categories**: Generally high accuracy due to distinct technical vocabulary
- **Professional Categories**: Good accuracy with domain-specific terminology
- **General Categories**: May have lower confidence due to broader scope

### Latency Impact

- **Classification Time**: &lt;50ms average for category determination
- **Reasoning Overhead**: Additional 200-500ms when reasoning is enabled
- **Model Selection**: &lt;10ms for routing decision based on configuration

### Optimization Tips

- Adjust confidence thresholds based on your accuracy requirements
- Use reasoning mode selectively to balance quality and performance
- Monitor category distribution to identify potential classification issues
- Consider batch processing for high-volume scenarios

## Future Roadmap

The vLLM Semantic Router category system is continuously evolving to support more sophisticated routing scenarios. Here are the planned category expansions:

### Multimodal Categories

#### Vision + Text

- **Image Analysis**: Visual content understanding, image description, OCR
- **Document Processing**: PDF analysis, form extraction, diagram interpretation
- **Medical Imaging**: Radiology reports, medical image analysis
- **Technical Diagrams**: Engineering drawings, architectural plans, flowcharts

#### Audio + Text

- **Speech Processing**: Transcription, voice commands, audio analysis
- **Music Theory**: Musical composition, theory, instrument identification
- **Audio Content**: Podcast analysis, sound classification

### RAG-Enhanced Categories

#### Knowledge-Intensive Domains

- **Scientific Research**: Literature review, research synthesis, citation analysis
- **Legal Research**: Case law analysis, statute interpretation, legal precedent
- **Medical Research**: Clinical studies, drug interactions, treatment protocols
- **Technical Documentation**: API references, software manuals, troubleshooting

#### Domain-Specific Knowledge Bases

- **Enterprise Knowledge**: Company policies, internal documentation, procedures
- **Academic Research**: Journal articles, thesis work, academic writing
- **Regulatory Compliance**: Industry standards, compliance requirements, auditing

### Specialized Routing Categories

#### Intent-Based Routing

- **Creative Writing**: Story generation, poetry, creative content
- **Code Generation**: Specific programming languages, frameworks, libraries
- **Data Analysis**: Statistical analysis, data visualization, reporting
- **Translation**: Language pairs, cultural context, technical translation

#### Workflow-Based Categories

- **Multi-Step Tasks**: Complex procedures requiring sequential processing
- **Collaborative Tasks**: Multi-agent workflows, review processes
- **Real-Time Processing**: Streaming data, live analysis, immediate responses

### Advanced Classification Features

#### Context-Aware Categories

- **Conversation History**: Context-dependent routing based on dialogue state
- **User Profiles**: Personalized routing based on user expertise and preferences
- **Temporal Context**: Time-sensitive routing for urgent vs. routine queries

#### Confidence-Based Routing

- **Uncertainty Handling**: Specialized routing for ambiguous queries
- **Multi-Category Queries**: Handling queries that span multiple domains
- **Fallback Strategies**: Intelligent degradation when classification confidence is low

### Community Contributions

We welcome community input on category expansion:

- **Feature Requests**: Suggest new categories based on your use cases
- **Domain Expertise**: Contribute domain-specific knowledge for better classification
- **Testing & Feedback**: Help validate new categories in real-world scenarios
- **Custom Categories**: Share successful custom category implementations

## Next Steps

- [**Configuration Guide**](configuration.md) - Learn how to configure category-based routing
- [**Technical Details**](technical-details.md) - Deep dive into classifier implementation
- [**Category Overview**](overview.md) - Understanding the category system architecture
