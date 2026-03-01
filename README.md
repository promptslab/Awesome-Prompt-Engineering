<h2 align="center">Awesome Prompt Engineering 🧙‍♂️</h2>

<p align="center">
  <img width="650" src="https://raw.githubusercontent.com/promptslab/Awesome-Prompt-Engineering/main/_source/prompt.png">
</p>

<p align="center">
  A hand-curated collection of resources for Prompt Engineering and Context Engineering — covering papers, tools, models, APIs, benchmarks, courses, and communities for working with Large Language Models.
</p>

<p align="center">
https://promptslab.github.io
  </p>
 <h4 align="center">
  
  ```
     Master Prompt Engineering. Join the Course at https://promptslab.github.io
  ```
  <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome" /></a>
  <a href="https://github.com/promptslab/Awesome-Prompt-Engineering/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License" /></a>
  <a href="http://makeapullrequest.com"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome" /></a>
  <a href="https://discord.gg/m88xfYMbK6"><img src="https://img.shields.io/badge/Discord-Community-orange" alt="Community" /></a>
  <img src="https://img.shields.io/badge/Last%20Updated-February%202026-brightgreen" alt="Last Updated" />
</p>

---

## 🚀 Start Here

New to prompt engineering? Follow this path:

1. **Learn the basics** → [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) (free, ~90 min)
2. **Read the guide** → [Prompt Engineering Guide by DAIR.AI](https://www.promptingguide.ai/) (open-source, comprehensive)
3. **Study provider docs** → [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) · [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
4. **Understand where the field is heading** → [Anthropic: Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
5. **Read the research** → [The Prompt Report](https://arxiv.org/abs/2406.06608) — taxonomy of 58+ prompting techniques from 1,500+ papers

---

## Table of Contents

- [Papers](#papers)
  - [Major Surveys](#major-surveys)
  - [Prompt Optimization and Automatic Prompting](#prompt-optimization-and-automatic-prompting)
  - [Prompt Compression](#prompt-compression)
  - [Reasoning Advances](#reasoning-advances)
  - [In-Context Learning](#in-context-learning)
  - [Agentic Prompting and Multi-Agent Systems](#agentic-prompting-and-multi-agent-systems)
  - [Multimodal Prompting](#multimodal-prompting)
  - [Structured Output and Format Control](#structured-output-and-format-control)
  - [Prompt Injection and Security](#prompt-injection-and-security)
  - [Applications of Prompt Engineering](#applications-of-prompt-engineering)
  - [Text-to-Image Generation](#text-to-image-generation)
  - [Text-to-Music/Audio Generation](#text-to-musicaudio-generation)
  - [Foundational Papers (Pre-2024)](#foundational-papers-pre-2024)
- [Tools and Code](#tools-and-code)
  - [Prompt Management and Testing](#prompt-management-and-testing)
  - [LLM Evaluation Tools](#llm-evaluation-tools)
  - [Agent Frameworks](#agent-frameworks)
  - [Prompt Optimization Tools](#prompt-optimization-tools)
  - [Red Teaming and Prompt Security](#red-teaming-and-prompt-security)
  - [MCP (Model Context Protocol)](#mcp-model-context-protocol)
  - [Vibe Coding and AI Coding Assistants](#vibe-coding-and-ai-coding-assistants)
  - [Other Notable Repositories](#other-notable-repositories)
- [APIs](#apis)
- [Datasets and Benchmarks](#datasets-and-benchmarks)
- [Models](#models)
- [AI Content Detectors](#ai-content-detectors)
- [Books](#books)
- [Courses](#courses)
- [Tutorials and Guides](#tutorials-and-guides)
- [Videos](#videos)
- [Communities](#communities)
- [How to Contribute](#how-to-contribute)

---

## Papers
📄

### Major Surveys

- [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608) [2024] — Most comprehensive survey: taxonomy of 58 text and 40 multimodal prompting techniques from 1,500+ papers. Co-authored with OpenAI, Microsoft, Google, Stanford.
- [A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications](https://arxiv.org/abs/2402.07927) [2024] — 44 techniques across application areas with per-task performance summaries.
- [A Survey of Prompt Engineering Methods in LLMs for Different NLP Tasks](https://arxiv.org/abs/2407.12994) [2024] — 39 prompting methods across 29 NLP tasks.
- [A Survey of Automatic Prompt Engineering: An Optimization Perspective](https://arxiv.org/abs/2502.11560) [2025] — Formalizes auto-PE methods as discrete/continuous/hybrid optimization problems.
- [Efficient Prompting Methods for Large Language Models: A Survey](https://arxiv.org/abs/2404.01077) [2024] — Survey of efficiency-oriented prompting (compression, optimization, APE) for reducing compute and latency.
- [Navigate through Enigmatic Labyrinth: A Survey of Chain of Thought Reasoning](https://arxiv.org/abs/2309.15402) [2023, ACL 2024] — Systematic CoT survey.
- [Demystifying Chains, Trees, and Graphs of Thoughts](https://arxiv.org/abs/2401.14295) [2024] — Unified framework for multi-prompt reasoning topologies.
- [Towards Goal-oriented Prompt Engineering for Large Language Models: A Survey](https://arxiv.org/abs/2401.14043) [2024] — Focuses on prompts designed around explicit task goals.
- [Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning LLMs](https://arxiv.org/abs/2503.09567) [2025] — Distinguishes Long CoT from Short CoT in o1/R1-era models.

### Prompt Optimization and Automatic Prompting

- [OPRO: Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409) [2023, NeurIPS 2024] — Uses LLMs as optimizers via meta-prompts; optimized prompts outperform human-designed ones by up to 50% on BBH.
- [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714) [2023, ICLR 2024] — Framework for programming (not prompting) LLMs with automatic prompt optimization.
- [MIPRO: Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://arxiv.org/abs/2406.11695) [2024, EMNLP 2024] — Bayesian optimization for multi-stage LM programs; up to 13% accuracy gains.
- [TextGrad: Automatic "Differentiation" via Text](https://arxiv.org/abs/2406.07496) [2024] — Treats compound AI systems as computation graphs with textual feedback as gradients. Published in Nature.
- [EvoPrompt](https://arxiv.org/abs/2309.08532) [2023, ACL 2024] — Evolutionary algorithm approach for automatically optimizing discrete prompts.
- [Meta Prompting for AI Systems](https://arxiv.org/abs/2311.11482) [2023, ICLR 2024 Workshop] — Example-agnostic structural templates formalized using category theory.
- [Prompt Engineering a Prompt Engineer (PE²)](https://arxiv.org/abs/2311.05661) [2024, ACL Findings] — Uses LLMs to meta-prompt themselves, refining prompts with step-by-step templates to significantly improve reasoning.
- [Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/abs/2211.01910) [2022] — Automatic prompt generation via APE.
- [Hard Prompts Made Easy: Gradient-Based Discrete Optimization for Prompt Tuning](https://arxiv.org/abs/2302.03668) [2023]
- [SPO: Self-Supervised Prompt Optimization](https://arxiv.org/abs/2502.06855) [2025] — Competitive performance at 1–6% of the cost of prior methods.

### Prompt Compression

- [LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression](https://arxiv.org/abs/2403.12968) [2024, ACL 2024] — 3x–6x faster than LLMLingua with GPT-4 data distillation.
- [LongLLMLingua](https://arxiv.org/abs/2310.06839) [2023, ACL 2024] — Question-aware compression for long contexts; 21.4% performance boost with 4x fewer tokens.
- [Prompt Compression for Large Language Models: A Survey](https://arxiv.org/abs/2410.12388) [2024] — Comprehensive survey of hard and soft prompt compression methods.

### Reasoning Advances

- [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) [2024] — Shows optimal test-time compute allocation can outperform 14x larger models.
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) [2025] — Pure RL-trained reasoning model matching o1; open-source with distilled variants.
- [s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393) [2025] — SFT on just 1,000 examples creates competitive reasoning model via "budget forcing."
- [Reasoning Language Models: A Blueprint](https://arxiv.org/abs/2501.11223) [2025] — Systematic framework organizing reasoning LM approaches.
- [Demystifying Long Chain-of-Thought Reasoning in LLMs](https://arxiv.org/abs/2502.03373) [2025] — Analyzes long CoT behavior in modern reasoning models.
- [Graph of Thoughts: Solving Elaborate Problems with LLMs](https://arxiv.org/abs/2308.09687) [2023, AAAI 2024] — Models thoughts as arbitrary graphs; 62% quality improvement over ToT on sorting.
- [Tree of Thoughts: Deliberate Problem Solving with LLMs](https://arxiv.org/abs/2305.10601) [2023, NeurIPS 2023] — Tree search over reasoning paths.
- [Everything of Thoughts](https://arxiv.org/abs/2311.04254) [2023] — Integrates CoT, ToT, and external solvers via MCTS.
- [Skeleton-of-Thought](https://arxiv.org/abs/2307.15337) [2023] — Parallel decoding via answer skeleton generation for up to 2.69x speedup.
- [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) [2022] — The foundational CoT paper.
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) [2022] — Aggregating multiple CoT outputs for reliability.
- [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) [2022] — "Let's think step by step" as a zero-shot reasoning trigger.
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) [2022] — Interleaving reasoning and tool use.

### In-Context Learning

- [Many-Shot In-Context Learning](https://arxiv.org/abs/2404.11018) [2024, NeurIPS 2024 Spotlight] — Significant gains scaling ICL to hundreds/thousands of examples; introduces Reinforced and Unsupervised ICL.
- [Many-Shot In-Context Learning in Multimodal Foundation Models](https://arxiv.org/abs/2405.09798) [2024] — Scales multimodal ICL to ~2,000 examples across 14 datasets.
- [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/abs/2202.12837) [2022]
- [Fantastically Ordered Prompts and Where to Find Them](https://arxiv.org/abs/2104.08786) [2021] — Overcoming few-shot prompt order sensitivity.
- [Calibrate Before Use: Improving Few-Shot Performance of Language Models](https://arxiv.org/abs/2102.09690) [2021]

### Agentic Prompting and Multi-Agent Systems

- [Agentic Large Language Models: A Survey](https://arxiv.org/abs/2503.23037) [2025] — Comprehensive survey organizing agentic LLMs by reasoning, acting, and interacting capabilities.
- [Large Language Model based Multi-Agents: A Survey of Progress and Challenges](https://arxiv.org/abs/2402.01680) [2024] — Covers profiling, communication, and growth mechanisms.
- [Multi-Agent Collaboration Mechanisms: A Survey of LLMs](https://arxiv.org/abs/2501.06322) [2025] — Reviews debate and cooperation strategies in LLM-based multi-agent systems.
- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155) [2023] — Microsoft's foundational multi-agent framework paper.
- [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs](https://arxiv.org/abs/2307.16789) [2023, ICLR 2024] — Trains LLMs to use massive real-world API collections.
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770) [2023, ICLR 2024] — The benchmark driving agentic coding progress.
- [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688) [2023, ICLR 2024] — Benchmark across 8 environments.
- [PAL: Program-aided Language Models](https://arxiv.org/abs/2211.10435) [2023] — Offloading computation to code interpreters.

### Multimodal Prompting

- [Visual Prompting in Multimodal Large Language Models: A Survey](https://arxiv.org/abs/2409.15310) [2024] — First comprehensive survey on visual prompting methods in MLLMs.
- [Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V](https://arxiv.org/abs/2310.11441) [2023] — Visual markers dramatically improve visual grounding.
- [A Comprehensive Survey and Guide to Multimodal Large Language Models in Vision-Language Tasks](https://arxiv.org/abs/2411.06284) [2024] — Covers text, image, video, audio MLLMs.
- [Multimodal Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2302.00923) [2023]
- [From Prompt Engineering to Prompt Craft](https://arxiv.org/abs/2411.13422) [2024] — Design-research view of prompt "craft" for diffusion models.

### Structured Output and Format Control

- [Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of LLMs](https://arxiv.org/abs/2408.02442) [2024] — Examines how constraining outputs to structured formats impacts reasoning performance.
- [Batch Prompting: Efficient Inference with LLM APIs](https://arxiv.org/abs/2301.08721) [2023]
- [Structured Prompting: Scaling In-Context Learning to 1,000 Examples](https://arxiv.org/abs/2212.06713) [2022]

### Prompt Injection and Security

- [Formalizing and Benchmarking Prompt Injection Attacks and Defenses](https://arxiv.org/abs/2310.12815) [2023, USENIX Security 2024] — Formal framework with systematic evaluation of 5 attacks and 10 defenses across 10 LLMs.
- [The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions](https://arxiv.org/abs/2404.13208) [2024] — OpenAI's priority-level training for injection defense.
- [AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses](https://arxiv.org/abs/2406.13352) [2024] — Realistic agent scenario benchmark.
- [InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents](https://arxiv.org/abs/2403.02691) [2024]
- [SecAlign: Defending Against Prompt Injection with Preference Optimization](https://arxiv.org/abs/2410.05451) [2024] — DPO-based defense.
- [WASP: Benchmarking Web Agent Security Against Prompt Injection](https://arxiv.org/abs/2504.18575) [2025] — Security benchmark for web/computer-use agents.
- [Many-Shot Jailbreaking](https://www.anthropic.com/research/many-shot-jailbreaking) [2024] — Scaling harmful examples in long-context windows enables jailbreaking (Anthropic Technical Report).
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) [2022]
- [Ignore Previous Prompt: Attack Techniques For Language Models](https://arxiv.org/abs/2211.09527) [2022]
- [Artificial Intelligence and Cybersecurity: Documented Risks, Enterprise Guardrails, and Emerging Threats in 2024–2025](https://www.ijfmr.com/research-paper.php?id=62200) [2025] — Survey of real prompt-injection incidents with practical governance prompt patterns.

### Applications of Prompt Engineering

- [Rephrase and Respond: Let Large Language Models Ask Better Questions for Themselves](https://arxiv.org/abs/2311.04205) [2023]
- [Legal Prompt Engineering for Multilingual Legal Judgement Prediction](https://arxiv.org/abs/2212.02199) [2023]
- [Conversing with Copilot: Exploring Prompt Engineering for Solving CS1 Problems](https://arxiv.org/abs/2210.15157) [2022]
- [Commonsense-Aware Prompting for Controllable Empathetic Dialogue Generation](https://arxiv.org/abs/2302.01441) [2023]
- [PLACES: Prompting Language Models for Social Conversation Synthesis](https://arxiv.org/abs/2302.03269) [2023]
- [Medical Image Segmentation Using Transformer Encoders and Prompt-Based Learning: A Systematic Review](https://ieeexplore.ieee.org/document/11313186/) [2025]
- [TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning](https://arxiv.org/abs/2506.10380) [2025] — SQL-based interface preserving tabular structure for multi-hop queries.

### Text-to-Image Generation

- [A Taxonomy of Prompt Modifiers for Text-To-Image Generation](https://arxiv.org/abs/2204.13988) [2022]
- [Design Guidelines for Prompt Engineering Text-to-Image Generative Models](https://arxiv.org/abs/2109.06977) [2021]
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) [2021]
- [DALL·E: Creating Images from Text](https://arxiv.org/abs/2102.12092) [2021]
- [Investigating Prompt Engineering in Diffusion Models](https://arxiv.org/abs/2211.15462) [2022]

### Text-to-Music/Audio Generation

- [MusicLM: Generating Music From Text](https://arxiv.org/abs/2301.11325) [2023]
- [ERNIE-Music: Text-to-Waveform Music Generation with Diffusion Models](https://arxiv.org/pdf/2302.04456) [2023]
- [AudioLM: A Language Modeling Approach to Audio Generation](https://arxiv.org/pdf/2209.03143) [2023]
- [Make-An-Audio: Text-To-Audio Generation with Prompt-Enhanced Diffusion Models](https://arxiv.org/pdf/2301.12661.pdf) [2023]

### Foundational Papers (Pre-2024)

These papers established the core concepts that modern prompt engineering builds on:

- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) [2020] — Demonstrated few-shot prompting at scale.
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) [2021]
- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) [2021]
- [Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm](https://arxiv.org/abs/2102.07350) [2021]
- [Show Your Work: Scratchpads for Intermediate Computation with Language Models](https://arxiv.org/abs/2112.00114) [2021]
- [Generated Knowledge Prompting for Commonsense Reasoning](https://arxiv.org/abs/2110.08387) [2021]
- [Making Pre-trained Language Models Better Few-shot Learners](https://aclanthology.org/2021.acl-long.295) [2021]
- [AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts](https://arxiv.org/abs/2010.15980) [2020]
- [How Can We Know What Language Models Know?](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00324/96460/) [2020]
- [A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT](https://arxiv.org/abs/2302.11382) [2023]
- [Synthetic Prompting: Generating Chain-of-Thought Demonstrations for LLMs](https://arxiv.org/abs/2302.00618) [2023]
- [Progressive Prompts: Continual Learning for Language Models](https://arxiv.org/abs/2301.12314) [2023]
- [Successive Prompting for Decompleting Complex Questions](https://arxiv.org/abs/2212.04092) [2022]
- [Decomposed Prompting: A Modular Approach for Solving Complex Tasks](https://arxiv.org/abs/2210.02406) [2022]
- [PromptChainer: Chaining Large Language Model Prompts through Visual Programming](https://arxiv.org/abs/2203.06566) [2022]
- [Ask Me Anything: A Simple Strategy for Prompting Language Models](https://paperswithcode.com/paper/ask-me-anything-a-simple-strategy-for) [2022]
- [Prompting GPT-3 To Be Reliable](https://arxiv.org/abs/2210.09150) [2022]
- [On Second Thought, Let's Not Think Step by Step! Bias and Toxicity in Zero-Shot Reasoning](https://arxiv.org/abs/2212.08061) [2022]

---

## Tools and Code
🔧

### Prompt Management and Testing

| Name | Description | Link |
|:-----|:-----------|:----:|
| **Promptfoo** | Open-source CLI for testing, evaluating, and red-teaming LLM prompts. YAML configs, CI/CD integration, adversarial testing. ~9K+ ⭐ | [GitHub](https://github.com/promptfoo/promptfoo) |
| **Promptify** | Solve NLP Problems with LLM's & Easily generate different NLP Task prompts for popular generative models like GPT, PaLM, and more with Promptify | [[Github]](https://github.com/promptslab/Promptify) |
| **Agenta** | Open-source LLM developer platform for prompt management, evaluation, human feedback, and deployment. | [GitHub](https://github.com/Agenta-AI/agenta) |
| **PromptLayer** | Version, test, and monitor every prompt and agent with robust evals, tracing, and regression sets. | [Website](https://promptlayer.com/) |
| **Helicone** | Production prompt monitoring and optimization platform. | [Website](https://helicone.ai/) |
| **LangGPT** | Framework for structured and meta-prompt design. 10K+ ⭐ | [GitHub](https://github.com/langgpt/LangGPT) |
| **ChainForge** | Visual toolkit for building, testing, and comparing LLM prompt responses without code. | [GitHub](https://github.com/ianarawjo/ChainForge) |
| **LMQL** | A query language for LLMs making complex prompt logic programmable. | [GitHub](https://github.com/eth-sri/lmql) |
| **Promptotype** | Platform for developing, testing, and managing structured LLM prompts. | [Website](https://www.promptotype.io) |
| **PromptPanda** | AI-powered prompt management system for streamlining prompt workflows. | [Website](https://promptpanda.io) |
| **Promptimize AI** | Browser extension to automatically improve user prompts for any AI model. | [Website](https://promptimize.ai) |
| **PROMPTMETHEUS** | Web-based "Prompt Engineering IDE" for iteratively creating and running prompts. | [Website](https://promptmetheus.com) |
| **Better Prompt** | Test suite for LLM prompts before pushing to production. | [GitHub](https://github.com/krrishdholakia/betterprompt) |
| **OpenPrompt** | Open-source framework for prompt-learning research. | [GitHub](https://github.com/thunlp/OpenPrompt) |
| **Prompt Source** | Toolkit for creating, sharing, and using natural language prompts. | [GitHub](https://github.com/bigscience-workshop/promptsource) |
| **Prompt Engine** | NPM utility library for creating and maintaining prompts for LLMs (Microsoft). | [GitHub](https://github.com/microsoft/prompt-engine) |
| **PromptInject** | Framework for quantitative analysis of LLM robustness to adversarial prompt attacks. | [GitHub](https://github.com/agencyenterprise/PromptInject) |

### LLM Evaluation Tools

| Name | Description | Link |
|:-----|:-----------|:----:|
| **DeepEval** | Open-source evaluation framework covering RAG, agents, and conversations with CI/CD integration. ~7K+ ⭐ | [GitHub](https://github.com/confident-ai/deepeval) |
| **Ragas** | RAG evaluation with knowledge-graph-based test set generation and 30+ metrics. ~8K+ ⭐ | [GitHub](https://github.com/explodinggradients/ragas) |
| **LangSmith** | LangChain's platform for debugging, testing, evaluating, and monitoring LLM applications. | [Website](https://smith.langchain.com/) |
| **Langfuse** | Open-source LLM observability with tracing, prompt management, and human annotation. ~7K+ ⭐ | [GitHub](https://github.com/langfuse/langfuse) |
| **Braintrust** | End-to-end AI evaluation platform, SOC2 Type II certified. | [Website](https://www.braintrust.dev/) |
| **Arize AI / Phoenix** | Real-time LLM monitoring with drift detection and tracing. | [GitHub](https://github.com/Arize-ai/phoenix) |
| **TruLens** | Evaluating and explaining LLM apps; tracks hallucinations, relevance, groundedness. | [GitHub](https://github.com/truera/trulens) |
| **InspectAI** | Purpose-built for evaluating agents against benchmarks (UK AISI). | [GitHub](https://github.com/UKGovernmentBEIS/inspect_ai) |
| **Opik** | Evaluate, test, and ship LLM applications across dev and production lifecycles. | [GitHub](https://github.com/comet-ml/opik) |

### Agent Frameworks

| Name | Description | Link |
|:-----|:-----------|:----:|
| **LangChain / LangGraph** | Most widely adopted LLM app framework; LangGraph adds graph-based multi-step agent workflows. ~100K+ / ~10K+ ⭐ | [GitHub](https://github.com/langchain-ai/langchain) · [LangGraph](https://github.com/langchain-ai/langgraph) |
| **CrewAI** | Role-playing AI agent orchestration with 700+ integrations. ~44K+ ⭐ | [GitHub](https://github.com/crewAIInc/crewAI) |
| **AutoGen (AG2)** | Microsoft's multi-agent conversational framework. ~40K+ ⭐ | [GitHub](https://github.com/microsoft/autogen) |
| **DSPy** | Stanford's framework for programming LLMs with automatic prompt/weight optimization. ~22K+ ⭐ | [GitHub](https://github.com/stanfordnlp/dspy) |
| **OpenAI Agents SDK** | Official agent framework with function calling, guardrails, and handoffs. ~10K+ ⭐ | [GitHub](https://github.com/openai/openai-agents-python) |
| **Semantic Kernel** | Microsoft's AI framework powering M365 Copilot; C#, Python, Java. ~24K+ ⭐ | [GitHub](https://github.com/microsoft/semantic-kernel) |
| **LlamaIndex** | Data framework for RAG and agent capabilities. ~40K+ ⭐ | [GitHub](https://github.com/run-llama/llama_index) |
| **Haystack** | Open-source NLP framework with pipeline architecture for RAG and agents. ~20K+ ⭐ | [GitHub](https://github.com/deepset-ai/haystack) |
| **Agno (formerly Phidata)** | Python agent framework with microsecond instantiation. ~20K+ ⭐ | [GitHub](https://github.com/agno-agi/agno) |
| **Smolagents** | Hugging Face's minimalist code-centric agent framework (~1000 LOC). ~15K+ ⭐ | [GitHub](https://github.com/huggingface/smolagents) |
| **Pydantic AI** | Type-safe agent framework using Pydantic for structured validation. ~8K+ ⭐ | [GitHub](https://github.com/pydantic/pydantic-ai) |
| **Mastra** | TypeScript AI agent framework with assistants, RAG, and observability. ~20K+ ⭐ | [GitHub](https://github.com/mastra-ai/mastra) |
| **Google ADK** | Agent Development Kit deeply integrated with Gemini and Google Cloud. | [GitHub](https://github.com/google/adk-python) |
| **Strands Agents (AWS)** | Model-agnostic framework with deep AWS integrations. | [GitHub](https://github.com/strands-agents/sdk-python) |
| **Langflow** | Node-based visual agent builder with drag-and-drop. ~50K+ ⭐ | [GitHub](https://github.com/langflow-ai/langflow) |
| **n8n** | Workflow automation with AI agent capabilities and 400+ integrations. ~60K+ ⭐ | [GitHub](https://github.com/n8n-io/n8n) |
| **Dify** | All-in-one backend for agentic workflows with tool-using agents and RAG. | [GitHub](https://github.com/langgenius/dify) |
| **PraisonAI** | Multi-AI Agents framework with 100+ LLM support, MCP integration, and built-in memory. | [GitHub](https://github.com/MervinPraison/PraisonAI) |
| **Neurolink** | Multi-provider AI agent framework unifying 12+ providers with workflow orchestration. | [GitHub](https://github.com/juspay/neurolink) |
| **Composio** | Connect 100+ tools to AI agents with zero setup. | [GitHub](https://github.com/composiohq/composio) |

### Prompt Optimization Tools

| Name | Description | Link |
|:-----|:-----------|:----:|
| **DSPy** | Multiple optimizers (MIPROv2, BootstrapFewShot, COPRO) for automatic prompt tuning. ~22K+ ⭐ | [GitHub](https://github.com/stanfordnlp/dspy) |
| **TextGrad** | Automatic differentiation via text (Stanford). ~2K+ ⭐ | [GitHub](https://github.com/zou-group/textgrad) |
| **OPRO** | Google DeepMind's optimization by prompting. | [GitHub](https://github.com/google-deepmind/opro) |

### Red Teaming and Prompt Security

| Name | Description | Link |
|:-----|:-----------|:----:|
| **Garak (NVIDIA)** | LLM vulnerability scanner for hallucination, injection, and jailbreaks — the "nmap for LLMs." ~3K+ ⭐ | [GitHub](https://github.com/NVIDIA/garak) |
| **PyRIT (Microsoft)** | Python Risk Identification Tool for automated red-teaming. ~3K+ ⭐ | [GitHub](https://github.com/Azure/PyRIT) |
| **DeepTeam** | 40+ vulnerabilities, 10+ attack methods, OWASP Top 10 support. | [GitHub](https://github.com/confident-ai/deepteam) |
| **LLM Guard** | Security toolkit for LLM I/O validation. ~2K+ ⭐ | [GitHub](https://github.com/protectai/llm-guard) |
| **NeMo Guardrails (NVIDIA)** | Programmable guardrails for conversational systems. ~5K+ ⭐ | [GitHub](https://github.com/NVIDIA/NeMo-Guardrails) |
| **Guardrails AI** | Define strict output formats (JSON schemas) to ensure system reliability. | [Website](https://www.guardrailsai.com) |
| **Lakera** | AI security platform for real-time prompt injection detection. | [Website](https://lakera.ai/) |
| **Purple Llama (Meta)** | Open-source LLM safety evaluation including CyberSecEval. | [GitHub](https://github.com/meta-llama/PurpleLlama) |
| **GPTFuzz** | Automated jailbreak template generation achieving >90% success rates. | [GitHub](https://github.com/sherdencooper/GPTFuzz) |
| **Rebuff** | Open-source tool for detection and prevention of prompt injection. | [GitHub](https://github.com/protectai/rebuff) |

### MCP (Model Context Protocol)

MCP is an open standard developed by Anthropic (Nov 2024, donated to Linux Foundation Dec 2025) for connecting AI assistants to external data sources and tools through a standardized interface. It has **97M+ monthly SDK downloads** and has been adopted by GitHub, Google, and most major AI providers.

| Name | Description | Link |
|:-----|:-----------|:----:|
| **MCP Specification** | The core protocol specification and SDKs. ~15K+ ⭐ | [GitHub](https://github.com/modelcontextprotocol/modelcontextprotocol) |
| **MCP Reference Servers** | Official implementations: fetch, filesystem, GitHub, Slack, Postgres. | [GitHub](https://github.com/modelcontextprotocol/servers) |
| **FastMCP (Python)** | High-level Pythonic framework for building MCP servers. ~5K+ ⭐ | [GitHub](https://github.com/jlowin/fastmcp) |
| **GitHub MCP Server** | GitHub's official MCP server for repo, issue, PR, and Actions interaction. ~15K+ ⭐ | [GitHub](https://github.com/github/github-mcp-server) |
| **Awesome MCP Servers** | Curated list of 10,000+ community MCP servers. ~30K+ ⭐ | [GitHub](https://github.com/punkpeye/awesome-mcp-servers) |
| **Context7** | MCP server providing version-specific documentation to reduce code hallucination. | [GitHub](https://github.com/upstash/context7) |
| **GitMCP** | Creates remote MCP servers for any GitHub repo by changing the domain. | [Website](https://gitmcp.io/) |
| **MCP Inspector** | Visual testing tool for MCP server development. | [GitHub](https://github.com/modelcontextprotocol/inspector) |

### Vibe Coding and AI Coding Assistants

| Name | Description | Link |
|:-----|:-----------|:----:|
| **Claude Code** | Anthropic's command-line AI coding tool; widely considered one of the best AI coding assistants (2026). | [Docs](https://docs.anthropic.com/en/docs/claude-code) |
| **Cursor** | AI-native code editor; Composer feature generates entire applications from natural language. | [Website](https://cursor.sh/) |
| **Windsurf (Codeium)** | "First agentic IDE" with multi-file editing and project-wide context. | [Website](https://codeium.com/windsurf) |
| **GitHub Copilot** | AI pair programmer; ~30% of new GitHub code comes from Copilot. | [Website](https://github.com/features/copilot) |
| **Aider** | Open-source terminal AI pair programmer with Git integration. ~25K+ ⭐ | [GitHub](https://github.com/paul-gauthier/aider) |
| **Cline** | Open-source VS Code AI assistant connecting editor and terminal through MCP. ~20K+ ⭐ | [GitHub](https://github.com/cline/cline) |
| **Continue** | Open-source IDE extensions for custom AI code assistants. ~22K+ ⭐ | [GitHub](https://github.com/continuedev/continue) |
| **OpenAI Codex CLI** | Lightweight terminal coding agent. | [GitHub](https://github.com/openai/codex) |
| **Gemini CLI** | Google's open-source terminal AI agent. | [GitHub](https://github.com/google-gemini/gemini-cli) |
| **Bolt.new** | Browser-based prompt-to-app generation with one-click deployment. | [Website](https://bolt.new/) |
| **Lovable** | Full-stack apps from natural language descriptions. | [Website](https://lovable.dev/) |
| **v0 (Vercel)** | AI assistant for building Next.js frontend components from text. | [Website](https://v0.dev/) |
| **Firebase Studio** | Google's agentic cloud-based development environment. | [Website](https://firebase.google.com/studio) |

### Other Notable Repositories

| Name | Description | Link |
|:-----|:-----------|:----:|
| **Prompt Engineering Guide (DAIR.AI)** | The definitive open-source guide and resource hub. 3M+ learners. ~55K+ ⭐ | [GitHub](https://github.com/dair-ai/Prompt-Engineering-Guide) |
| **Awesome ChatGPT Prompts / Prompts.chat** | World's largest open-source prompt library. 1000s of prompts for all major models. | [GitHub](https://github.com/f/awesome-chatgpt-prompts) |
| **12-Factor Agents** | Principles for building production-grade LLM-powered software. ~17K+ ⭐ | [GitHub](https://github.com/humanlayer/12-factor-agents) |
| **NirDiamant/Prompt_Engineering** | 22 hands-on Jupyter Notebook tutorials. ~3K+ ⭐ | [GitHub](https://github.com/NirDiamant/Prompt_Engineering) |
| **Context Engineering Repository** | First-principles handbook for moving beyond prompt engineering to context design. | [GitHub](https://github.com/davidkimai/Context-Engineering) |
| **AI Agent System Prompts Library** | Collection of system prompts from production AI coding agents (Claude Code, Gemini CLI, Cline, Aider, Roo Code). | [GitHub](https://github.com/tallesborges/agentic-system-prompts) |
| **Awesome Vibe Coding** | Curated list of 245+ tools and resources for building software through natural language prompts. | [GitHub](https://github.com/taskade/awesome-vibe-coding) |
| **OpenAI Cookbook** | Official recipes for prompts, tools, RAG, and evaluations. | [GitHub](https://github.com/openai/openai-cookbook) |
| **Embedchain** | Framework to create ChatGPT-like bots over your dataset. | [GitHub](https://github.com/embedchain/embedchain) |
| **ThoughtSource** | Framework for the science of machine thinking. | [GitHub](https://github.com/OpenBioLink/ThoughtSource) |
| **Promptext** | Extracts and formats code context for AI prompts with token counting. | [GitHub](https://github.com/1broseidon/promptext) |
| **Price Per Token** | Compare LLM API pricing across 200+ models. | [Website](https://pricepertoken.com/) |

---

## APIs
💻

### OpenAI

| Model | Context | Price (Input/Output per 1M tokens) | Key Feature |
|:------|:--------|:-----------------------------------|:------------|
| GPT-5.2 / 5.2 Thinking | 400K | $1.75 / $14 | Latest flagship, 90% cached discount, configurable reasoning |
| GPT-5.1 | 400K | $1.25 / $10 | Previous generation flagship |
| GPT-4.1 / 4.1 mini / nano | 1M | $2 / $8 | Best non-reasoning model, 40% faster and 80% cheaper than GPT-4o |
| o3 / o3-pro | 200K | Varies | Reasoning models with native tool use |
| o4-mini | 200K | Cost-efficient | Fast reasoning, best on AIME at its cost class |
| GPT-OSS-120B / 20B | 128K | $0.03 / $0.30 | First open-weight models, Apache 2.0 |

Key features: Responses API, Agents SDK, Structured Outputs, function calling, prompt caching (90% discount), Batch API (50% discount), MCP support. [Platform Docs](https://platform.openai.com/docs/models)

### Anthropic (Claude)

| Model | Context | Price (Input/Output per 1M tokens) | Key Feature |
|:------|:--------|:-----------------------------------|:------------|
| Claude Opus 4.6 | 1M (beta) | $5 / $25 | Most powerful, state-of-the-art coding and agentic tasks |
| Claude Sonnet 4.5 | 200K | $3 / $15 | Best coding model, 61.4% OSWorld (computer use) |
| Claude Haiku 4.5 | 200K | Fast tier | Near-frontier, fastest model class |
| Claude Opus 4 / Sonnet 4 | 200K | $15/$75 (Opus) | Opus: 72.5% SWE-bench, Sonnet 4 powers GitHub Copilot |

Key features: Extended Thinking with tool use, Computer Use, MCP (originated here), prompt caching, Claude Code CLI, available on AWS Bedrock and Google Vertex AI. [API Docs](https://docs.anthropic.com/)

### Google (Gemini)

| Model | Context | Price (Input/Output per 1M tokens) | Key Feature |
|:------|:--------|:-----------------------------------|:------------|
| Gemini 3 Pro Preview | 1M | $2 / $12 | Most intelligent Google model, deployed to 2B+ Search users |
| Gemini 2.5 Pro | 1M | $1.25 / $10 | Best for coding/agentic tasks, thinking model |
| Gemini 2.5 Flash / Flash-Lite | 1M | $0.30/$1.50 · $0.10/$0.40 | Price-performance leaders |

Key features: Thinking (all 2.5+ models), Google Search grounding, code execution, Live API (real-time audio/video), context caching. [Google AI Studio](https://ai.google.dev/)

### Meta (Llama)

| Model | Architecture | Context | Key Feature |
|:------|:------------|:--------|:------------|
| Llama 4 Scout | 109B MoE / 17B active | 10M | Fits single H100, multimodal, open-weight |
| Llama 4 Maverick | 400B MoE / 17B active, 128 experts | 1M | Beats GPT-4o, open-weight |
| Llama 3.3 70B | Dense | 128K | Matches Llama 3.1 405B |

Available on 25+ cloud partners, Hugging Face, and inference APIs. [Llama](https://ai.meta.com/llama/)

### Other Notable Providers

| Provider | Description | Link |
|:---------|:-----------|:----:|
| **Mistral AI** | Mistral Large 3 (675B MoE), Devstral 2, Ministral 3. Apache 2.0. | [Website](https://mistral.ai) |
| **DeepSeek** | V3.2 (671B MoE), R1 (reasoning, MIT license). $0.15/$0.75 per 1M tokens. | [Website](https://deepseek.com) |
| **xAI (Grok)** | Grok 4.1 Fast: 2M context, $0.20/$0.50 per 1M tokens. | [Website](https://x.ai) |
| **Cohere** | Command A (111B, 256K context), Embed v4, Rerank 4.0. Excels at RAG. | [Website](https://cohere.com) |
| **Together AI** | 200+ open models with sub-100ms latency. | [Website](https://together.ai) |
| **Groq** | LPU hardware with ~300+ tokens/sec inference. | [Website](https://groq.com) |
| **Fireworks AI** | Fast inference with HIPAA + SOC2 compliance. | [Website](https://fireworks.ai) |
| **OpenRouter** | Unified API for 300+ models from all providers. | [Website](https://openrouter.ai) |
| **Cerebras** | Wafer-scale chips with best total response time. | [Website](https://cerebras.ai) |
| **Perplexity AI** | Search-augmented API with citations. | [Website](https://perplexity.ai) |
| **Amazon Bedrock** | Managed multi-model service with Claude, Llama, Mistral, Cohere. | [Website](https://aws.amazon.com/bedrock/) |
| **Hugging Face Inference** | Access to open models via API. | [Website](https://huggingface.co/docs/api-inference/index) |

---

## Datasets and Benchmarks
💾

### Major Benchmarks (2024–2026)

| Name | Description | Link |
|:-----|:-----------|:----:|
| **Chatbot Arena / LM Arena** | 6M+ user votes for Elo-rated pairwise LLM comparisons. De facto standard for human preference. | [Website](https://lmarena.ai/) |
| **MMLU-Pro** | 12,000+ graduate-level questions across 14 domains. NeurIPS 2024 Spotlight. | [GitHub](https://github.com/TIGER-AI-Lab/MMLU-Pro) |
| **GPQA** | 448 "Google-proof" STEM questions; non-expert validators achieve only 34%. | [arXiv](https://arxiv.org/abs/2311.12022) |
| **SWE-bench Verified** | Human-validated 500-task subset for real-world GitHub issue resolution. | [Website](https://www.swebench.com/) |
| **SWE-bench Pro** | 1,865 tasks across 41 professional repos; best models score only ~23%. | [Leaderboard](https://scale.com/leaderboard/swe_bench_pro_public) |
| **Humanity's Last Exam (HLE)** | 2,500 expert-vetted questions; top AI scores only ~10–30%. | [Website](https://agi.safe.ai/) |
| **BigCodeBench** | 1,140 coding tasks across 7 domains; AI achieves ~35.5% vs. 97% human success. | [Leaderboard](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard) |
| **LiveBench** | Contamination-resistant with frequently updated questions. | [Paper](https://openreview.net/forum?id=sKYHBTAxVa) |
| **FrontierMath** | Research-level math; AI solves only ~2% of problems. | Research |
| **ARC-AGI v2** | Abstract reasoning measuring fluid intelligence. | Research |
| **IFEval** | Instruction-following evaluation with formatting/content constraints. | [arXiv](https://arxiv.org/abs/2311.07911) |
| **MLE-bench** | OpenAI's ML engineering evaluation via Kaggle-style tasks. | [GitHub](https://github.com/openai/mle-bench) |
| **PaperBench** | Evaluates AI's ability to replicate 20 ICML 2024 papers from scratch. | [GitHub](https://github.com/openai/preparedness) |

### Leaderboards and Meta-Benchmarks

| Name | Description | Link |
|:-----|:-----------|:----:|
| **Hugging Face Open LLM Leaderboard v2** | Evaluates open models on MMLU-Pro, GPQA, IFEval, MATH. | [Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) |
| **Artificial Analysis Intelligence Index v3** | Aggregates 10 evaluations. | [Website](https://artificialanalysis.ai/) |
| **SEAL by Scale AI** | Hosts SWE-bench Pro and agentic evaluations. | [Leaderboard](https://scale.com/leaderboard) |

### Prompt and Instruction Datasets

| Name | Description | Link |
|:-----|:-----------|:----:|
| **P3 (Public Pool of Prompts)** | Prompt templates for 270+ NLP tasks used to train T0 and similar models. | [HuggingFace](https://huggingface.co/datasets/bigscience/P3) |
| **System Prompts Dataset** | 944 system prompt templates for agent workflows (by Daniel Rosehill, Aug 2025). | [HuggingFace](https://huggingface.co/datasets/danielrosehill/system_prompts) |
| **OpenAssistant Conversations (OASST)** | 161,443 messages in 35 languages with 461,292 quality ratings. | [HuggingFace](https://huggingface.co/datasets/OpenAssistant/oasst1) |
| **UltraChat / UltraFeedback** | Large-scale synthetic instruction and preference datasets for alignment training. | HuggingFace |
| **SoftAge Prompt Engineering Dataset** | 1,000 diverse prompts across 10 categories for benchmarking prompt performance. | HuggingFace |
| **Text Transformation Prompt Library** | Comprehensive collection of text transformation prompts (May 2025). | HuggingFace |
| **Writing Prompts** | ~300K human-written stories paired with prompts from r/WritingPrompts. | [Kaggle](https://www.kaggle.com/datasets/ratthachat/writing-prompts) |
| **Midjourney Prompts** | Text prompts and image URLs scraped from MidJourney's public Discord. | [HuggingFace](https://huggingface.co/datasets/succinctly/midjourney-prompts) |
| **CodeAlpaca-20k** | 20,000 programming instruction-output pairs. | [HuggingFace](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) |
| **ProPEX-RAG** | Dataset for prompt optimization in RAG workflows. | HuggingFace |
| **NanoBanana Trending Prompts** | 1,000+ curated AI image prompts from X/Twitter, ranked by engagement. | [GitHub](https://github.com/jau123/nanobanana-trending-prompts) |

### Red Teaming and Adversarial Datasets

| Name | Description | Link |
|:-----|:-----------|:----:|
| **HarmBench** | 510 harmful behaviors across standard, contextual, copyright, and multimodal categories. | [Website](https://safetyprompts.com/) |
| **JailbreakBench** | Open robustness benchmark for jailbreaking with 100 prompts. | Research |
| **AgentHarm** | 110 malicious agent tasks across 11 harm categories. | [arXiv](https://arxiv.org/abs/2410.09024) |
| **DecodingTrust** | 243,877 prompts evaluating trustworthiness across 8 perspectives. | Research |
| **SafetyPrompts.com** | Aggregator tracking 50+ safety/red-teaming datasets. | [Website](https://safetyprompts.com/) |

---

## Models
🧠

### Frontier Models (2025–2026)

| Model | Provider | Context | Key Strength |
|:------|:---------|:--------|:-------------|
| **GPT-5.2** | OpenAI | 400K | General intelligence, 100% AIME 2025 |
| **Claude Opus 4.6** | Anthropic | 1M (beta) | Coding, agentic tasks, extended thinking |
| **Gemini 3 Pro** | Google | 1M | #1 LMArena (~1500 Elo), multimodal |
| **Grok 4.1** | xAI | 2M | #2 LMArena (1483 Elo), low hallucination |
| **Mistral Large 3** | Mistral AI | 256K | Best open-weight (675B MoE/41B active), Apache 2.0 |
| **DeepSeek-V3.2** | DeepSeek | 128K | Best value (671B MoE/37B active), MIT license |
| **Llama 4 Maverick** | Meta | 1M | Beats GPT-4o (400B MoE/17B active), open-weight |

### Reasoning Models

| Model | Key Detail |
|:------|:-----------|
| **OpenAI o3 / o3-pro** | 87.7% GPQA Diamond. Native tool use. |
| **OpenAI o4-mini** | Best AIME at its cost class with visual reasoning. |
| **DeepSeek-R1 / R1-0528** | Open-weight, RL-trained. 87.5% on AIME 2025. MIT license. |
| **QwQ (Qwen with Questions)** | 32B reasoning model. Apache 2.0. Comparable to R1. |
| **Gemini 2.5 Pro/Flash (Thinking)** | Built-in reasoning with configurable thinking budget. |
| **Claude Extended Thinking** | Hybrid mode with visible chain-of-thought and tool use. |
| **Phi-4 Reasoning / Plus** | 14B reasoning models rivaling much larger models. Open-weight. |
| **GPT-OSS-120B** | OpenAI's open-weight with CoT. Near-parity with o4-mini. Apache 2.0. |

### Notable Open-Source Models

| Model | Provider | Key Detail |
|:------|:---------|:-----------|
| **Qwen3-235B-A22B** | Alibaba | Flagship MoE. Strong reasoning/code/multilingual. Apache 2.0. Most downloaded family on HuggingFace. |
| **Gemma 3** | Google | 270M to 27B. Multimodal. 128K context. 140+ languages. |
| **OLMo 2/3** | Allen AI | Fully open (data, code, weights, logs). OLMo 2 32B surpasses GPT-3.5. Apache 2.0. |
| **SmolLM3-3B** | Hugging Face | Outperforms Llama-3.2-3B. Dual-mode reasoning. 128K context. |
| **Kimi K2** | Moonshot AI | 32B active. Open-weight. Tailored for coding/agentic use. |
| **Llama 4 Scout** | Meta | 109B MoE/17B active. 10M token context. Fits single H100. |

### Code-Specialized Models

| Model | Key Detail |
|:------|:-----------|
| **Qwen3-Coder (480B-A35B)** | 69.6% SWE-bench — milestone for open-source coding. 256K context. Apache 2.0. |
| **Devstral 2 (123B)** | 72.2% SWE-bench Verified. 7x more cost-efficient than Claude Sonnet. |
| **Codestral 25.01** | Mistral's code model. 80+ languages. Fill-in-the-Middle support. |
| **DeepSeek-Coder-V2** | 236B MoE / 21B active. 338 programming languages. |
| **Qwen 2.5-Coder** | 7B/32B. 92 programming languages. 88.4% HumanEval. Apache 2.0. |

### Foundational Models (Historical Reference)

These models established key concepts but are largely superseded for practical use:

| Model | Provider | Significance |
|:------|:---------|:-------------|
| BLOOM 176B | BigScience | First major open multilingual LLM (2022) |
| GLM-130B | Tsinghua | Open bilingual English/Chinese LLM (2023) |
| Falcon 180B | TII | Large open generative model (2023) |
| Mixtral 8x7B | Mistral AI | Pioneered MoE architecture for open models (2023) |
| GPT-NeoX-20B | EleutherAI | Early open autoregressive LLM |
| GPT-J-6B | EleutherAI | Early open causal language model |

---

## AI Content Detectors
🔎

### Leading Commercial Detectors

| Name | Accuracy | Key Feature | Link |
|:-----|:---------|:------------|:----:|
| **GPTZero** | 99% claimed | 10M+ users, #1 on G2 (2025). Detects GPT-4/5, Gemini, Claude, Llama. Free tier available. | [Website](https://gptzero.me) |
| **Originality.ai** | 98–100% (peer-reviewed) | Consistently rated most accurate. Combines AI detection + plagiarism + fact checking. From $14.95/month. | [Website](https://originality.ai) |
| **Turnitin AI Detection** | 98%+ on unmodified AI text | Dominant in academia. Launched AI bypasser/humanizer detection (Aug 2025). Institutional licensing. | [Website](https://www.turnitin.com/solutions/topics/ai-writing/) |
| **Copyleaks** | 99%+ claimed | Enterprise tool detecting AI in 30+ languages. LMS integrations. | [Website](https://copyleaks.com) |
| **Winston AI** | 99.98% claimed | OCR for scanned documents, AI image/deepfake detection. 11 languages. | [Website](https://gowinston.ai) |
| **Pangram Labs** | 99.3% (COLING 2025) | Highest score in COLING 2025 Shared Task. 100% TPR on "humanized" text. 97.7% adversarial robustness. | [Website](https://www.pangram.com) |

### Free and Research Detectors

| Name | Description | Link |
|:-----|:-----------|:----:|
| **Binoculars** | Open-source research detector using cross-perplexity between two LLMs. | [arXiv](https://arxiv.org/abs/2401.12070) |
| **DetectGPT / Fast-DetectGPT** | Statistical method comparing log-probabilities of original text vs. perturbations. | [arXiv](https://arxiv.org/abs/2301.11305) |
| **Openai Detector** | AI classifier for indicating AI-written text (OpenAI Detector Python wrapper)  | [[GitHub]](https://github.com/promptslab/openai-detector) |
| **Sapling AI Detector** | Free browser-based detector (up to 2,000 chars). 97% accuracy in some studies. | [Website](https://sapling.ai/) |
| **QuillBot AI Detector** | Free, no sign-up required. | [Website](https://quillbot.com/ai-content-detector) |
| **Writer AI Content Detector** | Free tool with color-coded results. | [Website](https://writer.com/ai-content-detector/) |
| **ZeroGPT** | Popular free detector evaluated in multiple academic studies. | [Website](https://www.zerogpt.com/) |

### Watermarking Approaches

| Name | Description | Link |
|:-----|:-----------|:----:|
| **SynthID (Google DeepMind)** | Watermarking for AI text, images, and audio via statistical token sampling. Deployed in Google products. | [Website](https://deepmind.google/technologies/synthid/) |
| **OpenAI Text Watermarking** | Developed but still experimental as of 2025. Research shows fragility concerns. | Experimental |

**Important caveat:** No detector claims 100% accuracy. Mixed human/AI text remains hardest to detect (50–70% accuracy). Adversarial robustness varies widely. The AI detection market is projected to grow from ~$2.3B (2025) to $15B by 2035.

---

## Books
📖

### Prompt Engineering

| Title | Author(s) | Publisher | Year |
|:------|:----------|:---------|:-----|
| **Prompt Engineering for LLMs** | John Berryman & Albert Ziegler | O'Reilly | 2024 |
| **Prompt Engineering for Generative AI** | James Phoenix & Mike Taylor | O'Reilly | 2024 |
| **Prompt Engineering for LLMs** | Thomas R. Caldwell | Independent | 2025 |

### LLM Application Development

| Title | Author(s) | Publisher | Year |
|:------|:----------|:---------|:-----|
| **AI Engineering: Building Applications with Foundation Models** | Chip Huyen | O'Reilly | 2025 |
| **Build a Large Language Model (From Scratch)** | Sebastian Raschka | Manning | 2024 |
| **Building LLMs for Production** | Louis-François Bouchard & Louie Peters | O'Reilly | 2024 |
| **LLM Engineer's Handbook** | Paul Iusztin & Maxime Labonne | Packt | 2024 |
| **The Hundred-Page Language Models Book** | Andriy Burkov | Self-Published | 2025 |

### AI Agents

| Title | Author(s) | Publisher | Year |
|:------|:----------|:---------|:-----|
| **Building Applications with AI Agents** | Michael Albada | O'Reilly | 2025 |
| **AI Agents and Applications** | Roberto Infante | Manning | 2025 |
| **AI Agents in Action** | Micheal Lanham | Manning | 2025 |

### Production, Reliability, and Security

| Title | Author(s) | Publisher | Year |
|:------|:----------|:---------|:-----|
| **LLMs in Production** | Christopher Brousseau & Matthew Sharp | Manning | 2025 |
| **Building Reliable AI Systems** | Rush Shahani | Manning | 2025 |
| **The Developer's Playbook for LLM Security** | Steve Wilson | O'Reilly | 2024 |

---

## Courses
👩‍🏫

### Free Short Courses

- [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) — Co-taught by Andrew Ng and OpenAI's Isa Fulford. The foundational starting point. (DeepLearning.AI)
- [Building Systems with the ChatGPT API](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/) — Multi-step LLM system design for production. (DeepLearning.AI)
- [AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) — Agentic dataflows with tool use and research agents. (DeepLearning.AI)
- [Building Agentic RAG with LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/) — RAG research agent construction. (DeepLearning.AI)
- [Functions, Tools and Agents with LangChain](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/) — Function calling and agent building. (DeepLearning.AI)
- [Prompt Engineering for Vision Models](https://www.deeplearning.ai/short-courses/prompt-engineering-for-vision-models/) — Visual prompting techniques. (DeepLearning.AI)

### University and Platform Courses

- [Prompt Engineering Specialization (Vanderbilt)](https://www.coursera.org/specializations/prompt-engineering) — 3-course series by Dr. Jules White covering foundational to advanced PE. (Coursera)
- [Generative AI with LLMs (DeepLearning.AI + AWS)](https://www.coursera.org/learn/generative-ai-with-llms) — LLM lifecycle, transformers, RLHF, deployment. (Coursera)
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) — Build an LLM end-to-end. (Stanford, 2024–2026)
- [MIT 6.S191: Introduction to Deep Learning](https://introtodeeplearning.com/) — Annual course including LLMs and generative AI. (MIT, 2024–2026)
- [The Complete Prompt Engineering for AI Bootcamp](https://www.udemy.com/course/prompt-engineering-for-ai/) — Covers GPT-5, DSPy, LangGraph, agent architectures. 58K+ ratings. (Udemy, updated Feb 2026)

### Free Platform Courses

- [Google Prompting Essentials](https://grow.google/prompting-essentials/) — 5-step prompt design, meta-prompting, Gemini. Under 6 hours.
- [Microsoft Azure AI Fundamentals: Generative AI](https://learn.microsoft.com/en-us/training/paths/introduction-generative-ai/) — Free learning path covering LLMs, prompts, agents, Azure OpenAI.
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) — Community-driven course covering transformers, fine-tuning, building reasoning models.
- [Hugging Face AI Agents Course](https://huggingface.co/learn) — Agent theory to practice. 100K+ registered students.

### Learn Prompting Courses

- [ChatGPT for Everyone](https://learnprompting.org/courses/chatgpt-for-everyone)
- [Introduction to Prompt Engineering](https://learnprompting.org/courses/introduction_to_prompt_engineering)
- [Advanced Prompt Engineering](https://learnprompting.org/courses/advanced-prompt-engineering)
- [Introduction to Prompt Hacking](https://learnprompting.org/courses/intro-to-prompt-hacking)
- [Advanced Prompt Hacking](https://learnprompting.org/courses/advanced-prompt-hacking)
- [Introduction to Generative AI Agents for Business Professionals](https://learnprompting.org/courses/introduction-to-agents)
- [AI Safety](https://learnprompting.org/courses/ai-safety)

---

## Tutorials and Guides
📚

### Official Provider Guides

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) — Comprehensive, covering GPT-4.1/5 prompting, reasoning models, structured outputs, agentic workflows. Continuously updated.
- [OpenAI GPT-4.1 Prompting Guide](https://cookbook.openai.com/articles/gpt-4-1-prompting-guide) [2025] — Structured agent-like prompt design: goal persistence, tool integration, long-context processing.
- [Anthropic Prompt Engineering Overview](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) — Iterative prompt design, XML tags, chain-of-thought, role assignment. Includes prompt generator.
- [Anthropic Claude 4 Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices) [2025–2026] — Parallel tool execution, thinking capabilities, image processing.
- [Anthropic: Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) [2025] — The evolution from prompt engineering to context engineering: agent state, memory, tools, MCP.
- [Google Gemini Prompting Strategies](https://ai.google.dev/docs/prompt_best_practices) — Multimodal prompting for Gemini via Vertex AI and AI Studio.
- [Microsoft Prompt Engineering in Azure AI Studio](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering) — Tool calling, function design, few-shot prompting, prompt chaining.

### Community and Independent Guides

- [Prompt Engineering Guide (DAIR.AI / promptingguide.ai)](https://www.promptingguide.ai/) — Most comprehensive open-source guide. 18+ techniques, model-specific guides, research papers. 3M+ learners. Now includes context engineering.
- [Learn Prompting (learnprompting.org)](https://learnprompting.org/) — Structured free platform. Beginner to advanced PE, AI security, HackAPrompt competition.
- [IBM 2026 Guide to Prompt Engineering](https://www.ibm.com/think/prompt-engineering) [2026] — Curated tools, tutorials, real-world examples with Python code.
- [Anthropic Interactive Tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial) — 9-chapter Jupyter notebook course with hands-on exercises.
- [Lilian Weng's Prompt Engineering Guide](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) [2023] — Highly respected technical blog from OpenAI researcher.
- [Google Prompt Engineering Guide (68-page PDF)](https://www.reddit.com/r/PromptEngineering/comments/1kggmh0/google_dropped_a_68page_prompt_engineering_guide/) [2025] — Internal-style best-practice guide for Gemini with concrete patterns.
- [DigitalOcean: Prompt Engineering Best Practices](https://www.digitalocean.com/resources/articles/prompt-engineering-best-practices) [2025] — Updated guide summarizing techniques: few-shot, chain-of-thought, role prompting, etc.
- [Aakash Gupta: Prompt Engineering in 2025](https://news.aakashg.com) [2025] — Practical guide with wisdom from shipping AI at OpenAI, Shopify, and Google.
- [Best practices for prompt engineering with OpenAI API](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api) — OpenAI's introductory best practices.
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook) — Official recipes for function calling, RAG, evaluation, and complex workflows.
- [Microsoft Prompt Engineering Docs](https://microsoft.github.io/prompt-engineering) — Microsoft's open prompt engineering resources.
- [DALLE Prompt Book](https://dallery.gallery/the-dalle-2-prompt-book) — Visual guide for text-to-image prompting.
- [Best 100+ Stable Diffusion Prompts](https://mpost.io/best-100-stable-diffusion-prompts-the-most-beautiful-ai-text-to-image-prompts) — Community-curated image generation prompts.
- [Vibe Engineering (Manning)](https://www.manning.com/books/vibe-engineering) — Book by Tomasz Lelek & Artur Skowronski on building software through natural language prompts.
- [Prompt Guide](https://prompt-guide.com) — Free, interactive prompt engineering platform with 300+ curated prompts, 30+ AI agent workflows, AI-evaluated exercises, and a glossary. Available in English and French.

---

## Videos
🎥

- [Andrej Karpathy: "Deep Dive into LLMs" & "How I Use LLMs"](https://www.youtube.com/@AndrejKarpathy) [2024–2025] — Two of the most influential AI videos of 2024–2025. Comprehensive technical deep dive followed by practical usage patterns.
- [Karpathy: "Software in the Era of AI" (YC AI Startup School)](https://karpathy.ai/) [2025] — Coined "vibe coding" (Feb 2025) and championed "context engineering" (Jun 2025).
- [Karpathy: Neural Networks: Zero to Hero](https://www.youtube.com/@AndrejKarpathy) [2023–2024] — Full lecture series building from backpropagation to GPT.
- [3Blue1Brown: Neural Networks Series](https://www.youtube.com/@3blue1brown) [Updated 2024] — Iconic animated visual explanations of transformers and attention mechanisms. 7M+ subscribers.
- [AI Explained](https://www.youtube.com/@aiexplained-official) [2024–2025] — Long-form analysis breaking down papers, model capabilities, and PE developments.
- [Sam Witteveen](https://www.youtube.com/@samwitteveen) [2024–2025] — Practical tutorials on prompt engineering, LangChain, RAG, and agents.
- [Matthew Berman](https://www.youtube.com/@matthew_berman) [2024–2025] — Popular channel covering model releases and practical LLM usage. 600K+ subscribers.
- [DeepLearning.AI YouTube](https://www.youtube.com/@Deeplearningai) [2024–2026] — Structured lessons, course previews, and Andrew Ng talks on agents and AI careers.
- [Lex Fridman Podcast (AI Episodes)](https://www.youtube.com/@lexfridman) [2024–2025] — Long-form interviews with Altman, Hinton, Amodei on LLMs, prompting, and safety.
- [ICSE 2025: AIware Prompt Engineering Tutorial](https://conf.researchr.org/details/icse-2025/icse-2025-tutorials/) [2025] — Conference tutorial covering prompt patterns, fragility, anti-patterns, and optimization DSLs.
- [CMU Advanced NLP 2022: Prompting](https://youtube.com/watch?v=5ef83Wljm-M) — Foundational academic lecture on prompting methods.
- [ChatGPT: 5 Prompt Engineering Secrets For Beginners](https://www.youtube.com/watch?v=2zg3V66-Fzs) — Accessible intro for beginners.

---

## Communities
🤝

### Discord Servers

- [Learn Prompting](https://learnprompting.org/discord) — 40,000+ members. Largest PE Discord with courses, hackathons, HackAPrompt competitions.
- [PromptsLab Discord](https://discord.gg/m88xfYMbK6)  - Community
- [Midjourney](https://discord.gg/midjourney) — 1M+ members. Primary hub for text-to-image prompt sharing.
- [OpenAI Discord](https://discord.gg/openai) — Official community with channels for GPTs, Sora, DALL-E, and API help.
- [Anthropic Discord](https://discord.gg/anthropic) — Official Claude community for AI development collaboration.
- [Hugging Face Discord](https://discord.gg/huggingface) — Model discussions, library support, community events.
- [FlowGPT](https://flowgpt.com/) — 33K+ members. 100K+ prompts across ChatGPT, DALL-E, Stable Diffusion, Claude.

### Reddit

- [r/PromptEngineering](https://reddit.com/r/PromptEngineering) — Dedicated subreddit for prompt crafting techniques and discussions.
- [r/ChatGPT](https://reddit.com/r/ChatGPT) — 10M+ members. Primary hub for ChatGPT users and prompt sharing.
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA) — Highly technical community for running open-source LLMs locally.
- [r/ClaudeAI](https://reddit.com/r/ClaudeAI) — Anthropic's Claude community: prompt sharing, API tips, model comparisons.
- [r/MachineLearning](https://reddit.com/r/MachineLearning) — Academic-oriented ML research discussions.
- [r/OpenAI](https://reddit.com/r/OpenAI) — OpenAI product and API discussions.
- [r/StableDiffusion](https://reddit.com/r/StableDiffusion) — 450K+ members for AI art prompting and workflows.
- [r/ChatGPTPromptGenius](https://reddit.com/r/ChatGPTPromptGenius) — 35K+ members sharing and refining prompts.


### Forums and Platforms

- [OpenAI Developer Community](https://community.openai.com/) — Official forum for API help, best practices, project sharing.
- [Hugging Face Community](https://huggingface.co/) — Hub for open-source AI collaboration.
- [DeepLearning.AI Community](https://community.deeplearning.ai/) — Forum for learners discussing courses and AI careers.
- [LessWrong](https://www.lesswrong.com/) — In-depth technical posts on AI capabilities and safety.
- [AI Alignment Forum](https://www.alignmentforum.org/) — Specialized alignment research discussions.
- [CivitAI](https://civitai.com/) — Generative AI creators platform for sharing models, LoRAs, and prompts.

### GitHub Organizations

- [LangChain](https://github.com/langchain-ai) — Open-source LLM app framework. 100K+ stars.
- [Promptslab](https://github.com/promptslab)  — Generative Models | Prompt-Engineering | LLMs 
- [Hugging Face](https://github.com/huggingface) — Central hub: Transformers, Diffusers, Datasets, TRL.
- [DSPy (Stanford NLP)](https://github.com/stanfordnlp/dspy) — Growing community for systematic prompt optimization.
- [OpenAI](https://github.com/openai) — Open-source models, benchmarks, and tools.

---

## How to Contribute

We welcome contributions to this list! Before contributing, please take a moment to review our [contribution guidelines](contributing.md). These guidelines will help ensure that your contributions align with our objectives and meet our standards for quality and relevance.

**What we're looking for:**
- New high-quality papers, tools, or resources with a brief description of why they matter
- Updates to existing entries (broken links, outdated information)
- Corrections to star counts, pricing, or model details
- Translations and accessibility improvements

**Quality standards:**
- All tools should be actively maintained (updated within the last 6 months)
- Papers should be from peer-reviewed venues or have significant community adoption
- Datasets should be publicly accessible
- Please include a one-line description explaining why the resource is valuable

Thank you for your interest in contributing to this project!

---

<p align="center">
  <sub>Maintained by <a href="https://promptslab.github.io">PromptsLab</a> · <a href="https://github.com/promptslab/Awesome-Prompt-Engineering">Star this repo</a> if you find it useful!</sub>
</p>
