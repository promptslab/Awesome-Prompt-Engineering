<h2 align="center">Awesome Prompt Engineering 🧙‍♂️ </h2>

<p align="center">
  <img width="650" src="https://raw.githubusercontent.com/promptslab/Awesome-Prompt-Engineering/main/_source/prompt.png">
</p>

<p align="center">
  <p align="center"> This repository contains a hand-curated resources for Prompt Engineering with a focus on Generative Pre-trained Transformer (GPT), ChatGPT, PaLM etc

</p>
 <h4 align="center">
  
  ```
     Prompt Engineering Course is coming soon..
  ```
  
  <a href="https://awesome.re">
    <img src="https://awesome.re/badge.svg" alt="Awesome" />
  </a>
  <a href="https://github.com/promptslab/Awesome-Prompt-Engineering/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Awesome-Prompt-Engineering is released under the Apache 2.0 license." />
  </a>
  <a href="http://makeapullrequest.com">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="http://makeapullrequest.com" />
  </a>
  <a href="https://discord.gg/m88xfYMbK6">
    <img src="https://img.shields.io/badge/Discord-Community-orange" alt="Community" />
  </a>
  <a href="https://colab.research.google.com/drive/1f4YG9stX9aHmsmh6ZhzjekJU4X4BIynO?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab" />
  </a>
</h4>


# Table of Contents

- [Papers](#papers)
- [Tools & Code](#tools--code)
- [Apis](#apis)
- [Datasets](#datasets)
- [Models](#datasets)
- [Educational](#educational)
  - [Tutorials](#tutorials)
- [Videos](#videos)
- [Books](#books)
- [Communities](#communities)
- [Other Resources](#other-resources)


## Papers
📄
- **Prompt Engineering Techniques**:

  - [Hard Prompts Made Easy: Gradient-Based Discrete Optimization for Prompt Tuning and Discovery](https://arxiv.org/abs/2302.03668) [2023] (Arxiv)
  - [Synthetic Prompting: Generating Chain-of-Thought Demonstrations for Large Language Models](https://arxiv.org/abs/2302.00618) [2023] (Arxiv) 
  - [Progressive Prompts: Continual Learning for Language Models](https://arxiv.org/abs/2301.12314) [2023] (Arxiv) 
  - [Batch Prompting: Efficient Inference with LLM APIs](https://arxiv.org/abs/2301.08721) [2023] (Arxiv)
  - [Successive Prompting for Decompleting Complex Questions](https://arxiv.org/abs/2212.04092) [2022] (Arxiv) 
  - [Structured Prompting: Scaling In-Context Learning to 1,000 Examples](https://arxiv.org/abs/2212.06713) [2022] (Arxiv) 
  - [Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/abs/2211.01910) [2022] (Arxiv) 
  - [Ask Me Anything: A simple strategy for prompting language models](https://paperswithcode.com/paper/ask-me-anything-a-simple-strategy-for) [2022] (Arxiv) 
  - [Prompting GPT-3 To Be Reliable](https://arxiv.org/abs/2210.09150) [2022](Arxiv) 
  - [Decomposed Prompting: A Modular Approach for Solving Complex Tasks](https://arxiv.org/abs/2210.02406) [2022] (Arxiv) 
  - [PromptChainer: Chaining Large Language Model Prompts through Visual Programming](https://arxiv.org/abs/2203.06566) [2022] (Arxiv) 
  - [Investigating Prompt Engineering in Diffusion Models](https://arxiv.org/abs/2211.15462) [2022] (Arxiv) 
  - [Show Your Work: Scratchpads for Intermediate Computation with Language Models](https://arxiv.org/abs/2112.00114) [2021] (Arxiv) 
  - [Reframing Instructional Prompts to GPTk's Language](https://arxiv.org/abs/2109.07830) [2021] (Arxiv) 
  - [Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity](https://arxiv.org/abs/2104.08786) [2021] (Arxiv) 
  - [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) [2021] (Arxiv) 
  - [Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm](https://arxiv.org/abs/2102.07350) [2021] (Arxiv) 
  - [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) [2021] (Arxiv) 
  
 
- **Reasoning and In-Context Learning**:

  - [Multimodal Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2302.00923) [2023] (Arxiv) 
  - [On Second Thought, Let's Not Think Step by Step! Bias and Toxicity in Zero-Shot Reasoning](https://arxiv.org/abs/2212.08061) [2022] (Arxiv) 
  - [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) [2022] (Arxiv) 
  - [Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought](https://arxiv.org/abs/2210.01240v3) [2022] (Arxiv) 
  - [On the Advance of Making Language Models Better Reasoners](https://arxiv.org/abs/2206.02336) [2022] (Arxiv) 
  - [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) [2022] (Arxiv) 
  - [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) [2022] (Arxiv) 
  - [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/abs/2202.12837) [2022] (Arxiv) 
  - [Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering](https://arxiv.org/abs/2209.09513v2) [2022] (Arxiv) 
  - [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) [2021] (Arxiv) 
  - [Generated Knowledge Prompting for Commonsense Reasoning](https://arxiv.org/abs/2110.08387) [2021] (Arxiv) 
  - [BERTese: Learning to Speak to BERT](https://aclanthology.org/2021.eacl-main.316) [2021] (Acl) 
  
  
- **Evaluating and Improving Language Models**:

  - [Large Language Models Can Be Easily Distracted by Irrelevant Context](https://arxiv.org/abs/2302.00093) [2023] (Arxiv) 
  - [Crawling the Internal Knowledge-Base of Language Models](https://arxiv.org/abs/2301.12810) [2023] (Arxiv) 
  - [Discovering Language Model Behaviors with Model-Written Evaluations](https://arxiv.org/abs/2212.09251) [2022] (Arxiv) 
  - [Calibrate Before Use: Improving Few-Shot Performance of Language Models](https://arxiv.org/abs/2102.09690) [2021] (Arxiv) 
  
  
- **Applications of Language Models**:

  - [Prompting for Multimodal Hateful Meme Classification](https://arxiv.org/abs/2302.04156) [2023] (Arxiv) 
  - [PLACES: Prompting Language Models for Social Conversation Synthesis](https://arxiv.org/abs/2302.03269) [2023] (Arxiv) 
  - [Commonsense-Aware Prompting for Controllable Empathetic Dialogue Generation](https://arxiv.org/abs/2302.01441) [2023] (Arxiv) 
  - [PAL: Program-aided Language Models](https://arxiv.org/abs/2211.10435) [2023](Arxiv) 
  - [Legal Prompt Engineering for Multilingual Legal Judgement Prediction](https://arxiv.org/abs/2212.02199) [2023] (Arxiv) 
  - [Conversing with Copilot: Exploring Prompt Engineering for Solving CS1 Problems Using Natural Language](https://arxiv.org/abs/2210.15157) [2022] (Arxiv) 
  - [Plot Writing From Scratch Pre-Trained Language Models](https://aclanthology.org/2022.inlg-main.5) [2022] (Acl) 
  - [AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts](https://arxiv.org/abs/2010.15980) [2020] (Arxiv) 
  
  
- **Threat Detection and Adversarial Examples**:

  - [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) [2022] (Arxiv) 
  - [Ignore Previous Prompt: Attack Techniques For Language Models](https://arxiv.org/abs/2211.09527) [2022] (Arxiv) 
  - [Machine Generated Text: A Comprehensive Survey of Threat Models and Detection Methods](https://arxiv.org/abs/2210.07321) [2022] (Arxiv) 
  - [Evaluating the Susceptibility of Pre-Trained Language Models via Handcrafted Adversarial Examples](https://arxiv.org/abs/2209.02128) [2022] (Arxiv) 
  - [Toxicity Detection with Generative Prompt-based Inference](https://arxiv.org/abs/2205.12390) [2022] (Arxiv) 
  - [How Can We Know What Language Models Know?](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00324/96460/How-Can-We-Know-What-Language-Models-Know) [2020] (Mit) 
  
  
- **Few-shot Learning and Performance Optimization**:

  - [Promptagator: Few-shot Dense Retrieval From 8 Examples](https://arxiv.org/abs/2209.11755) [2022] (Arxiv) 
  - [The Unreliability of Explanations in Few-shot Prompting for Textual Reasoning](https://arxiv.org/abs/2205.03401) [2022] (Arxiv) 
  - [Making Pre-trained Language Models Better Few-shot Learners](https://aclanthology.org/2021.acl-long.295) [2021] (Acl) 
  - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) [2020] (Arxiv) 
  
  
- **Text to Image Generation**:

  - [A Taxonomy of Prompt Modifiers for Text-To-Image Generation](https://arxiv.org/abs/2204.13988) [2022] (Arxiv) 
  - [Design Guidelines for Prompt Engineering Text-to-Image Generative Models](https://arxiv.org/abs/2109.06977) [2021] (Arxiv)
  - [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) [2021] (Arxiv)
  - [DALL·E: Creating Images from Text](https://arxiv.org/abs/2102.12092) [2021] (Arxiv)
  
- **Text to Music/Sound Generation**:

  - [MusicLM: Generating Music From Text](https://arxiv.org/abs/2301.11325) [2023] (Arxiv) 
  - [ERNIE-Music: Text-to-Waveform Music Generation with Diffusion Models](https://arxiv.org/pdf/2302.04456) [2023] (Arxiv)
  - [Noise2Music: Text-conditioned Music Generation with Diffusion Models](https://arxiv.org/abs/2301.11325) [2023) (Arxiv)
  - [AudioLM: a Language Modeling Approach to Audio Generation](https://arxiv.org/pdf/2209.03143) [2023] (Arxiv)
  - [Make-An-Audio: Text-To-Audio Generation with Prompt-Enhanced Diffusion Models](https://arxiv.org/pdf/2301.12661.pdf) [2023] (Arxiv)
  
- **Text to Video Generation**:

  - [Dreamix: Video Diffusion Models are General Video Editors](https://arxiv.org/pdf/2302.01329.pdf) [2023] (Arxiv) 
  - [Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation](https://arxiv.org/pdf/2212.11565.pdf) [2022] (Arxiv)
  - [Noise2Music: Text-conditioned Music Generation with Diffusion Models](https://arxiv.org/abs/2301.11325) [2023) (Arxiv)
  - [AudioLM: a Language Modeling Approach to Audio Generation](https://arxiv.org/pdf/2209.03143) [2023] (Arxiv)
  
  
- **Overviews**:

  - [Piloting Copilot and Codex: Hot Temperature, Cold Prompts, or Black Magic?](https://arxiv.org/abs/2210.14699) [2022] (Arxiv) 
  
  


## Tools & Code
🔧

|      Name                | Description  | Url |
| :-------------------- | :----------: | :----------: |
| **GPT Index** | GPT Index is a project consisting of a set of data structures designed to make it easier to use large external knowledge bases with LLMs. | [[Github]](https://github.com/jerryjliu/gpt_index) |
| **Promptify** | Solve NLP Problems with LLM's & Easily generate different NLP Task prompts for popular generative models like GPT, PaLM, and more with Promptify | [[Github]](https://github.com/promptslab/Promptify) |
| **Better Prompt** | Test suite for LLM prompts before pushing them to PROD | [[Github]](https://github.com/krrishdholakia/betterprompt) |
| **Interactive Composition Explorerx** | ICE is a Python library and trace visualizer for language model programs. | [[Github]](https://github.com/oughtinc/ice) |
| **LangChainx** | Building applications with LLMs through composability | [[Github]](https://github.com/hwchase17/langchain) |
| **OpenPrompt** | An Open-Source Framework for Prompt-learning | [[Github]](https://github.com/thunlp/OpenPrompt) |
| **Prompt Engine** | This repo contains an NPM utility library for creating and maintaining prompts for Large Language Models (LLMs). | [[Github]](https://github.com/microsoft/prompt-engine) |
| **PromptInject** | PromptInject is a framework that assembles prompts in a modular fashion to provide a quantitative analysis of the robustness of LLMs to adversarial prompt attacks. | [[Github]](https://github.com/agencyenterprise/PromptInject) |
| **Prompts AI** | Advanced playground for GPT-3 | [[Github]](https://github.com/sevazhidkov/prompts-ai) |
| **Prompt Source** | PromptSource is a toolkit for creating, sharing and using natural language prompts. | [[Github]](https://github.com/bigscience-workshop/promptsource) |
| **ThoughtSource** | A framework for the science of machine thinking | [[Github]](https://github.com/OpenBioLink/ThoughtSource) |


## Apis
💻

|      Name                | Description  | Url | Paid or Open-Source |
| :-------------------- | :----------: | :----------: | :----------: |
| **OpenAI** | GPT-n for natural language tasks, Codex for translates natural language to code, and DALL·E for creates and edits original images | [[OpenAI]](https://openai.com/api/) | Paid |
| **CohereAI** | Cohere provides access to advanced Large Language Models and NLP tools through one API | [[CohereAI]](https://cohere.ai/) | Paid |
| **Anthropic** | Coming soon | [[Anthropic]](https://www.anthropic.com/) | Paid |
| **FLAN-T5 XXL** | Coming soon | [[HugginFace]](https://huggingface.co/docs/api-inference/index) | Open-Source |



## Datasets
💾

|      Name                | Description  | Url |
| :-------------------- | :----------: | :----------: |
| **P3 (Public Pool of Prompts)** | P3 (Public Pool of Prompts) is a collection of prompted English datasets covering a diverse set of NLP tasks. | [[HuggingFace]](https://huggingface.co/datasets/bigscience/P3) |
| **Awesome ChatGPT Prompts** | Repo includes ChatGPT prompt curation to use ChatGPT better. | [[Github]](https://github.com/f/awesome-chatgpt-prompts) |
| **Writing Prompts** | Collection of a large dataset of 300K human-written stories paired with writing prompts from an online forum(reddit) | [[Kaggle]](https://www.kaggle.com/datasets/ratthachat/writing-prompts) |
| **Midjourney Prompts** | Text prompts and image URLs scraped from MidJourney's public Discord server | [[HuggingFace]](https://huggingface.co/datasets/succinctly/midjourney-prompts) |


## Models
🧠

|      Name                | Description  | Url | 
| :-------------------- | :----------: | :----------: |
| **ChatGPT** | ChatGPT  | [[OpenAI]](https://chat.openai.com/) |
| **Codex** | The Codex models are descendants of our GPT-3 models that can understand and generate code. Their training data contains both natural language and billions of lines of public code from GitHub  | [[Github]](https://platform.openai.com/docs/models/codex) |
| **Bloom** | BigScience Large Open-science Open-access Multilingual Language Model  | [[HuggingFace]](https://huggingface.co/bigscience/bloom) |
| **Facebook LLM** | OPT-175B is a GPT-3 equivalent model trained by Meta. It is by far the largest pretrained language model available with 175 billion parameters.  | [[Alpa]](https://opt.alpa.ai/) |
| **GPT-NeoX** | GPT-NeoX-20B, a 20 billion parameter autoregressive language model trained on the Pile  | [[HuggingFace]](https://huggingface.co/docs/transformers/model_doc/gpt_neox) |
| **FLAN-T5 XXL** | Flan-T5 is an instruction-tuned model, meaning that it exhibits zero-shot-like behavior when given instructions as part of the prompt.  | [[HuggingFace/Google]](https://huggingface.co/google/flan-t5-xxl) |
| **XLM-RoBERTa-XL** | XLM-RoBERTa-XL model pre-trained on 2.5TB of filtered CommonCrawl data containing 100 languages.  | [[HuggingFace]](https://huggingface.co/facebook/xlm-roberta-xxl) |
| **GPT-J** | It is a GPT-2-like causal language model trained on the Pile dataset  | [[HuggingFace]](https://huggingface.co/docs/transformers/model_doc/gptj) |
| **PaLM-rlhf-pytorch** | Implementation of RLHF (Reinforcement Learning with Human Feedback) on top of the PaLM architecture. Basically ChatGPT but with PaLM | [[Github]](https://github.com/lucidrains/PaLM-rlhf-pytorch)
| **GPT-Neo** | An implementation of model parallel GPT-2 and GPT-3-style models using the mesh-tensorflow library. | [[Github]](https://github.com/EleutherAI/gpt-neo) |
| **LaMDA-rlhf-pytorch** | Open-source pre-training implementation of Google's LaMDA in PyTorch. Adding RLHF similar to ChatGPT. | [[Github]](https://github.com/conceptofmind/LaMDA-rlhf-pytorch) |
| **RLHF** | Implementation of Reinforcement Learning from Human Feedback (RLHF) | [[Github]](https://github.com/xrsrke/instructGOOSE) |
| **GLM-130B** | GLM-130B: An Open Bilingual Pre-Trained Model | [[Github]](https://github.com/THUDM/GLM-130B) |


## Tutorials
📚

  - **Introduction to Prompt Engineering**

    - [Prompt Engineering 101 - Introduction and resources](https://www.linkedin.com/pulse/prompt-engineering-101-introduction-resources-amatriain)
    - [Prompt Engineering 101](https://humanloop.com/blog/prompt-engineering-101)

  - **Beginner's Guide to Generative Language Models**

    - [A beginner-friendly guide to generative language models - LaMBDA guide](https://aitestkitchen.withgoogle.com/how-lamda-works)
    - [Generative AI with Cohere: Part 1 - Model Prompting](https://txt.cohere.ai/generative-ai-part-1)

  - **Best Practices for Prompt Engineering**

    - [Best practices for prompt engineering with OpenAI API](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)
    - [How to write good prompts](https://andymatuschak.org/prompts)

  - **Complete Guide to Prompt Engineering**

    - [A Complete Introduction to Prompt Engineering for Large Language Models](https://www.mihaileric.com/posts/a-complete-introduction-to-prompt-engineering)
    - [Prompt Engineering Guide: How to Engineer the Perfect Prompts](https://richardbatt.co.uk/prompt-engineering-guide-how-to-engineer-the-perfect-prompts)

  - **Technical Aspects of Prompt Engineering**

    - [3 Principles for prompt engineering with GPT-3](https://www.linkedin.com/pulse/3-principles-prompt-engineering-gpt-3-ben-whately)
    - [A Generic Framework for ChatGPT Prompt Engineering](https://medium.com/@thorbjoern.heise/a-generic-framework-for-chatgpt-prompt-engineering-7097f6513a0b)
    - [Methods of prompt programming](https://generative.ink/posts/methods-of-prompt-programming)

  - **Resources for Prompt Engineering**

    - [Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts)
    - [Best 100+ Stable Diffusion Prompts](https://mpost.io/best-100-stable-diffusion-prompts-the-most-beautiful-ai-text-to-image-prompts)
    - [DALLE Prompt Book](https://dallery.gallery/the-dalle-2-prompt-book)
    - [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
    - [Prompt Engineering by Microsoft](https://microsoft.github.io/prompt-engineering)

## Videos
🎥

- [Advanced ChatGPT Prompt Engineering](https://www.youtube.com/watch?v=bBiTR_1sEmI)
- [ChatGPT: 5 Prompt Engineering Secrets For Beginners](https://www.youtube.com/watch?v=2zg3V66-Fzs)
- [CMU Advanced NLP 2022: Prompting](https://youtube.com/watch?v=5ef83Wljm-M&feature=shares)
- [Prompt Engineering - A new profession ?](https://www.youtube.com/watch?v=w102J3_9Bcs&ab_channel=PatrickDebois)
- [ChatGPT Guide: 10x Your Results with Better Prompts](https://www.youtube.com/watch?v=os-JX1ZQwIA)
- [Language Models and Prompt Engineering: Systematic Survey of Prompting Methods in NLP](https://youtube.com/watch?v=OsbUfL8w-mo&feature=shares)
- [Prompt Engineering 101: Autocomplete, Zero-shot, One-shot, and Few-shot prompting](https://youtube.com/watch?v=v2gD8BHOaX4&feature=shares)


## Communities
🤝

- [OpenAI Discord](https://discord.com/invite/openai)
- [PromptsLab Discord](https://discord.gg/m88xfYMbK6)
- [Learn Prompting](https://discord.gg/7enStJXQzD)
- [r/ChatGPT Discord](https://discord.com/invite/r-chatgpt-1050422060352024636)
- [MidJourney Discord](https://discord.com/invite/MidJourney)


<h6 align="center">
<small><small>Image Source: docs.cohere.ai </small> </small>
</h6>
