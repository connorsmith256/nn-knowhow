Legend:
- 📜: papers
- 📰: blog posts, project pages
- 📖: books
- 🌐: broad/general resources
- 🧪: code, experiments
- 📺: videos

06/21/2024
- 📜[Connecting the Dots: LLMs can Infer and Verbalize Latent Structure from Disparate Training Data](https://arxiv.org/pdf/2406.14546)
    - inductive out-of-context reasoning (OOCR): infer value of latent information during (finetuning) training. High variance, but better than ICL. GPT-4 did better than GPT-3.5
- 📰[Attention Output SAEs Improve Circuit Analysis](https://www.alignmentforum.org/posts/EGvtgB7ctifzxZg6v/attention-output-saes-improve-circuit-analysis)
    - Trained SAEs on every layer of GPT-2 Small, did interp on IOI circuit, built [Circuit Explorer](https://d483a8995f67545839dbee56b8c864fca.clg07azjl.paperspacegradient.com/) tool for recursive DFA. Acknowledge SAEs are still unreliable

06/20/2024
- 📜[Safety Cases: How to Justify the Safety of Advanced AI Systems](https://arxiv.org/pdf/2403.10462)
    - Proposes framework for decomposing complex AI systems & indentifying arguments for reaching acceptably low risk in ability, control, trustworthiness, & deference
- 📰[UK AI SI Inspect](https://ukgovernmentbeis.github.io/inspect_ai/)
    - OSS evals framework. Flexible, supports popular providers
- 🧪[Experimented with Inspect](./projects/inspect-experiment/)
    - submitted [bug fix](https://github.com/UKGovernmentBEIS/inspect_ai/pull/58)
- 📖[AI Safety Book (Hendrycks), Chapters 6.1-6.10](https://drive.google.com/file/d/1JN7-ZGx9KLqRJ94rOQVwRSa7FPZGl2OY/view)
    - covered machine ethics & a variety of approaches to align AI to human values, preferences, utility

06/19/2024
- 🧪[Experimented with using control vectors to steer Llama 3 8B](./projects/repengy/repengy.ipynb)
    - only succeeded with very simple controls (e.g. all-caps). Suspect this scales better with larger models

06/18/2024
- 📺[METR's Talk on Evaluations Research - Beth Barnes, Daniel Ziegler, Ted Suzman](https://www.youtube.com/watch?v=KO72xvYAP-w)
- 📖[Foundational Challenges in Assuring Alignment and Safety of Large Language Models](https://llm-safety-challenges.github.io/challenges_llms.pdf)
- 📜[Pretraining Language Models with Human Preferences](https://arxiv.org/pdf/2302.08582)
- 📜[KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/pdf/2402.01306)
- 📜[Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought](https://arxiv.org/pdf/2403.05518)
- 📜[Sycophancy To Subterfuge: Investigating Reward Tampering In Language Models](https://arxiv.org/pdf/2406.10162)

06/17/2024
- 📜[Supervising strong learners by amplifying weak experts](https://arxiv.org/pdf/1810.08575)
- 📜[Scalable agent alignment via reward modeling: a research direction](https://arxiv.org/pdf/1811.07871)
- 📜[AI safety via debate](https://arxiv.org/pdf/1805.00899)
- 📜[Self-critiquing models for assisting human evaluators](https://arxiv.org/pdf/2206.05802)
- 📰[Specification gaming: the flip side of AI ingenuity](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)
- 📜[Goal Misgeneralization in Deep Reinforcement Learning](https://arxiv.org/pdf/2105.14111)
- 📜[Eliciting latent knowledge: How to tell if your eyes deceive you](https://docs.google.com/document/d/1WwsnJQstPq91_Yh-Ch2XRL8H_EpsnjrC1dwZXR37PC8/)
- 📰[Discussion: Challenges with Unsupervised LLM Knowledge Discovery](https://www.lesswrong.com/posts/wtfvbsYjNHYYBmT3k/discussion-challenges-with-unsupervised-llm-knowledge-1)
- 📜[Eliciting Latent Knowledge from “Quirky” Language Models](https://arxiv.org/pdf/2312.01037)

06/16/2024
- 📜[Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report](https://arxiv.org/pdf/2406.07394)
- 📜[Improve Mathematical Reasoning in Language Models by Automated Process Supervision](https://arxiv.org/pdf/2406.06592)
- 📜[Scalable MatMul-free Language Modeling](https://arxiv.org/pdf/2406.02528)

06/13/2024
- 📜[Training Language Models with Language Feedback at Scale](https://arxiv.org/pdf/2303.16755)
- 📜[Cooperative Inverse Reinforcement Learning](https://arxiv.org/pdf/1606.03137)
- 📜[Representation Engineering: A Top-Down Approach To Ai Transparency](https://arxiv.org/pdf/2310.01405)
- 📜[Improving Alignment and Robustness with Circuit Breakers](https://arxiv.org/pdf/2406.04313)

06/11/2024
- 📜[Algorithms for Inverse Reinforcement Learning](https://people.eecs.berkeley.edu/~russell/papers/ml00-irl.pdf)
- 📜[Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization](https://arxiv.org/pdf/1603.00448)
- 📜[Deep Reinforcement Learning from Human Preferences](https://arxiv.org/pdf/1706.03741)
- 📜[Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155)
- 📜[Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/pdf/2204.05862)
- 📜[Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/pdf/2212.08073)
- 📜[RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/pdf/2309.00267)
- 📜[Direct Nash Optimization: Teaching Language Models to Self-Improve with General Preferences](https://arxiv.org/pdf/2404.03715)

06/10/2024
- 📖[Reinforcement Learning: An Introduction (Sutton & Barto), Chapters 1-3, 13](http://www.incompleteideas.net/book/RLbook2020.pdf)
- 📜[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347)

06/07/2024
- 📜[Sigma-GPTs: A New Approach to Autoregressive Models](https://arxiv.org/pdf/2404.09562)
- 📰[Situational Awareness (Aschenbrenner)](https://situational-awareness.ai/)

06/06/2024
- 📜[Adaptive Mixtures of Local Experts](https://people.engr.tamu.edu/rgutier/web_courses/cpsc636_s10/jacobs1991moe.pdf)
- 📜[Hierarchical mixtures of experts and the EM algorithm](https://www.cs.toronto.edu/~hinton/absps/hme.pdf)
- 📜[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-Of-Experts Layer](https://arxiv.org/pdf/1701.06538)
- 📜[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961)
- 📜[Unified Scaling Laws For Routed Language Models](https://arxiv.org/pdf/2202.01169)
- 📜[Measuring the Effects of Data Parallelism on Neural Network Training](https://arxiv.org/pdf/1811.03600)
- 📜[An Empirical Model of Large-Batch Training](https://arxiv.org/pdf/1812.06162)
- 📜[The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/pdf/1803.03635)

06/05/2024
- 📜[LLMs Can’t Plan, But Can Help Planning in LLM-Modulo Frameworks](https://arxiv.org/pdf/2402.01817)
- 🌐[EECS 498-007 Deep Learning for Computer Vision (Lectures 19-21)](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2019/schedule.html)
- 📖[AI Safety Book (Hendrycks), Chapters 5.1-5.5](https://drive.google.com/file/d/1JN7-ZGx9KLqRJ94rOQVwRSa7FPZGl2OY/view)

06/04/2024
- 🧪[Found interpretable features in SAE](./projects/sae)

06/03/2024
- 🧪[Trained a SAE on GPT2-small](./projects/sae)

06/02/2024
- 📰[Apollo Research 1-year update](https://www.alignmentforum.org/posts/qK79p9xMxNaKLPuog/apollo-research-1-year-update)
- 📜[Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning](https://arxiv.org/pdf/2405.12241)
- 📜[The Local Interaction Basis: Identifying Computationally-Relevant and Sparsely Interacting Features in Neural Networks](https://arxiv.org/pdf/2405.10928)
- 📰[Sparsify: A mechanistic interpretability research agenda](https://www.alignmentforum.org/posts/64MizJXzyvrYpeKqm/sparsify-a-mechanistic-interpretability-research-agenda)
- 📜[Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control](https://arxiv.org/pdf/2405.08366)

05/31/2024
- 📰[The Engineer’s Interpretability Sequence](https://www.alignmentforum.org/s/a6ne2ve5uturEEQK7)
- 📜[Robustness May Be at Odds with Accuracy](https://arxiv.org/pdf/1805.12152)
- 📜[Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/pdf/1905.02175)
- 🧪[ARENA 3.0, Chapter 1.4](https://arena3-chapter1-transformer-interp.streamlit.app/[1.4]_Superposition_&_SAEs)

05/30/2024
- 🧪[ARENA 3.0, Chapter 1.3](https://arena3-chapter1-transformer-interp.streamlit.app/[1.3]_Indirect_Object_Identification)

05/29/2024
- 🧪[ARENA 3.0, Chapter 1.2](https://arena3-chapter1-transformer-interp.streamlit.app/[1.2]_Intro_to_Mech_Interp)

05/28/2024
- 📜[Towards Guaranteed Safe AI: A Framework for Ensuring Robust and Reliable AI Systems](https://arxiv.org/pdf/2405.06624)
- 📜[Suppressing Pink Elephants with Direct Principle Feedback](https://arxiv.org/pdf/2402.07896)

05/27/2024
- 📜[Circuit Component Reuse Across Tasks In Transformer Language Models](https://arxiv.org/pdf/2310.08744)
- 📜[How to use and interpret activation patching](https://arxiv.org/pdf/2404.15255)
- 📜[Towards Automated Circuit Discovery for Mechanistic Interpretability](https://arxiv.org/pdf/2304.14997)
- 📰[Attribution Patching: Activation Patching At Industrial Scale](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)
- 📜[Causal Scrubbing: a method for rigorously testing interpretability hypotheses](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing)
- 📜[Toward Transparent AI: A Survey on Interpreting the Inner Structures of Deep Neural Networks](https://arxiv.org/pdf/2207.13243)

05/24/2024
- 📜[Grokking: Generalization Beyond Overfitting On Small Algorithmic Datasets](https://arxiv.org/pdf/2201.02177)
- 📰[A Mechanistic Interpretability Analysis of Grokking](https://www.alignmentforum.org/posts/N6WM6hs7RQMKDhYjB/a-mechanistic-interpretability-analysis-of-grokking)
- 📜[Scaling Laws and Interpretability of Learning from Repeated Data](https://arxiv.org/pdf/2205.10487)
- 📜[Deep Double Descent: Where Bigger Models And More Data Hurt](https://arxiv.org/pdf/1912.02292)
- 📜[Interpretability In The Wild: A Circuit For Indirect Object Identification In Gpt-2 Small](https://arxiv.org/pdf/2211.00593)

05/23/2024
- 🧪[ARENA 3.0, Chapter 1.1](https://arena3-chapter1-transformer-interp.streamlit.app/[1.1]_Transformer_from_Scratch)

05/22/2024
- 📖[AI Safety Book (Hendrycks), Chapters 3.4-4.9](https://drive.google.com/file/d/1JN7-ZGx9KLqRJ94rOQVwRSa7FPZGl2OY/view)
- 📜[Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)

05/20/2024
- 🧪[Finetuned GPT-2 models with custom dataset](./projects/bomgen/gpt2-finetune/)

05/17/2024
- 🧪[Created Resumable to support suspending/resuming training mid-epoch](./projects/resumable/resumable.py)

05/15/2024
- 🧪[My first neural style transfer](./projects/style-transfer/)
- 🌐[EECS 498-007 Deep Learning for Computer Vision (Lectures 16-18)](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2019/schedule.html)

05/14/2024
- 📰[Effort, a possibly new algorithm for LLM inference](https://kolinko.github.io/effort/index.html)
- 📰[Introduction to Weight Quantization](https://towardsdatascience.com/introduction-to-weight-quantization-2494701b9c0c)
- 📜[LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/pdf/2208.07339)
- 📜[GPTQ (OPTQ): Accurate Post-training Quantization For Generative Pre-trained Transformers](https://arxiv.org/pdf/2210.17323)
- 📰[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- 📜[QuIP: 2-Bit Quantization of Large Language Models With Guarantees](https://arxiv.org/pdf/2307.13304)
- 📜[QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/pdf/2402.04396)
- 🌐[EECS 498-007 Deep Learning for Computer Vision (Lectures 14-15)](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2019/schedule.html)
- 🧪[Set up repo & research log](./research-log.md)

05/10/2024
- 📜[Receptance Weighted Key Value (RWKV)](https://arxiv.org/pdf/2305.13048)
- 🌐[Ilya's 30u30 Deep Learning](https://arc.net/folder/D0472A20-9C20-4D3F-B145-D2865C0A9FEE)

05/09/2024
- 📰[HuggingFace NLP Course](https://huggingface.co/learn/nlp-course)
- 📖[Alice's Adventures In a Differentiable Wonderland](https://arxiv.org/pdf/2404.17625)
- 📜[An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale](https://arxiv.org/pdf/2010.11929)
- 📰[A Visual Guide to Vision Transformers ](https://blog.mdturp.ch/posts/2024-04-05-visual_guide_to_vision_transformer.html)

05/08/2024
- 📜[xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)

05/06/2024
- 📰[Inducing Unprompted Misalignment in LLMs](https://www.lesswrong.com/posts/ukTLGe5CQq9w8FMne/inducing-unprompted-misalignment-in-llms)
- 📰[Simple probes can catch sleeper agents](https://www.anthropic.com/research/probes-catch-sleeper-agents)
- 📜[The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions](https://arxiv.org/abs/2404.13208)
- 📜[Efficiently Modeling Long Sequences with Structured State Spaces (S4)](https://arxiv.org/pdf/2111.00396)
- 📜[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/pdf/2312.00752)
- 📜[Transformer Circuits In-Context Learning & Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- 📜[Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/)

05/05/2024
- 📰[Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html)
- 📜[Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)](https://arxiv.org/pdf/2305.18290)
- 📜[Direct Preference Optimization with an Offset (ODPO)](https://arxiv.org/pdf/2402.10571)

05/03/2024
- 📜[Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/pdf/1708.04552)
- 📜[AutoAugment: Learning Augmentation Strategies from Data](https://arxiv.org/pdf/1805.09501)
- 📜[RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/pdf/1909.13719)
- 📜[Augmix: A Simple Data Processing Method To Improve Robustness And Uncertainty](https://arxiv.org/pdf/1912.02781)
- 📜[TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation](https://arxiv.org/pdf/2103.10158)
- 🧪[My first ResNet (LR search, schedulers, data augmentation)](./projects/cnns/imagenet/)

05/02/2024
- 📜[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677)
- 🧪[My first ResNet (optimizer search)](./projects/cnns/imagenet/)

04/29/2024
- 📰[xFormers](https://github.com/facebookresearch/xformers)
- 📜[SDXS: Real-Time One-Step Latent Diffusion Models with Image Conditions](https://arxiv.org/pdf/2403.16627)
- 📜[Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis](https://arxiv.org/pdf/2404.13686)

04/25/2024
- 🧪Experiments with local models: oobabooga & a1111
- 📜[OpenELM: An Efficient Language Model Family with Open Training and Inference Framework](https://arxiv.org/pdf/2404.14619)
- 📜[Phi-3 Technical Report: A Highly Capable Language Modle Locally on Your Phone](https://arxiv.org/pdf/2404.14219)

04/24/2024
- 🧪[My first ResNet (training on CIFAR10)](./projects/cnns/imagenet/)

04/23/2024
- 🧪[My first ResNet (training on imagenette)](./projects/cnns/imagenet/)

04/22/2024
- 🧪[My first ResNet](./projects/cnns/imagenet/)

04/21/2024
- 🧪[My first CNN](./projects/cnns/lenet/)
- 📜[Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842)
- 📜[Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/pdf/1512.03385.pdf)
- 📜[Neural Architecture Search with Reinforcement Learning](https://arxiv.org/pdf/1611.01578)
- 📜[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/pdf/1707.01083)
- 📜[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861)
- 📜[Gaussian Error Linear Units (GELUs)](https://arxiv.org/pdf/1606.08415)
- 📜[GELU Activation Function in Deep Learning: A Comprehensive Mathematical Analysis and Performance](https://arxiv.org/pdf/2305.12073)
- 🌐[EECS 498-007 Deep Learning for Computer Vision (Lectures 9-13)](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2019/schedule.html)

04/20/2024
- 🌐[EECS 498-007 Deep Learning for Computer Vision (Lectures 1-8)](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2019/schedule.html)

04/19/2024
- 📜[A Baseline For Detecting Misclassified And Out-of-distribution Examples In Neural Networks](https://arxiv.org/pdf/1610.02136)
- 📜[Benchmarking Neural Network Robustness To Common Corruptions And Perturbations](https://arxiv.org/pdf/1903.12261)
- 📜[Natural Adversarial Examples](https://arxiv.org/pdf/1907.07174)
- 📜[The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization](https://arxiv.org/pdf/2006.16241)

04/17/2024
- 📜[Testing Robustness Against Unforeseen Adversaries](https://arxiv.org/pdf/1908.08016)
- 📜[HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refus](https://arxiv.org/pdf/2402.04249)
- 📜[Aligning Ai With Shared Human Values](https://arxiv.org/pdf/2008.02275)
- 📖[AI Safety Book (Hendrycks), Chapters 1-3.3](https://drive.google.com/file/d/1JN7-ZGx9KLqRJ94rOQVwRSa7FPZGl2OY/view)
- 📰[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- 📜[Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/pdf/2203.15556.pdf)

04/15/2024
- 📜[WaveNet: A Generative Model for Raw Audio (CNN)](https://arxiv.org/pdf/1609.03499)
- 📜[Attention Is All You Need](https://arxiv.org/pdf/1706.0376)
- 📰[Yes You Should Understand Backprop (Karpathy)](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)
- 📺[Let's build GPT: from scratch, in code, spelled out (Karpathy)](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- 📰[tiktoken](https://github.com/openai/tiktoken)
- 🧪[My first GPT](./projects/bomgen/my_gpt)

04/14/2024
- 📜[Recurrent Neural Network Based Language Model](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- 📜[Generating Sequences With Recurrent Neural Networks (LSTM)](https://arxiv.org/pdf/1308.0850)
- 📜[On the Properties of Neural Machine Translation: Encoder–Decoder Approaches (GRU)](https://arxiv.org/pdf/1409.1259)
- 🧪[My first RNN](./projects/rnn/rnn-manual.py)
- 🧪[My first GRU](./projects/rnn/gru.py)
- 📰[Gemma PyTorch](https://github.com/google/gemma_pytorch)

04/13/2024
- 📜[SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774)

04/11/2024
- 🧪[My first FNN](./projects/fnn)

04/10/2024
- 🌐[NN Zero to Hero (Karpathy)](https://github.com/karpathy/nn-zero-to-hero)
- 📜[A Neural Probabilistic Language Model (MLP)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- 📜[Adam: A Method For Stochastic Optimization](https://arxiv.org/pdf/1412.6980)
- 📜[On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745)

04/09/2024
- 🌐[makemore (Karpathy)](https://github.com/karpathy/makemore)

04/08/2024
- 📰[Faulty reward functions in the wild](https://openai.com/research/faulty-reward-functions)
- 📰[The Singular Value Decompositions of Transformer Weight Matrices are Highly Interpretable](https://www.lesswrong.com/posts/mkbGjzxD8d8XqKHzA/the-singular-value-decompositions-of-transformer-weight?ref=conjecture.dev)
- 📰[unRLHF - Efficiently Undoing LLM Safeguards](https://www.conjecture.dev/research/unrlhf-efficiently-undoing-llm-safeguards)
- 📜[Interpreting Neural Networks through the Polytope Lens](https://arxiv.org/pdf/2211.12312)
- 📜[Representational Strengths and Limitations of Transformers](https://arxiv.org/pdf/2306.02896)

04/02/2024
- 🌐[micrograd (Karpathy)](https://github.com/karpathy/micrograd)
- 📜[Layer Normalization](https://arxiv.org/pdf/1607.06450)

03/31/2024
- 📺[Implementing GPT-2 From Scratch (Nanda)](https://www.youtube.com/watch?v=dsjUDacBw8o&t=2611s)

03/28/2024
- 🌐[Transformers - A Comprehensive Mechanistic Interpretability Explainer & Glossary (Nanda)](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=pndoEIqJ6GPvC1yENQkEfZYR&q=encode)
- 📺[What is a Transformer? (Nanda)](https://www.youtube.com/watch?v=bOYE6E8JrtU)
- 📜[Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
- 📜[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)