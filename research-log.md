Legend:
- 📜: papers
- 📰: blog posts, project pages
- 📖: books
- 🌐: broad/general resources
- 🧪: code, experiments
- 📺: videos

07/25/2024
- 📰[AI achieves silver-medal standard solving International Mathematical Olympiad problems](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)
    - AlphaProof uses an LLM to translate math problems into formal language. The backend uses AlphaZero's algorithm (MCTS + self-play with synthetic data), which sounds very similar to the Lean-STaR paper from 07/22. Note for the headline performance, problems were manually translated to Lean, and LLM translation is still WIP: "the results showed great promise"
- 📜[Rule Based Rewards for Language Model Safety](https://cdn.openai.com/rule-based-rewards-for-language-model-safety.pdf)
    - Leverages LLMs' ability on specific classification tasks to evaluate whether a completion follows a behavior policy. A grader LLM estimates the probability a completion meets each proposition/combined class features defined in the behavior policy. The classification prompts fed into the grader are tuned from a small Gold dataset created by human researchers. The classification probabilities are fed into the Rule-Based Reward (RBR) model, a small linear model fitted against a synthetic dataset. The RBR score is combined with the helpful-only RM for the total reward used in PPO training. Including RBR in training led to fewer over-refusals on safe prompts while maintaining appropriate refusals and model performance
- 📜[Exploring Scaling Trends in LLM Robustness](https://far.ai/post/2024-07-robust-llm/paper.pdf)
    - Found AT effectiveness scaled with model size (on Pythia models from 14M to 12B), and importantly, larger models were more sample efficient. AT against one attack also transferred to other attacks

07/24/2024
- 📜[Defending Against Unforeseen Failure Modes with Latent Adversarial Training](https://arxiv.org/pdf/2403.05030)
    - LAT perturbs latent state instead of inputs, as in AT. The optimal layer to perturb is found empirically. Models were fine-tuned using poisoned data to insert trojans, then fine-tuned with clean data & the given technique. LAT pareto dominates AT in image classification, text classification, and text generation (7B model) for data forgetting, although it can entrench trojans sometimes, just like AT
- 📜[Targeted Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs](https://arxiv.org/pdf/2407.15549)
    - Applied LAT to multiple layers of Llama2-7B-chat and Llama3-8B-instruct, as an additional step beyond refusal training (RT). The LAT models maintained performance on benign requests and reduced attack success rates better than robust refusal dynamic defense (R2D2), while requiring 36x fewer GPU hours to fine-tune. DPO + LAT was also able to remove backdoor triggers with minimal impact on general performance. Adding LAT to unlearning methods also improved success with minimal impact to performance, although relearning unwanted knowledge remains trivial

07/23/2024
- 📜[The Alignment Problem from a Deep Learning Perspective](https://arxiv.org/pdf/2209.00626)
    - First published in Aug 2022. Covers popular ideas: reward misspecification + situational awareness from RLHF can lead to reward hacking, which can exacerbate misalignment. As systems become more generally capable, deception and power-seeking become more likely and risky, especially as we cede control to autonomous agents
- 📰[Thoughts on the impact of RLHF research](https://www.alignmentforum.org/posts/vwu4kegAEZTBtpT6p/thoughts-on-the-impact-of-rlhf-research)
    - Christiano makes the case that RLHF was a relatively simple alignment technique that gave the field much-needed empirical data, and more complicated techniques will share technical ingredients, so the development was a net positive. He thinks RLHF had a small marginal impact on timelines, avoiding RLHF would have introduced a capability overhang, and effective empirical safety work requires working with systems that are closer to posing a risk
- 📺[RLHF: How to Learn from Human Feedback with Reinforcement Learning](https://www.youtube.com/watch?v=56PlUikhB3o)
    - Good refresher:
        - RL will often over-exploit, so including KL-control in the loss prevents too much divergence from the base (or SFT) model (limits to infinite self-play)
        - human ratings are expensive, and RL is sample-hungry and unstable, so turning it into supervised RL with a reward model is much cheaper/efficient
        - offline RL (training the RM) leverages large amount of existing data & allows reusing existing supervised learning infrastructure
        - high-quality labels for the RM is a necessity
- 📜[WARM: On the Benefits of Weight Averaged Reward Models](https://arxiv.org/pdf/2401.12187)
    - Trains a single RM created from the average of RMs trained using different hyperparameters & starting from different SFT checkpoints. Linear interpolation of weights relies on "linear mode connectivity" of models with shared pre-training. Weight averaging improves reliability on OOD tests and is more robust than ensembling (possibly due to reduced memorization)
- 📜[Simple Synthetic Data Reduces Sycophancy In Large Language Models](https://arxiv.org/pdf/2308.03958)
    - By default, instruction tuning increases sycophancy, and larger models exhibit this trait more. Sycophancy can be modestly reduced by training on a synthetic dataset with examples disregarding user opinions, particularly those which the model knows are incorrect
- 📜[Compositional Preference Models For Aligning LMs](https://arxiv.org/pdf/2310.13011)
    - Decomposes a single preference score into distinct features, each of which gets a score from an LLM. Feature scores are re-aggregated using a logistic regression. CPMs were more robust to overoptimization and more preferred by another LLM than reference PMs

07/22/2024
- 📜[Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders](https://storage.googleapis.com/jumprelu-saes-paper/JumpReLU_SAEs.pdf)
    - Trained SAEs on Gemma 2 9B, using JumpReLU and an L0 penalty (both requiring pseudo-derivatives to train) to decrease false positives of activations and encourage sparsity. JumpReLU had a similar number of very-high-frequency (>10%) features to TopK (more than Gated), but fewer high-frequency (>1%) than TopK and Gated. All three architectures exhibit similar manual (human) and automated interpretability
- 📜[Truth is Universal: Robust Detection of Lies in LLMs](https://arxiv.org/pdf/2407.12831)
    - Previous research failed to find a single "truth" direction in LLM activation space that generalizes from affirmative statements to negations. They found a 2D subspace consisting of a general truth direction and a polarity-sensitive truth direction, which accounts for most of the model's sense of truth and can generalize to disjunctions and conjunctions. Activations projected onto these directions can be checked as a rudimentary lie detector, with strong accuracy for simple facts (though less robust for more complicated statements)
- 📜[The Platonic Representation Hypothesis](https://arxiv.org/pdf/2405.07987)
    - Mostly interesting from a philosophical perspective. Makes an argument for "convergent realism": just like human science, even though training data is biased/limited, models can capture "true" representations. Requiring multi-task performance (more general), increasing model capacity, and encouraging simplicity (either via explicit regularization or an implicit Occam's razor) are three hypotheses for why convergence would happen. Predictions for this view include scaling being sufficient (though not efficient), training data helping cross-modality performance (only up to the limit that the different modalities can share information), and reduced hallucination/bias.
- 📜[Lean-STaR: Learning to Interleave Thinking and Proving](https://arxiv.org/pdf/2407.10040)
    - Applies the idea behind STaR to theorem proving. Human-written proofs + retrospective thoughts from GPT-4 are used to start. The rationale is used to help predict subsequent tactics. Successful trajectories are added to the dataset (as in STaR) and used to finetune for the next iteration. This method outperformed SFT & expert iteration alone on multiple models. Unclear whether this would scale to bigger models, or if the initial thoughts from GPT-4 are enabling the improvement

07/18/2024
- 📜[Open-Ended Learning Leads to Generally Capable Agents](https://arxiv.org/pdf/2107.12808)
    - Defined a 3D "XLand" environment and used population based training (PBT) and generational training to improve agent fitness over time. Agents have a Goal Attention Module (GOAT) to structure, process, and attend to its goal. Agents achieved some amount of generalization on held-out tasks and finetuning for transfer
- 📰[A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)
    - Good overview of concepts, popular algorithms, and development over time. The reference to the "deadly triad" (bootstrapping, function approximation, off-policy training) leading to instability was helpful
- 📰[Stitching SAEs of different sizes](https://www.alignmentforum.org/posts/baJyjpktzmcmRfosq/stitching-saes-of-different-sizes)
    - Categorized features in larger vs smaller SAEs as "novel" vs "reconstruction" (sparsify) features. Mixing novel features from larger SAEs into smaller SAEs improved the performance of the smaller SAE. They also created "Frankenstein" SAEs by iteratively merging in novel features, achieving (slightly) better performance with smaller size
- 📰[SAEs (usually) Transfer Between Base and Chat Models](https://www.alignmentforum.org/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models)
    - Found that SAEs trained on base models perform well on chat models, and the gap can be closed further by fine-tuning the SAE. Seems to be further evidence that chat models' residual streams are very similar to base models
- 📰[An Introduction to Representation Engineering - an activation-based paradigm for controlling LLMs](https://www.alignmentforum.org/posts/3ghj8EuKzwD3MQR5G/an-introduction-to-representation-engineering-an-activation)
    - No new information, but a great summary of the approach
- 📜[Prover-Verifier Games Improve Legibility Of LLM Outputs](https://cdn.openai.com/prover-verifier-games-improve-legibility-of-llm-outputs/legibility.pdf)
    - OpenAI trained provers (helpful and sneaky) and verifiers to investigate whether we can train models to produce accurate outputs that are legible to humans. Joint training resulted in a model that was more accurate than initialization and still legibile. The sneaky prover's inserted errors became more subtle over time. A model trained only for correctness had the highest performance (and poor legibility), indicating a "legibility tax" tradeoff between accuracy and legibility

07/15/2024
- 📜[CRADLE: Empowering Foundation Agents Towards General Computer Control](https://arxiv.org/pdf/2403.03186)
    - Fed screenshot/low-FPS video into GPT-4o. Scaffolded with self-reflection on inpupts, inference to select next task, learned & stored skills (code to interact with a mouse + keyboard), and episodic & procedural memory for improving performance over time. This framework was able to perform a variety of tasks in games/software with >50% success
- 📜[STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning](https://arxiv.org/pdf/2203.14465)
    - Fine-tuned GPT-J-6B using STaR: starting with an initial few-shot prompt showing rationales, the model was trained to generate rationales to the input questions. Correct answer + rationale examples were added to the dataset. If the model wasn't able to come up with the right answer on its own, a hint was given in the form of the correct answer, and the model was able to generate a corresponding rationale. The process was iterated using the augmented dataset until performance plateaued. Performance after STaR was close to GPT-3 (30x larger), indicating models can "bootstrap" some amount of reasoning
- 📜[Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/pdf/2403.09629)
    - Generalizes the idea from STaR by having the model generate internal thoughts/rationales at each token position (in parallel), based off the preceding tokens. At the end of a thought, the post-rationale and base logits are mixed using a shallow MLP ("mixing head"). Rationales are optimized during training using REINFORCE, where the reward for a thought is based on how well it improves prediction of future tokens (the base future tokens are assumed to be ground-truth). Performance of Mistral 7B improved on GSM8K and CommonsenseQA, with "difficult" tokens benefiting more from internal reasoning. A future step could be dynamically determining when to generate/end thought, allowing a model to allocate variable compute during generation

07/12/2024
- 📖[Deep Learning (Goodfellow, Bengio, Courville), Chapter 8](https://www.deeplearningbook.org/)
    - Chapter 8, Optimization: local minima in high-dimensional space are unlikely to be far from the global minimum, but saddle points are common, incentivizing optimization algorithms that can escape locally small gradients. Gradient clipping can prevent taking too large a step off a cliff. Momentum overcomes poor conditioning of the Hessian by using the gradient to update the momentum/velocity rather than the weights directly. Interesting to see the recommendation on treating weight initialization as a hyperparameter, as more recent texts have not
- 📖[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
    - Brushed through quickly. The visual explanation for why neural nets can approximate any function was nice. Both the vanishing and exploding gradient problems in deep networks result from the gradient in early layers being the product of terms from many later layers, leading to instability

07/11/2024
- 📖[Deep Learning (Goodfellow, Bengio, Courville), Chapters 6-7](https://www.deeplearningbook.org/)
    - Chapter 6, Feedforward Networks: Historical coverage was really helpful, e.g. dominance of ReLU resulted from avoiding saturation & two-sided activations, cross-entropy improved over MSE's saturation/slow learning
    - Chapter 7, Regularization: Good coverage, esp. L2 as MAP Bayesian inference with a Gaussian prior on weights (discourages high weights) & L1 as the same but with a Laplace distribution prior (encourages sparsity). Also interesting to think of dropout as approximating bagging

07/10/2024
- 📖[Deep Learning (Goodfellow, Bengio, Courville), Chapters 1-5](https://www.deeplearningbook.org/)
    - Revisited/brushed up on foundations

07/09/2024
- 📜[Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs](https://arxiv.org/pdf/2407.04694)
    - New benchmark for measuring situational awareness, composed of self-knowledge (facts, causal influence, mechanistic introspection), inferences (training vs. deployment stages, self-recognition of text authorship), & actions (leveraging knowledge of identity, avoiding pattern imitation). No models are currently close to saturation, but scores were higher than I expected

07/08/2024
- 📜[On scalable oversight with weak LLMs judging strong LLMs](https://arxiv.org/pdf/2407.04622)
    - Compared debate to consultancy and direct question-answering for inference (not training). Debate outperforms consultancy, but QA with article access does significantly better than either. Obvious next steps are to train the debaters via self-play using the judge's decision as the reward signal
- 📜[Eureka: Human-Level Reward Design Via Coding Large Language Models](https://arxiv.org/pdf/2310.12931)
    - Given RL environment code, an LLM (GPT-4) generates candidate reward functions. Each are simulated, and the LLM is given detailed performance statistics as feedback to iteratively generate a new batch of reward functions. After several iterations, the final reward function often outperforms one defined by human experts
- 📜[DrEureka: Language Model Guided Sim-To-Real Transfer](https://arxiv.org/pdf/2406.01967)
    - Extends Eureka by generating reward functions that are (1) robust to domain randomization (DR) to account for real-world physics and (2) produce safer behavior. Feasible ranges on DR parameters to guide the LLM are learned via parallelized simulations. They achieved better than human-designed performance and got a robot dog to walk on a yoga ball for several minutes, without intermediate real-world testing
- 📰[An Extremely Opinionated Annotated List of My Favourite Mechanistic Interpretability Papers v2](https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite)
    - Great summary from Nanda on latest mechinterp research. Pleased to see most of the papers he mentions are linked in this research log

07/06/2024
- 📜[Understanding Generalization through Visualizations](https://arxiv.org/pdf/1906.03291)
    - Despite intuitions from the past, models do tend to generalize well to test data, perhaps because the loss landscape in high dimensions is mostly occupied by flat basins, leading to implicit regularization

07/05/2024
- 📜[Uncovering Latent Human Wellbeing In Language Model Embeddings](https://arxiv.org/pdf/2402.11777)
    - Used PCA to reduce dimensionality from embeddings and extract features relevant to wellbeing, using labeled prompts from ETHICS utilitarianism dataset. Small models represented wellbeing to an extent, bigger models did better
- 📜[When Representations Align: Universality in Representation Learning Dynamics](https://arxiv.org/pdf/2402.09142)
    - Makes a case that given smooth encoder/decoder maps, high expressivity (enough model parameters/complexity), and small initial weights, structured representations (as opposed to overfitting) minimizes loss and is a natural consequence of gradient descent. Analysis & empirical data are limited to simple datasets & ignore inductive biases of models

07/03/2024
- 🌐[Intro to ML Safety, Lectures 13-14](https://course.mlsafety.org/index.html)
    - Lecture 13, Trojans: data poisoning of public datasets works even when a small fraction is poisoned. Open-weight models can also be manipulated. Anomaly detection, Neural Cleanse, & meta-networks can be used to detect trojans, but not 100% of the time
    - Lecture 14, Detecting Emergent Behavior: many examples of unanticipated capabilities emerging from more params/compute. Emergent, instrumentally convergent goals (e.g. self-preservation) are concerning for safety. Proxy gaming has emerged many times, and can sometimes be detected by comparing to a trusted policy
    - Lecture 15, Honest Models: assessing a model's "beliefs" is hard. Older models did poorly on TruthfulQA, but this lecture appears outdated, since modern RLHF models perform significantly better

07/02/2024
- 📖[AI Safety Book (Hendrycks), Chapters 8.1-8.9](https://drive.google.com/file/d/1JN7-ZGx9KLqRJ94rOQVwRSa7FPZGl2OY/view)
    - Covered uncertainty around timelines/takeoff speeds, economic growth, distribution of AI power, distribution of access. Discussed tradeoffs of open access (allowing bottom-up misuse) vs. tight control (allowing top-down misuse/lock-in). Compute is a natural target for governance since it's physical and quantifiable. Government and international cooperation will become increasingly important as risk increases.
- 🌐[Intro to ML Safety, Lectures 1-12](https://course.mlsafety.org/index.html)
    - Lectures 1-9: mostly recap/covered in safety book
    - Lecture 10, Anomaly Detection: AUROC, AUPR, FPR95 can all be used to evaluate anomaly detection. (Negative) prediction confidence can be used for anomaly detection, but isn't robust to adversarial inputs. Outlier exposure can help detect unseen OOD examples/anomalies. Training on geometric transformations (rotation, translation) can also help
    - Lecture 11, Interpretable Uncertainty: modern NNs are miscalibrated (often overconfident), especially on OOD data. Temperature scaling (fixed, post-training) & ensembles can significantly reduce calibration error.
    - Lecture 12, Transparency: covered saliency maps & feature visualization. Good reminder that a lot of interpretability work (on transformers) is less than two years old!

07/01/2024
- 📖[AI Safety Book (Hendrycks), Chapters 7.1-7.7](https://drive.google.com/file/d/1JN7-ZGx9KLqRJ94rOQVwRSa7FPZGl2OY/view)
    - Covered game theory, cooperation/conflict, collective action problems, & evolutionary pressures. The most salient idea is maliciousness from AI systems is not even necessary for bad outcomes for humans; rationality and selection pressure is enough

06/28/2024
- 📜[Confidence Regulation Neurons in Language Models](https://arxiv.org/pdf/2406.16254)
    - LLMs have "entropy neurons" which can modify the output distribution without directly impacting the logits, and "frequency neurons," which directly modify logits along the direction of the token frequency distribution. Reproduced from GPT-2 small up to Llama 7B (entropy neurons) and Pythia 1B (frequency neurons)
- 📜[Interpreting Attention Layer Outputs with Sparse Autoencoders](https://arxiv.org/pdf/2406.17759)
    - Same as Attention Output SAEs Improve Circuit Analysis entry from 06/21
- 📜[LLM Critics Help Catch LLM Bugs](https://cdn.openai.com/llm-critics-help-catch-llm-bugs-paper.pdf)
    - OpenAI trained CriticGPT for scalable oversight. Started with GPT4, used RLHF pipeline: human-rated critiques of (question, answer) data, used to train a reward model, optimized a policy with PPO, with Force Sampling Beam Search (FSBS) to reduce rate of hallucinations/nitpicks. CriticGPT outperforms humans and ChatGPT

06/27/2024
- 🧪[Implemented PPO for GridWorld](./projects/rl_gym/ppo.py)
    - When the agent can find a solution, it converged much more quickly and stably than DQN. Still surpisingly fails to solve some seeds with bad obstacle configurations/sparser reward

06/26/2024
- 🧪[Implemented DQN for GridWorld](./projects/rl_gym/dql.py)
    - Holy instability Batman. Had to use L1 loss to get reasonable level of success. Sparse reward likely makes this harder for larger grids

06/25/2024
- 🧪[Implemented Q-Learning for a GridWorld](./projects/rl_gym/q-learning.py)
    - Surprisingly simple & effective, but this was a very simple task

06/21/2024
- 📜[Connecting the Dots: LLMs can Infer and Verbalize Latent Structure from Disparate Training Data](https://arxiv.org/pdf/2406.14546)
    - inductive out-of-context reasoning (OOCR): infer value of latent information during (finetuning) training. High variance, but better than ICL. GPT-4 did better than GPT-3.5
- 📰[Attention Output SAEs Improve Circuit Analysis](https://www.alignmentforum.org/posts/EGvtgB7ctifzxZg6v/attention-output-saes-improve-circuit-analysis)
    - Trained attention SAEs on every layer of GPT-2 Small, did interp on IOI circuit, built [Circuit Explorer](https://d483a8995f67545839dbee56b8c864fca.clg07azjl.paperspacegradient.com/) tool for recursive DFA. Acknowledge SAEs are still unreliable

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
    - since performance is sensitive to prompting, evals may be misleading. Tool use, CoT improve performance
    - METR's platform adopted by UKAISI
- 📖[Foundational Challenges in Assuring Alignment and Safety of Large Language Models](https://llm-safety-challenges.github.io/challenges_llms.pdf)
    - Notable: the nature of ICL is unknown, capabilities are not human-shaped, forecasting from scaling is hard, agents pose unique risks, finetuning is mostly superficial, evals are making progress but not robust enough, interp is a big challenge and can be misleading, jailbreaks are way too easy, data poisoning/backdoors are theoretically scary, it's not clear what value alignment means, capabilities are dual-use, governance is lagging capabilities
- 📜[Pretraining Language Models with Human Preferences](https://arxiv.org/pdf/2302.08582)
    - HF during pretraining might lead to better alignment, might not come at the cost of performance
- 📜[KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/pdf/2402.01306)
    - Generalizes approaches like DPO to HALOs (human-aware losses), proposes maximizing utility of generations instead of likelihood of preferences
- 📜[Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought](https://arxiv.org/pdf/2403.05518)
    - BCT may reduce bias such as sycophancy, post hoc rationalization, etc.
- 📜[Sycophancy To Subterfuge: Investigating Reward Tampering In Language Models](https://arxiv.org/pdf/2406.10162)
    - Started with curriculum of gameable environments, found models (rarely) learn to generalize to worse behavior e.g. reward tampering. Gaming remains even after attempting to train it away

06/17/2024
- 📜[Supervising strong learners by amplifying weak experts](https://arxiv.org/pdf/1810.08575)
    - Iterated amplification (HCH), proposed to progressively specify more complicated goals (eventual goal of iterating beyond human ability to evaluate)
- 📜[Scalable agent alignment via reward modeling: a research direction](https://arxiv.org/pdf/1811.07871)
    - Iterated amplification applied to scaffold agent alignment
- 📜[AI safety via debate](https://arxiv.org/pdf/1805.00899)
    - Idea to scale beyond human ability by having humans judge AI debates trained via self-play (assumes evaluating arguments is easier than generating them)
- 📜[Self-critiquing models for assisting human evaluators](https://arxiv.org/pdf/2206.05802)
    - Trained GPT-3 sized models to do self-critique (not as good as human critique)
- 📰[Specification gaming: the flip side of AI ingenuity](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)
    - Several examples of RL agents engaging in specification gaming
- 📜[Goal Misgeneralization in Deep Reinforcement Learning](https://arxiv.org/pdf/2105.14111)
    - Out-of-distribution generalization failures in maze solving, platformer, keys & chests games
- 📜[Eliciting latent knowledge: How to tell if your eyes deceive you](https://docs.google.com/document/d/1WwsnJQstPq91_Yh-Ch2XRL8H_EpsnjrC1dwZXR37PC8/)
    - Christiano's ELK proposal (Smart Vault)
- 📰[Discussion: Challenges with Unsupervised LLM Knowledge Discovery](https://www.lesswrong.com/posts/wtfvbsYjNHYYBmT3k/discussion-challenges-with-unsupervised-llm-knowledge-1)
    - Challenges with contrast-consistent search (CCS), related to ELK
- 📜[Eliciting Latent Knowledge from “Quirky” Language Models](https://arxiv.org/pdf/2312.01037)
    - Attempt by EleutherAI to do ELK

06/16/2024
- 📜[Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report](https://arxiv.org/pdf/2406.07394)
    - Using MCTS with Llama 3 8B to achieve (near) GPT-4 level performance on GSM8K & MATH
- 📜[Improve Mathematical Reasoning in Language Models by Automated Process Supervision](https://arxiv.org/pdf/2406.06592)
    - Using MCTS with Gemini Pro to achieve (near) GPT-4 level performance on MATH
- 📜[Scalable MatMul-free Language Modeling](https://arxiv.org/pdf/2406.02528)
    - Replaced matrix multiplications with ternary addition/negation. Implemented on FPGA

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