Legend:
- 📜: papers
- 📰: blog posts, project pages
- 📖: books
- 🌐: broad/general resources
- 🧪: code, experiments
- 📺: videos

09/25/2024
- 📜[LLMs Still Can't Plan; Can LRMs? A Preliminary Evaluation Of Openai's o1 On PlanBench](https://arxiv.org/pdf/2409.13373)
    - Refers to o1 as a "Large Reasoning Model", since its training and capabilities are noticeably different from other LLMs, and evaluates it on PlanBench. Previous models have struggled with Blocks world tasks on PlanBench, where solutions require 2-16 steps. Previous SOTA performance was 50-60% on Blocks world and below 5% on Mystery Blocks world, and one-shot prompting was not a strict improvement over zero-shot. o1-preview achieves 97.8% on Blocks world and 41.6%/52.8% (zero-/one-shot) on Mystery Blocks world, and is able to solve some problems with up to 28 steps, whereas other models struggle with solutions of only 5 steps. However, o1 still struggles to correctly identify when problems are unsolvable. o1 is also much slower and costs 10-100x more than other models (unpredictably, since the number of reasoning tokens cannot be explicitly controlled)

09/20/2024
- 📜[Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/pdf/2409.12917)
    - SFT on self-correction traces with rejection sampling (like STaR) amplifies the model's bias to not make any corrections/fails to improve its understanding of when to make modifications. The authors note a robust approach should 1) train on self-generated traces to avoid distribution shift during evaluation and 2) prevent a collapse to making only minor edits. SCoRe replaces conventional SFT with two stages: first, train for a model initialization that optimizes correction performance/avoids collapse by minimizing divergence from the base model, followed by online multi-turn RL with a substantial bonus for improving from the first to second response/penalty for a worse second response. The RL still uses a policy gradient and KL-divergence against a fixed model. Gemini 1.5 Flash improved on MATH (from 52.6% to 64.4%) and MBPP-R (from 47.3% to 60.6%), beating previous methods and making significantly more true and fewer false corrections. Performance can be improved further via test-time compute by generating several samples in parallel and self-correcting on each before performing majority voting

09/19/2024
- 📜[NVLM: Open Frontier-Class Multimodal LLMs](https://arxiv.org/pdf/2409.11402v1)
    - Introduces NVIDIA Vision Language Model, using Qwen2-72B-Instruct as the LLM backbone and InternViT-6B-448px-V1-5 as the ViT. Images are broken into 1-6 448px tiles, along with a scaled-down thumbnail for global context, encoded separately, and downsampled via pixel shuffle. They test three architectures: decoder-only (a two-layer MLP to project image tokens into the LLM embedding space), cross-attention between image and text tokens, and a hybrid (thumbnail tokens are processed alongside text tokens via self-attention, and high-res tiles are processed via cross-attention). Multimodal SFT usually degrades text-only performance, but this can be avoided with including a high-quality text-only dataset during SFT. Training was composed of two stages: during pretraining the LLM backbone & vision encoder are frozen and only the projector/cross-attention layers are trained, then during finetuning the vision encoder is frozen and the LLM and alignment modules are jointly trained. High-quality data is necessary to achieve strong multimodal performance. The decoder-only model generally performs the best and is less complex, but the longer sequence length requires more training and inference compute. The cross-attention architecture still has strong performance and is faster
- 📜[Data curation via joint example selection further accelerates multimodal learning](https://arxiv.org/pdf/2406.17711v1)
    - Introduces joint example selection (JEST), using contrastive learning to select the most learnable sub-batches from super-batches. Batches are scored by comparing the difference in loss between the learner and a pretrained reference model. Each sub-batch is split into low and high resolution, which also saves compute with minimal impact on performance. Scoring incurs a 10% overhead in FLOPs, but allows the learner to achieve SOTA performance with 13x fewer examples and 10x fewer FLOPs

09/18/2024
- 📜[A Comprehensive Evaluation of Quantized Instruction-Tuned Large Language Models: An Experimental Analysis up to 405B](https://arxiv.org/pdf/2409.11055)
    - Evaluates quantization of Vicuna, Gemma 1, and Llama 2 & 3 family of models on more recent benchmarks (MATH, MuSR, IFEval, GPQA, MMLU-PRO) using GPTQ, AWQ, SmoothQuant, and FP8. The authors confirm quantized models generally outperform full-precision smaller models (with exceptions for hallucination & instruction following). They find weight-only quants (GPTQ, AWQ) preserve accuracy better, especially for very large models (Llama 3.1 405B). Degradation from quantization does not significantly differ on harder evals

09/17/2024
- 📜[ALOHA Unleashed: A Simple Recipe for Robot Dexterity](https://aloha-unleashed.github.io/assets/aloha_unleashed.pdf)
    - Dextrous manipulation is hard to model, and past attempts at imitation learning for robots has been limited to non-dextrous tasks. The authors gather 26,000 teleoperated demonstrations across 5 tasks + 2,000 demonstrations on 3 simulated tasks and a transformer architecture which takes ResNet feature maps from multiple camera views and proprioception state. An L1 loss is insufficient to achieve high performance, but a diffusion policy with action chunking does better (40-95% success across tasks). Performance slightly generalizes to states not seen during training, but there are still edge cases where the policy fails to recover
- 📜[DemoStart: Demonstration-led auto-curriculum applied to sim-to-real with multi-fingered robots](https://arxiv.org/pdf/2409.06613)
    - DemoStart is an auto-curriculum RL method bootstrapped from a few (2-60) demonstrations in simulation with the goal of zero-shot sim-to-real transfer. Only task parameters which have a non-zero success and failure rate are used to train (called Zero-Variance Filtering, or ZVF), and training is biased toward states which occur earlier in demonstrations, to avoid focusing on what has already been learned. DemoStart is implemented with a distributed actor-learner setup, where the policy is updated via MPO (maximum a posteriori policy optimisation). After a teacher policy is learned, it is distilled into a student policy with visual observations using behavior cloning. Random force perturbations, physical constants, and camera poses/lighting/colors are used as DR. There is still a significant drop in success when transferring to real environments on harder tasks, but is a significant improvement over the compared SAC-X method
- 📜[H-ARC: A Robust Estimate of Human Performance on the Abstraction and Reasoning Corpus Benchmark](https://arxiv.org/pdf/2409.01374)
    - Provides a new estimate of human performance on ARC, using participants from Mechanical Turk. Average human performance on the training and test sets were 76.2% and 64.2%, after 3 attempts, compared to two-shot performance on Claude 3.5 Sonnet (19.3%) and GPT-4o with few-shot prompting (38.5%). LLMs make fewer systematic errors e.g. relating to grid dimension, and LLMs and humans share similar edit distances to correct solutions, but overall LLMs make different errors than humans, and humans are able to self-correct at a higher rate

09/16/2024
- 📜[Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems](https://arxiv.org/pdf/2408.16293)
    - Explores how to get an LLM (modified GPT-2) to self-correct using synthetic data consisting of elementary-school level math problems. Models often internally "know" they've made a mistake (determined via probing), so one technique is to regenerate a sentence if a mistake is detected, which is slightly more effective than vanilla beam search, at the cost of inference complexity & compute. Another approach is to introduce retries into the training data. When holding total tokens constant, introducing self-corrected mistakes (up to 50% of generated sentences) results in a significant performance boost and does not interfere with the model's ability to generate correct results. The results hold even when some of the corrections are fake/incorrect. Attempts to teach this ability with a LoRa finetune fail
- 📜[Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/pdf/2408.03314)
    - Investigates scaling test-time compute among several strategies (best-of-N, beam search, lookahead search) & optimal tradeoffs between train- and test-time compute on MATH (using PaLM-2). The optimal strategy depends on the difficulty of the problem (as predicted by the model). Beam search is most effective on harder problems and at lower compute budgets, and best-of-N is most effective on easier problems and higher budgets. For a given level of compute, lookahead search underperformed. On easier problems within reach of a model's capabilities, test-time compute can be more effective than a larger model, but harder problems still require more pretraining. They also investigate finetuning revision models for self-correction & compare parallel sampling vs. sequential revision, finding the ideal ratio depends on compute budget & question difficulty (purely sequential was better for easier questions, with a mix for harder questions)
- 📜[Synthetic Continued Pretraining](https://arxiv.org/pdf/2409.07431)
    - Introduces EntiGraph, an algorithm using an LLM (GPT-4) that transforms a small corpus into a larger corpus for continued pre-training (CPT). The LLM extracts a list of entities from the given document, generates a description for each entity, and analyzes relations among entities. Applying EntiGraph to the QuALITY dataset transforms 1.3M tokens into 600M synthetic tokens, which were then used for CPT on Llama 3 8B Base over 2 epochs. Closed book accuracy on questions related to the dataset increased from 39.49% to 56.42%, exceeding GPT-4's accuracy. Open book accuracy (using RAG) increased from 60.35% to 62.73%, showing EntiGraph provided 80% of the absolute performance improvement of RAG and was complementary to RAG. The authors note this approach relies on a powerful augmentation model and does not enable bootstrapping

09/13/2024
- 📜[Sapiens: Foundation for Human Vision Models](https://arxiv.org/pdf/2408.12569)
    - Introduces Sapiens, a family of vision transformers for human-centric vision tasks (2D pose estimation, body-part segmentation, depth estimation, surface normal prediction). Uses a proprietary dataset of 1B images at 1024px resolution. Models are pretrained using a masked auto-encoder (MAE), which excludes a random subset of patches. Models range in size from 300M to 2B, with the larger models achieving substantial improvement over previous SOTA
- 📜[Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming](https://arxiv.org/pdf/2408.16725)
    - Mini-Omni is a speech-to-speech model (not A-T-T-A as became popular first). A Whisper encoder transforms the audio input into tokens, and the model generates audio and text tokens simultaneously, allowing the output speech to be conditioned on text. Training was done in 3 stages: 1) modality alignment, where the core model is frozen and the ASR + TTS adapters learn to understand & generate speech; 2) adaption training, where the adapters are frozen and the core model learns to generate text from audio input; 3) multi-modal finetuning, where the entire model is unfrozen. The core model is Qwen2-0.5B, so while its understanding of speech is powerful, the core model is weak
- 📜[Language Agents Achieve Superhuman Synthesis Of Scientific Knowledge](https://storage.googleapis.com/fh-public/paperqa/Language_Agents_Science.pdf)
    - Introduces PaperQA2, a RAG agent designed for literature reviews, summarization, and contradiction-detection. They decompose RAG into tools: paper search (transform request into keyword search), gather evidence (top-k rank + rerank & contextual summarization (RCS)), generate answer, citation traversal (use the citation graph as hierarchical indexing). Performance on tasks was above average human expert level (using GPT-4-Turbo) and costs $1-3 per query
- 📜[Can LLMs Generate Novel Research Ideas?](https://arxiv.org/pdf/2409.04109v1)
    - Compared LLM-proposed (Claude 3.5 Sonnet) research ideas to 100+ NLP researchers. The LLM and humans were provided with the same idea template. The RAG setup retrieves and ranks existing papers from Semantic Scholar, generates many ideas, deduplicates (leaving 5%), and ranks them. The LLM ideas were rated as more novel and exciting, but slightly less feasible, after being transformed stylistically to be indistinguishable from human ideas. The authors note that despite producing many ideas, most of the ideas are similar, indicating raw sampling isn't a very effective form of test-time compute, and that LLMs are still sub-human at evaluating ideas

08/22/2024
- 📜[Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation](https://arxiv.org/pdf/2408.11812)
    - Introduces Crossformer, a single policy trained on 900k trajectories over 20 embodiments, varying in camera views, proprioceptive inputs, joint configurations, action outputs, and control frequencies. Tasks can be specified with images or language, and action-space specific heads handle emitting appropriate outputs. Input sequences consist of alternating chunks of image observations, proprioceptive observations, and readout tokens. Action chunking helps with temporal consistency & avoiding compounding errors. The policy avoids negative transfer (though also doesn't exhibit significant positive transfer), matches or outperforms SOTA policies tailored to each setting

08/21/2024
- 📜[Scaling Law With Learning Rate Annealing](https://arxiv.org/pdf/2408.11029)
    - Examines the loss curve of LLMs as a function of a forward area (a sum of the step-wise LR) and an annealing area (accounts for momentum & the more rapid decrease in loss as the LR decays). These two stages trade off with one another, and for WSD (warmup-stable-decay), the ideal annealing ratio is ~10% of total steps (decreasing w/ total steps). This framing aligns with several observed phenomena, e.g. optimal cosine annealing uses a cycle length equal to total number of steps & decays LR to zero, why constant LR can outperform cosine for a small number of steps, why higher re-warmup LR in continued pre-training spikes loss initially but results in a lower final loss, & why warmup steps matter less in continued pre-training. An advantage of this framing over Chinchilla scaling laws is because it predicts loss at any given step count, thousands of data points can be collected in a single training run, allowing for fitting a model with <1% the computational cost

08/20/2024
- 📜[ARCLE: The Abstraction And Reasoning Corpus Learning Environment For Reinforcement Learning](https://www.arxiv.org/pdf/2407.20806)
    - Introduces an RL environment (in Gymnasium) for the ARC benchmark. Actions are split into pixel-level selection and operation groups. Agents are trained via PPO. By default, the reward is exceptionally sparse, so auxiliary losses are added. Training using all losses and random initial grids yielded a success rate of >95%, 75% of the time. A separate experiment on policies shows that not assuming conditional independence between selection & operation is necessary for effective learning. The multi-task few-shot nature of ARC makes it a good fit for advanced RL approaches (meta-RL, generative models, & model-based RL)
- 📰[On the speed of ViTs and CNNs](http://lucasb.eyer.be/articles/vit_cnn_speed.html)
    - Pushes back against criticism that ViTs aren't practical at higher resolution for real-time processing. Makes the case that ViTs are fast enough for real-time image processing (>100 images/sec) at 1024x1024 resolution, and that for most tasks, we only need roughly 224px for most photos, 448px for reading text in digital images, 896px for reading a desktop screen/page of a document (which happen to be the resolutions used by PaliGemma)

08/19/2024
- 📰[AI Fundamentals: Energy-Based Models](https://mpmisko.github.io/2024/ai-fundamentals-energy-based-models/)
    - EBMs are a kind of generative models where the goal is to explicitly learn a probability distribution underlying training data, which allows the model to generate samples from the true distribution. Directly computing the MLE is expensive, so practical techniques include: contrastive divergence (CD) used to approximate the MLE; score matching, where the score function is the negative gradient of the log-prob wrt. x, and we minimize the square of the expected difference between the model's score and the data score; noise contrastive estimation (NCE), where the model is trained to distinguish between samples from the data distribution and a noise distribution. EBMs can be difficult to train (often diverge, sensitive to hyperparameters) and haven't been scaled as large as other models
- 📰[New LLM Pre-training and Post-training Paradigms](https://magazine.sebastianraschka.com/p/new-llm-pre-training-and-post-training)
    - Covers Qwen 2, Apple Intelligence Foundation Language Models (AFM), Gemma 2, Llama 3.1. Dataset filtering (quality over quantity), increased vocab size, synthetic data (incl. for context lengthening during pre-training), fancier RMs, & knowledge distillation have all become more popular (although Llama 3.1 notably did not use distillation). DPO/combinations of RLHF algorithms are now more popular than just PPO

08/14/2024
- 📜[Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers](https://arxiv.org/pdf/2408.06195)
    - Introduces rStar for small language models (SLMs), in an attempt to improve reasoning without a stronger teacher model. (Bigger models can achieve better improvement to reasoning on their own than smaller models.) rStar uses MCTS where the action space includes: propose a one-step thought, propose remaining thoughts (standard CoT), propose the next sub-question + answser (RAP), re-answer the sub-question, rephrase the sub-question. Rewards are determined by how likely the action is to lead to the right answer. Upper Confidence Bounds applied to Trees (UCT) is used to select each node. Since it's difficult to define a single metric that reliably selects the best trajectory, all trajectories are collected, and a second discriminator SLM is used to perform mutual reasoning consistency. MCTS alone provides a boost to performance, beating previous methods, and combined with the discriminator, rStar does even better. Weak discriminators seem to work fine, almost as well as GPT-4 (at least on GSM8K)

08/13/2024
- 📰[Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)
    - Used human annotators to verify a subset (500 samples) of SWE-bench, filtering out poorly specified issues & those with unfair tests. SOTA is now ~35%
- 📜[The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/pdf/2408.06292)
    - Claims to be a pipeline for e2e paper generation at ~$15/paper. The pipeline is broken into idea generation, experiment iteration, & paper write-up. Generation involves multiple rounds of CoT & self-reflection, conditioned on an existing archive of research, & a semantic filter to exclude ideas too similar to existing literature. Experiments use Aider to plan & execute ideas, try fixing errors, & take notes on results. To draft the paper, Aider fills out a LaTeX template & refines using self-reflection. The system then polls the Semantic Scholar API to fill in citations & generate the Related Work section. A final round of self-reflection on the entire paper is used to streamline the arguments. Finally, the paper is fed through a LaTeX compiler & errors are passed back to Aider for correction. The pipeline is able to generate & produce empirical results for some ideas, but sometimes fails, or doesn't explain all reasoning, still hallucinates some details, produces a limited bibliography, & messes up visualizations (since it doesn't use vision capabilities). Claude 3.5 Sonnet produced the highest quality papers
    - Separately, they use an agent (GPT-4o) to review & score papers, which uses self-reflection, few-shot examples, & response ensembling, which achieves roughly human-level classification (accuracy, F1, AUC), although with a significantly higher false positive rate, at a cost of $0.25-50/review.
    - The system occasionally made attempts to modify its own code, resulting in excess resource usage. The authors recommend strict sandboxing of the system
- 📜[Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents](https://multion-research.s3.us-east-2.amazonaws.com/AgentQ.pdf)
    - Introduces Agent Q: a base model + RFT on successful trajectories + DPO over all trajectories. During training, MCTS is used to explore more options & collect rewards. Preferences for DPO are determined via a mixture of the MCTS reward + an estimate from a frozen critic LLM's ranking over potential actions, since the former reward from the outcome provides limited supervision. On the simulated WebShop benchmark, Agent Q marginally outperforms DPO using outcome supervision alone and doesn't quite meet human performance. By combining the model with MCTS at test time, human-level performance is achieved. When applied to booking a table on OpenTable, Agent Q + MCTS reaches a 95.4% success rate. The relative performance gap here is higher, possibly due to fine-grained feedback becoming more important as the number of required steps grows. The authors note this approach would not be safe in environments where interactions are risky or can't be undone

08/12/2024
- 📜[From LLMs to LLM-based Agents for Software Engineering: A Survey of Current, Challenges and Future](https://arxiv.org/pdf/2408.02479)
    - Surveys papers from H2 2023 - H1 2024. LLMs for SE are limited by context length, hallucinations, and lack of access to real-time info & tools. RAG is still faster and cheaper than using a long context. Multi-agent systems are becoming more common to work around limitations. Outputs from agents are more open-ended/harder to measure, so benchmarks/evals tend to be all over the place. The most concrete success has been from code gen (copilot) & test gen/bug fixes, especially using pass@k. Surprising how often GPT-3.5 (or equivalent) is still used
- 📜[A Survey of Mamba](https://arxiv.org/pdf/2408.01129)
    - The theoretical advantage of Mamba having linear scaling with length during inference has motivated a lot of research. Thus far most Mamba models have been small relative to frontier transformers, and struggle with recall early in the context. There's plenty of ongoing work, but the ecosystem & hardware optimization are far behind work on transformers
- 📜[Tree Attention: Topology-Aware Decoding for Long-Context Attention on GPU Clusters](https://arxiv.org/pdf/2408.04093)
    - Introduces a method to parallelize attention across GPUs using tree reduction, providing an asymptotic speedup over Ring Attention. They consider self-attention as a gradient of an energy function. Since computing the gradient of f(x) has the same time complexity as computing f(x), and since logsumexp is associative, it can be reduced in parallel, leading to a log(N) reduction in complexity. They achieved an 8x speedup and less memory usage over Ring Attention when using 128 GPUs on a sequence of 5.12M tokens

08/09/2024
- 📜[Reasons to Doubt the Impact of AI Risk Evaluations](https://arxiv.org/pdf/2408.02565)
    - Leading industry, government, & safety labs allocate significant resources to evals, in the hope that they improve understanding of risks & enable mitigating them. However, evals may fail to improve understanding (miss risks due to interactions with the real world, cost more than building scary demos of capabilities, fail to capture discoveries in deployment). They may also fail to mitigate risks after lines are crossed (voluntary commitments are not dependable, governments can be slow to react, evals don't improve safety culture). They may even backfire (becoming goals for dual-use capabilities, consuming resources that could be used for technical safety/governance progress, contributing to safety-washing, leaking scary demos). To improve the situation, stakeholders should be aware of hype, measure propensities as well as capabilities, ensure evals can be done pre-deployment (& white-box), & continue to make eval practices more rigorous. Labs should honor evals commitments, provide access to models, & share eval infrastructure. External evaluators should specialize & cooperate on standards Government should require lab cooperation & clarify protections for doing so. Researchers should advance a broad science of evaluation & develop better threat modeling.

08/08/2024
- 📜[POA: Pre-training Once for Models of All Sizes](https://www.arxiv.org/pdf/2408.01031)
    - POA builds on teacher-student distillation by adding an "elastic student" as another branch. The elastic student is a random subset of the student's parameters, chosen by randomly sampling from among a combination of widths and depths (biased toward smaller sub-networks). The elastic students acts as regularizers/an ensemble during training, and can be directly extracted from the pre-trained teacher. Both the teacher and extracted students achieve SOTA performance on k-NN classification for ImageNet-1K, object detection & segmentation on COCO, and semantic segmentation on ADE20K
- 📜[Grokfast: Accelerated Grokking by Amplifying Slow Gradients](https://arxiv.org/pdf/2405.20233)
    - Hypothesizes that grokking is caused by fast-varying gradients initially leading to over-fitting, followed by slow-varying gradients eventually yielding generalization. By considering the changes in parameters in the frequency domain, applying a low-pass filter (moving average) to the gradients could accelerate grokking. If the fast-moving gradients are excluded entirely, training speed & stability worsens. Grokking happens faster the most when averaging is applied after an initial overfitting phase, and faster still when adding weight decay (up to 50x faster on a modular arithmetic task). Since keeping a large window of historical gradients would require a lot of memory, an exponential moving average is also tested, yielding similar results (22x faster grokking on an MNIST classifier, faster/better convergence on a graph CNN and a two-layer LSTM for IMDB sentiment analysis). When studying parameter trajectories, the Grokfast model deviates much more from initial states before overfitting, but then much less during the grokking transition (& with 100x less variance), implying it's more deterministic
- 📜[Human vs. Machine: Behavioral Differences between Expert Humans and Language Models in Wargame Simulations](https://arxiv.org/pdf/2403.03407)
    - Used few-shot ICL (GPT-4) to train separate critique, refine, and rank models to help decompose competitive programming problems. Non-experts with assistance reached unassisted expert level. Decomposing problems also enabled the model to self-supervise (repair programs generated by itself)
- 📜[Achieving Human Level Competitive Robot Table Tennis](https://arxiv.org/pdf/2408.03906)
    - Uses a hierarchy with a high-level controller (HLC) to select among low-level skill policies (LLCs) (to avoid catastrophic forgetting & improve evaluation efficiency). Policies are trained using Blackbox Gradient Sensing (BGS), an evolutionary strategies (ES) algorithm, rather than gradient descent, since RL algorithms like PPO resulted in jerkier movements. Policy models are small (~10k params) dilated-gated CNNs. The HLC observes a timestep to estimate ball velocity and then chooses an LLC to return the ball. Policies were trained iteratively, starting with a seed dataset of 40 minutes of human-human play, then alternating between sim and zero-shot deployments against human players. Simulation was in MuJoCo with fluid dynamics. To reduce the sim-to-real gap, ~100 samples of real-world data were used to update LLC lookup. The robot played at an intermediate level, consistently beating beginners and losing to advanced players

08/07/2024
- 📰[A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
    - bfloat16 introduced to maintain range of 32-bit float with slightly less precision in fewer bits. Zero-point (asymmetric) quantization & clipping allow for compressing the range of values to fewer bits. Static quants use a calibration dataset to find appropriate scale and zeropoint values. GPTQ uses per-layer quants, computed using the inverse-Hessian to determine which weights are most sensitive. GGUF splits each layer into super blocks and again into sub blocks, where the sub blocks use absmax (symmetric) quantization with a scale factor informed by the super block. BitLinear (1-bit) quantizes weights to 1 bit during training by centering the distribution around 0 and uses absmax to quantize the activations
- 📜[The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/pdf/2402.17764v1)
    - BitNet replaces linear layer weights during training with absmean quants rounding to -1, 0, or 1, which allows matrix multiplications to become additions. This reduces memory consumption, latency, & energy, and increases throughput, with equal perplexity for larger models. For Llama 70B, improvements were 7x for memory consumption, 4x for latency, 41x for energy, and 9x for throughput.
- 📜[Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/pdf/2401.06118)
    - Introduces AQLM (Additive Quantization of Language Models), which splits weight rows of linear layers into groups of weights represented by a sum of vectors from learned codebooks and codes, optimized by iteratively performing a beam search over code values followed by gradient descent over codebooks. After quantizing linear layers, the remaining parameters are fine-tuned to approximate the original outputs. They achieve 2-bit PTQ (post-training quant) with minimal quality loss (much better than GPTQ and slightly better than QuIP#), achieving pareto-optimality below 3 bits
- 📜[PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression](https://arxiv.org/pdf/2405.14852)
    - Makes the case that straight-through estimation for fine-tuning compressed weights is sub-optimal and introduces PV-tuning, which iteratively alternates between optimizing continuous (P) and discrete (V) parameters. PV-tuning is designed to work with various PTQ methods and achieves pareto-optimality at 2.0 bits, allowing a 2-bit 13B model (Llama 2) to outperform the 16-bit 7B model

08/05/2024
- 📜[Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models](https://arxiv.org/pdf/2408.00113)
    - Applied SAEs to Othello and chess models and introduced p-annealing, where the sparsity penalty starts at the L1 norm and gradually approaches the L0 norm (which is non-differentiable). p-annealing on a standard SAE was comparable to gated SAEs. However, none achieve reconstruction comparable to a linear probe, offering further evidence that SAEs aren't capturing all information in the model's representations
- 📜[Tamper-Resistant Safeguards for Open-Weight LLMs](https://arxiv.org/pdf/2408.00761)
    - Introduces Tampering Attack Resistance (TAR), adversarial training designed to minimize the ability to fine-tune away safeguards. They start with a model with a general safeguard like circuit breaking. TAR performs an inner loop sampling attacks against the safety metric. The gradient from the inner loop is then mixed in an outer loop with the gradient from a retain loss (from representation engineering) to preserve capabilities performance. Notably, the tamper-resistance loss is negative entropy rather than negative cross-entropy, since the model can learn to exploit the latter. After fine-tuning attacks, TAR maintained near-random performance on WMDP, significantly more robust than prior approaches. TAR also achieves a lower attack success rate (ASR) on HarmBench, although the success rate is still 64%. TAR does impose a cost on capabilities, comparable to other approaches

08/02/2024
- 📜[The Larger the Better? Improved LLM Code-Generation via Budget Reallocation](https://arxiv.org/pdf/2404.00725)
    - Evaluated Code Llama (7B to 70B) on pass@k for HumanEval, MBPP, APPS. For a fixed compute/wall-time budget, 7B and 13B can outperform even the 70B model. Note this appears dependent on the difficulty of the task, since for the competition-level APPS split, 7B was dominated by larger models. In situations where unit tests aren't available, a larger model can be used as a ranker for generations from smaller models, but if wall-time isn't a constraint, simply generating from the larger model yields much higher accuracy.
- 📜[Safetywashing: Do AI Safety Benchmarks Actually Measure Safety Progress?](https://arxiv.org/pdf/2407.21792)
    - Argues that if a safety benchmark correlates highly with general capabilities benchmarks, it's liable for safetywashing. Since benchmarks impact incentives for resource allocation, care should be taken to use/develop safety benchmarks which implicitly control for capabilities, which requires empirical measurement. Scores on MT-Bench and LMSYS Chatbot Arena are highly correlated with capabilities. ETHICS is highly correlated with capabilities, likely because it's measuring recognition of moral considerations, whereas MACHIAVELLI and Sycophancy are are not. Bias benchmarks are generally not correlated with capabilities. TruthfulQA is highly correlated with capabilities. GPQA and QuALITY are (unsurprisingly?) highly correlated as well. RMS calibration error is safe to use, but Brier scores are not, since they entangle accuracy and calibration. Older adversarial robustness benchmarks (ANLI, AdvGLUE, ImageNet-A) are highly correlated with capabilities, but newer ones (HarmBench, PGD) are not. WMDP is anti-correlated with capabilities

08/01/2024
- 📰[Extrinsic Hallucinations in LLMs](https://lilianweng.github.io/posts/2024-07-07-hallucination/)
    - A lot of hallucinations come from incorrect pre-training data. Benchmarks like FactualityPrompt & FActScore measure general factuality, TruthfulQA measures accuracy on adversarial examples of common human falsehoods, and SelfAware measures a model's ability to know whether it knows a question is unanswerable. FAVABench measures fine-grained kinds of hallucinations. Pretrained models tend to be better calibrated on their correctness (scaling with model size), but RLHF reduces calibration. RAG, RARR, FAVA, RR, and Self-RAG are all methods that use external information to augment/correct answers. Chain-of-verification (CoVe) and recitation-augmented-generation (RECITE) both use the model itself to reduce hallucinations. There are several approaches to favoring factuality/attribution during SFT/DPO
- 📜[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401)
    - RAG augments the model with non-parametric memory: documents are embedded prior to queries (often in small chunks), selected at query-time by similarity (FAISS), and used to augment the prompt. The pre-trained retriever and generator are fine-tuned end-to-end. Describes RAG-token, where the generator produces a distribution for the next token for each (top K) document, and RAG-sequence, where a separate beam search is run over each (top K) document for the entire sequence
- 📜[Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/pdf/2312.10997)
    - Naive RAG can have issues with selecting misaligned chunks and integrating retrieved information with the query. Advanced techniques include optimizing indexing structure, optimizing the query (rewriting, expanding, decomposing), post-retrieval processing (reranking chunks, compressing context), iterative/recursive retrieval, and specialized modules (direct searches across different data sources, intelligent routing, using task-specific retrievers). There are many metrics/benchmarks to evaluate different approaches, but no standard

07/31/2024
- 📰[Circuits Updates - July 2024](https://transformer-circuits.pub/2024/july-update/index.html)
    - The Next Five Hurdles: missing features (SAEs are likely only extracting a small fraction of total features), cross-layer superposition (though residual stream SAEs can maybe address features from previous layers), attention superposition, interference weights, zooming out (how do we go from understanding features/circuits to understanding the model as a whole? How much will automated interp help?)
    - What is a Linear Representation? What is a Multidimensional Feature?: there has been some ambiguity around the linear representation hypothesis. Are features one-dimensional representations, or linear in a mathematical sense (addition and scaling)? Olah thinks the latter is the better definition and talks about multidimensional feature manifolds, but also ends with a note that definitions should be fluid in research and imperfect theories can still be productive
    - The Dark Matter of Neural Networks?: models may have "memorization features" which are extremely numerous & sparse (hence "dark matter")
    - Attention Pivot Tables: notes on reproducing early work on interpreting single-layer transformers as implementing skip-trigrams, and how "fiddly" this was
    - Measuring feature sensitivity using dataset filtering: despite SAEs finding interpretable features that are highly specific (only fire for a specific concept), many of them appear to not be very sensitive (don't fire even when humans/Claude think the text highly relates to the feature). This may be because the feature is subtly related to a concept rather than representing the concept as a whole
- 📰[Open Source Automated Interpretability for Sparse Autoencoder Features](https://blog.eleuther.ai/autointerp/)
    - Released a library to generate and score explanations of SAE features using LLMs. This has become drastically cheaper with the latest models (e.g. Llama-3 70B). This works some of the time, but explanations aren't precise enough to distinguish among similar concepts, and a significant fraction of explanations don't generate samples that activate the feature at all (consistent with above findings from Anthropic).
- 📰[Exploring Gemma Scope](https://www.neuronpedia.org/gemma-scope#main)
    - Interactive site explaining/demoing uses of extracting features with SAEs, including steering

07/30/2024
- 📜[Scaling Exponents Across Parameterizations and Optimizers](https://arxiv.org/pdf/2407.05872)
    - Gives theoretical and empirical backing (tens of thousands of models, up to 27B parameters) for per-layer learning rates and scaling epsilon, or removing it entirely, as in Adam-atan2. Alignment/correlation between parameter and data vectors can cause a significant (and non-monotonic) shift in activation norms across layers and over time, motivating the above recommendations. Following these guidelines can find hyperparameters on small versions of models that transfer well to a larger scale
- 📜[Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization](https://arxiv.org/pdf/2405.15071)
    - Provides empirical evidence of reasoning achieved only via grokking, and a mechanistic explanation for why composition of facts is harder than comparison of facts: the former requires multiple steps, so only facts seen during training get generalized, whereas the latter can be done in parallel, so the model more easily generalizes over OOD facts. They also perform an experiment on an extended version of the comparison task requiring anti-symmetry & transitivity. GPT-4-Turbo and Gemini-Pro-1.5 perform poorly even with RAG and CoT, while the much smaller grokked transformer achieves near-perfect accuracy
- 📜[Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/pdf/2407.04620)
    - Proposes a new layer with linear complexity where the hidden state is itself an ML model. In the forward pass, the layer performs self-supervised learning to reconstruct the input. Parameters from the outer loop (rest of the network, reconstruction views, initial inner weights, inner learning rate) are learned during training, and the layer's inner weights are learned during inference. To improve speed, the inner loop learns on a batch of tokens (e.g. 16), at a slight cost to perplexity. They tested with both linear and two-layer MLP hidden states and transformer and Mamba backbones, up to 1.3B. The TTT layers had lower perplexity than vanilla transformer and Mamba backbones, particularly on longer context lengths, although they didn't observe clean scaling laws.

07/29/2024
- 📜[DiLoCo: Distributed Low-Communication Training of Language Models](https://arxiv.org/pdf/2311.08105)
    - Distributed Low-Communication (DiLoCo) splits training into an outer and inner loop. The inner loop is performed on distributed heterogeneous workers. Each worker has its own data shard, is passed model parameters from the outer loop, performs its own optimization over many steps (using AdamW), and then sends gradients to the outer optimizer. The outer optimizer (using Nesterov momentum) averages gradients and updates parameters sent to the workers. Scaling the number of inner steps results in slightly worse perplexity but much faster training, with a sweet spot of 500. Results were robust for non-i.i.d. data sharding, adjusting compute per shard, and simulating dropped gradient communications, providing evidence total compute is what matters most. Note tested models were only 400M params
- 📜[DiPaCo: Distributed Path Composition](https://arxiv.org/pdf/2403.10616)
    - DIstributed PAths COmposition (DiPaCo) splits training and inferencing of a large model into sparsely activated modules (trained a la DiLoCo). DiPaCo is designed for a world where compute is cheap and communication is expensive (not the current world). During training, routing to a path is determined by the entire input sequence. At test/inference time, the sequence is split into chunks and each chunk is routed separately. A 150M path performs nearly as well as a dense 1.3B model in less wall-clock time, although it's difficult to compare FLOPs used in training given the architecture. Performance is improved by routing smaller chunks, although this would incur throughput issues in the real world (having to recompute the KV cache after each routing decision)
- 📜[The Future of Large Language Model Pre-training is Federated](https://arxiv.org/pdf/2405.10853)
    - Introduces Photon, a federated learning (FL) system similar to DiLoCo, but with an emphasis on heterogeneous compute and private data. A node abstracts over a single GPU, multi-GPU, or multi-machine client, which receive the model & send gradients back to an aggregator. They trained models up to 1.3B in size on heterogeneous clients, with the largest model performing as well as a centrally-trained one, and during later rounds federated training acts as a regularizer

07/26/2024
- 📜[Planning behavior in a recurrent neural network that plays Sokoban](https://arxiv.org/pdf/2407.15421)
    - Reproduced a prior setup using a Deep Repeating ConvLSTM (DRC) architecture to solve Sokoban puzzles. By repeating the initial observation and advancing the DRC hidden state at the start of an episode, the agent gets to "think" before taking an action. This improves the agent's ability to solve harder puzzles. The agent also naturally exhibits cycling/"pacing" behavior; however, the more thinking steps the agent is given at the start of the episode, the less it cycles on its own, indicating that pacing is a learned form of planning/mesa-optimization
- 📺[Eric Wallace: Memorization in language models](https://www.youtube.com/watch?v=dXTfY7tgb-o)
    - Repeating data makes it easier to memorize, but larger models are also better at memorizing after very few examples. There are ways to mitigate/undo memorization, but they're generally expensive and don't work against targeted attacks
- 📺[Martin Wattenberg: Models within models - how do LLMs represent the world?](https://www.youtube.com/watch?v=P_FmwO-wVK8)
    - Covered case studies of Othello-GPT (player-opponent board state) & Stable Diffusion (depth, foreground-background), speculated about a user-system model in LLMs, and raised the question of what data about the internal model would be useful (& not) to expose to the user
- 📺[Nicholas Carlini: The security of LLMs](https://www.youtube.com/watch?v=_EbntLk5QTk)
    - Adversarial robustness in image models remains a challenge after a decade of research, although attacks weren't relevant to bad actors. Adversarial robustness has become much more important for LLMs. Despite text not being differentiable, the embeddings are, so LLMs are also highly susceptible to gradient-based attacks. Just like image models, attacks transfer to LLMs with a different architecture and trained on different data. Data poisoning is becoming more important

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