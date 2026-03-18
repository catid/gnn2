## **Evolution Strategies at the Hyperscale** 

**Bidipta Sarkar** _[∗]_[1] _[,]_[2] **, Mattie Fellows** _[∗]_[1] **, Juan Agustin Duque** _[∗]_[2] _[,]_[3] **, Alistair Letcher** _[†]_[1] **, Antonio León Villares** _[†]_[1] **, Anya Sims** _[†]_[1] **, Clarisse Wibault** _[†]_[1] **, Dmitry Samsonov** _[†]_[6] **, Dylan Cope** _[†]_[1] **, Jarek Liesen** _[†]_[1] **, Kang Li** _[†]_[1] **, Lukas Seier** _[†]_[1] **, Theo Wolf** _[†]_[1] **, Uljad Berdica** _[†]_[1] **, Valentin Mohl** _[†]_[1] **, Alexander David Goldie**[1] _[,]_[2] **, Aaron Courville**[3] _[,]_[5] **, Karin Sevegnani**[4] **, Shimon Whiteson** _[‡]_[2] **, Jakob Nicolaus Foerster** _[‡]_[1] **.** 

> 1 FLAIR - University of Oxford, 2 WhiRL - University of Oxford, 3 MILA– Québec AI Institute 

> 4 NVIDIA AI Technology Center, 5 CIFAR AI Chair, 6 NormaCore.dev 

{bidipta.sarkar,matthew.fellows,jakob.foerster}@eng.ox.ac.uk juan.duque@mila.quebec, shimon.whiteson@cs.ox.ac.uk 

## **Abstract** 

Evolution Strategies (ES) is a class of powerful black-box optimisation methods that are highly parallelisable and can handle non-differentiable and noisy objectives. However, naïve ES becomes prohibitively expensive at scale on GPUs due to the low arithmetic intensity of batched matrix multiplications with unstructured random perturbations. We introduce Evolution Guided GeneRal Optimisation via Low-rank Learning (EGGROLL), which improves arithmetic intensity by structuring individual perturbations as rank- _r_ matrices, resulting in a hundredfold increase in training speed for billion-parameter models at large population sizes, achieving up to 91% of the throughput of pure batch inference. We provide a rigorous theoretical analysis of Gaussian ES for high-dimensional parameter objectives, investigating conditions needed for ES updates to converge in high dimensions. Our results reveal a linearising effect, and proving consistency between EGGROLL and ES as parameter dimension increases. Our experiments show that EGGROLL: (1) enables the stable pretraining of nonlinear recurrent language models that operate purely in integer datatypes, (2) is competitive with GRPO for post-training LLMs on reasoning tasks, and (3) does not compromise performance compared to ES in tabula rasa RL settings, despite being faster. Our code is available at https://eshyperscale.github.io/. 

**==> picture [385 x 110] intentionally omitted <==**

**----- Start of picture text -----**<br>
Rank-one perturbation  Ei Fitness evaluation Weighted average<br>Initial weights Final rank- N update<br>= f + σ<br>� �<br>= f + σ<br>� �<br>... ...<br>= f + σ<br>� �<br>**----- End of picture text -----**<br>


Figure 1: Schematic visualisation of EGGROLL using _N_ workers. 

## **1 Introduction** 

Evolution Strategies (ES) (Rechenberg, 1978; Beyer, 1995; Beyer & Schwefel, 2002) is an attractive alternative to first-order methods based on gradient backpropagation for several reasons. First, ES does not require differentiability; it can optimise a broader class of models, like those with discrete parametrisations (cellular 

> *Equal Contribution _†_ Core Contributor, sorted by alphabetical order in first names _‡_ Equal Senior Authors 

1 

**==> picture [430 x 118] intentionally omitted <==**

**----- Start of picture text -----**<br>
Normalised Training Speeds Pure Integer Pretraining<br>100 10 [6]<br>91 8 EGGROLL (int8)<br>80 Backprop (fp32)<br>10 [4]<br>60 6<br>40 34 10 [2]<br>4<br>20<br>10 [0]<br>0.41 10 [1] 10 [2] 10 [3]<br>0<br>EGGROLL PPO OpenES Training Step<br>Population Size<br>Normalised Speed Test Loss (bits/byte)<br>**----- End of picture text -----**<br>


Figure 2: (a) Relative speed of our method, EGGROLL, in terms of experience throughput versus prior methods, where 100 is the maximum batch inference throughput. See Appendix E for more details. (b) We use EGGROLL to train an int8 RNN language model from scratch, scaling population size from 2 to 1,048,576 with a fixed data batch size of 16. The dotted line is a fp32 Transformer trained with backprop SGD. EGGROLL’s test next-token cross-entropy of 3.40 bits/byte while backprop only gets 3.58 bits/byte. 

automata) or objectives for which gradients are unavailable or noisy, such as outcome-only rewards in LLM fine-tuning (Qiu et al., 2025). Second, ES can be more robust to noisy and ill-conditioned optimisation landscapes (Wierstra et al., 2011; Xue et al., 2021). Population-based exploration smooths irregularities (Salimans et al., 2017), tolerates discontinuities, and mitigates issues like ill-conditioned curvature or vanishing and exploding gradients in long-range or recurrent settings (Hansen, 2023). Third, ES is highly amenable to parallel scaling, since fitness evaluations are independent across population members and require only the communication of scalar fitnesses, which maps cleanly onto modern inference infrastructure and yields near-linear speedups on large clusters (Salimans et al., 2017). By contrast, backpropagation requires communicating and aggregating gradients across devices, yielding updates with high memory and computational costs. Furthermore, backpropagation requires special care when training models with low-precision datatypes (Fishman et al., 2025), whereas ES can directly optimise any model with the same datatypes used at inference time. Together, these properties position ES as a potentially powerful tool for training large, discrete, or hybrid architectures, and end-to-end systems with non-differentiable components, including LLMs (Brown et al., 2020; Chowdhery et al., 2023; Du et al., 2022; Fedus et al., 2022). 

However, there are currently practical obstacles to employing ES at scale. In deep learning architectures (Goodfellow et al., 2016), the majority of trainable parameters form linear mappings represented by matrices (Rosenblatt, 1962; Hochreiter & Schmidhuber, 1996; Bengio et al., 2000; Krizhevsky et al., 2012; Goodfellow et al., 2014; Kingma & Welling, 2014; Vaswani et al., 2017). Naïvely adapting ES therefore requires generating full-rank matrix perturbations that replicate the entire parameter set for every population member. This inflates memory costs and forces frequent movement of large weight tensors. Evaluating these perturbations then requires a separate sequence of matrix multiplications per member, so the total compute and wall-clock time scale roughly with the population size and sequence length since batched matrix multiplication has a low arithmetic intensity, i.e., the ratio of arithmetic operations to memory traffic (Williams, 2008). In billion-parameter regimes, these two costs dominate, limiting ES to small models and small populations (Qiu et al., 2025; Korotyshova et al., 2025). 

To mitigate both memory and computational bottlenecks, we introduce Evolution Guided GeneRal Optimisation via Low-rank Learning (EGGROLL), an ES algorithm that allows for the efficient training of neural network architectures with billions of parameters. Analogous to LoRA’s low-rank adapters in gradient-based training (Hu et al., 2022), EGGROLL generates _low-rank_ parameter-space perturbations for ES; instead of sampling a full-rank matrix _E ∈_ R _[m][×][n]_ , we sample _A ∈_ R _[m][×][r]_ and _B ∈_ R _[n][×][r]_ with _r ≪_ min( _m, n_ ) and form _E_ = 1[This][reduces][auxiliary][perturbation][matrix][storage][from] _[mn]_[to][(] _[m]_[ +] _[ n]_[)] _[r]_[per][layer,][and] ~~_√_~~ _r[AB][⊤]_[.] proportionally reduces tensor movement. 

Moreover, we use a counter-based deterministic random number generator (RNG) (Salmon et al., 2011; Bradbury et al., 2018) to reconstruct noise on demand, so matrix perturbations need not persist in memory. When evaluating the fitness of members of multiple perturbations in parallel, EGGROLL batches a population of low-rank adapters and shares the base activations, enabling a single forward pass that applies all _AB[⊤]_ updates via specialised batched matrix multiplications with significantly higher arithmetic intensity, resulting 

2 

in over a hundredfold increase in training throughput for large neural networks at large population sizes, as shown in Fig. 2a. Crucially, EGGROLL does not restrict updates to be low-rank, as the overall update is a weighted average of rank _r_ matrices across the population, making the matrix parameter update rank min( _Nr, m, n_ ) . 

To understand ES when applied to large parameter models, we analyse the convergence properties of general Gaussian ES in high dimensions, showing there exists a critical noise scaling _σd_ = _o_ ( _d[−]_[1] _[/]_[2] ) under which the update provably linearises and converges to the first-order derivative for a broad class of (possibly discontinuous) objectives. We identify three distinct regimes—linearisation, critical, and divergence—and establish provably tight conditions for stable ES optimisation in large models. Building on this, we extend the analysis to EGGROLL and prove that even fixed low-rank updates (including rank-1) converge to the true ES gradient as dimension grows, despite heavier-tailed perturbations. Our results explain the empirical success of EGGROLL in high-dimensional neural networks and connect its behaviour to neural tangent kernelstyle linearisation (Jacot et al., 2018), yielding explicit convergence rates under standard overparameterised regimes. We also provide a rigorous theoretical analysis of the low-rank approximation accuracy, proving that EGGROLL updates converge to the full-rank Gaussian ES updates at a fast _O_ ( _r[−]_[1] ) rate. 

Furthermore, in our extensive empirical evaluation, we test this hypothesis across a wide range of domains. In tabula rasa and multi-agent RL (MARL) settings, we show that EGGROLL does not compromise performance compared to naïve ES despite being faster. We demonstrate the scalability of EGGROLL for LLM fine-tuning with experiments on pretrained RWKV7 (Peng et al., 2025) models, modern recurrent language models that enable large batch inference due to their constant state size. Finally, we develop a nonlinear RNN language model that operates purely in integer datatypes, and demonstrate that EGGROLL can stably pretrain this language model, a feat which is only feasible due to the large population sizes enabled by EGGROLL. 

## **2 Preliminaries** 

## **2.1 Low-Rank Matrix Approximations** 

When adapting high-dimensional foundation models for specific tasks, updating the parameters using gradientbased methods has high memory requirements. LoRA (Hu et al., 2022) applies low-rank approximations to the matrix multiplications to reduce these costs. For each matrix _Mi ∈_ R _[m][×][n]_ in the model, a low-rank approximation can be made by decomposing each matrix: 

**==> picture [83 x 13] intentionally omitted <==**

where _Mi_[0][:=][StopGrad][(] _[M][i]_[)][is][the][imported][matrix][from][the][foundation][model][with][frozen][parameters] and _Ai ∈_ R _[m][×][r]_ and _Bi ∈_ R _[n][×][r]_ are low-width column matrices (i.e., _r ≪_ min( _m, n_ )) whose parameters are updated through gradient-based optimisation during task-specific adaptation. This reduces the number of optimisation parameters for each matrix from _mn_ to _r_ ( _m_ + _n_ ). EGGROLL uses a similar low-rank approximation for evolutionary strategies. 

## **2.2 Evolution Strategies** 

Evolution strategies (ES) (Rechenberg, 1978; Beyer, 1995; Beyer & Schwefel, 2002) is a set of black-box optimisation methods that has emerged as a useful alternative to first-order gradient-based methods like stochastic gradient descent (SGD), particularly for noisy or non-differentiable systems. Let _f_ : R _[d] →_ R denote an objective to be optimised, known as the _fitness_ , where the goal is to find an optimising set of parameters _x[⋆] ∈_ arg max _x∈_ R _d f_ ( _x_ ). Each set of parameters is collected into a _d_ -dimensional vector known as a genotype. We denote the derivative of the fitness _∇xf_ ( _x_ ) _|x_ = _a_ evaluated at _x_ = _a_ as _∇f_ ( _a_ ). Unlike first-order gradient-based methods, which query derivatives _∇f_ ( _x_ ) to update the vector of parameters _x_ , evolutionary methods update a parametric population distribution over the fitness parameter space _π_ ( _x|θ_ ), which is smoothly parametrised by a separate set of parameters _θ ∈_ Θ. The population distribution generates perturbations _x ∼ π_ ( _x|θ_ ) known as mutations. The problem of optimising the fitness _f_ ( _x_ ) for _x_ reduces to optimising the parameters of the population distribution _θ_ . This is achieved by solving a _secondary_ optimisation problem to maximise the expected fitness under _π_ ( _x|θ_ ) for _θ_ : 

_J_ ( _θ_ ) = E _x∼π_ ( _x|θ_ ) [ _f_ ( _x_ )] _._ 

3 

Introducing a population distribution _smooths_ the fitness landscape; since _π_ ( _x|θ_ ) is smooth in _θ_ , the resulting objective _J_ ( _θ_ ) is also smooth in _θ_ , provided _f_ ( _x_ ) is measurable and integrable but not necessarily differentiable. Evolution strategies can therefore optimise black-box problems that may be non-differentiable as the derivatives of _J_ ( _θ_ ) exist for fitness functions that are discontinuous, yielding a gradient with respect to _θ_ : 

**==> picture [174 x 12] intentionally omitted <==**

where _∇θ_ log _π_ ( _x|θ_ ) is known as the score function. A Monte Carlo estimate is formed by sampling _N_ search mutations _xi ∼ π_ ( _xi|θ_ ) and computing an average of the score-weighted fitnesses: 

**==> picture [297 x 30] intentionally omitted <==**

with which we update _θ_ via stochastic gradient ascent with a suitable stepsize _αt_ : 

**==> picture [104 x 12] intentionally omitted <==**

ES does not require taking derivatives directly through the fitness function; instead the Monte Carlo update in Eq. (1) only requires evaluation of _f_ ( _xi_ ) for each mutation _xi_ to estimate _∇θJ_ ( _θ_ ). As ES only queries _f_ ( _x_ ) and not _∇f_ ( _µ_ ), it is a _zeroth-order_ optimisation method. 

In this paper, we study ES using Gaussian population distributions: _π_ ( _x|θ_ ) = _N_ ( _µ, Idσ_[2] ). In addition to its mathematical convenience, the central limit theorem means that the Gaussian distribution emerges naturally from the EGGROLL low-rank approximation as rank increases, even if the matrices _A_ and _B_ are themselves non-Gaussian. Moreover, most widely-used ES algorithms assume Gaussian population distributions (Rechenberg, 1978; Schwefel, 1995; Hansen & Ostermeier, 2001a; Beyer & Schwefel, 2002; Auger & Hansen, 2011; Wierstra et al., 2011; Salimans et al., 2017). In our setting, ES optimises over the population mean _µ ∈_ R _[d]_ , which acts as a proxy for the true maximum of the fitness function, and the variance parameter _σ_[2] _≥_ 0 is treated as a hyperparameter to be tuned. 

For the Gaussian population distribution we study in this paper, the ES update can be written using an expectation under a standard normal distribution by making a transformation of variables _v_ = _[x][−] σ[µ]_ (Wierstra et al., 2011; Salimans et al., 2017): 

**==> picture [326 x 45] intentionally omitted <==**

where _v ∼ P_ ( _v_ ) = _N_ (0 _, Id_ ) and _p_ ( _v_ ) denotes the density of _P_ ( _v_ ). In this form, Eq. (2) shows that Gaussian ES methods optimise the fitness by generating search vectors from a standard normal distribution _N_ (0 _, Id_ ) around the mean parameter _µ_ . 

## **2.3 Evolution Strategies for Matrix Parameters** 

A key focus of this paper is to develop efficient methods for evolution strategies that target _matrix parameters_ . When working in matrix space, it is convenient to use the matrix Gaussian distribution (Dawid, 1981), which is defined directly over matrices _X ∈_ R _[m][×][n]_ : 

**==> picture [372 x 25] intentionally omitted <==**

where _M ∈_ R _[m][×][n]_ is the mean matrix, _U ∈_ R _[m][×][m]_ is the row covariance matrix and _V ∈_ R _[n][×][n]_ is the column covariance matrix. We use vec( _·_ ) to denote the vectorisation operator: 

**==> picture [174 x 13] intentionally omitted <==**

The matrix Gaussian distribution is a generalisation of the multivariate Gaussian distribution _N_ ( _µ,_ Σ) defined over vector space. Sampling a matrix _X ∼N_ ( _M, U, V_ ) from a matrix Gaussian distribution is equivalent 

4 

to sampling a vector vec( _X_ ) _∼N_ ( _µ,_ Σ) from a multivariate Gaussian distribution with mean _µ_ = vec( _M_ ) and covariance matrix Σ = _V ⊗ U_ where _⊗_ denotes the Kronecker product. For isotropic matrix Gaussian distributions with covariance matrices _U_ = _σIm_ and _V_ = _σIn_ , the equivalent multivariate Gaussian distribution is also isotropic with Σ = _σ_[2] _Imn_ . We denote the _ℓ_[2] vector norm as _∥·∥_ and to measure distance between matrices, we use the Frobenius norm: 

**==> picture [154 x 31] intentionally omitted <==**

which provides an upper bound on the matrix 2-norm (Petersen & Pedersen, 2012). Let _W ∈_ R _[m][×][n]_ be a set of matrix parameters where vec( _W_ ) forms a subset of the full parameter vector _x_ , typically parametrising the weights of a linear layer in a neural network. As we derive in Section B, the Gaussian ES update associated with the matrix _W_ is: 

**==> picture [342 x 45] intentionally omitted <==**

where _M_ is the mean matrix associated with _W_ , i.e. vec( _M_ ) forms a subset of _µ_ , and _P_ ( _E_ ) is a zero-mean standard normal matrix distribution: _p_ ( _E_ ) = _N_ (0 _, Im, In_ ). The gradient in Eq. (3) is estimated using the Monte Carlo estimate: 

**==> picture [184 x 30] intentionally omitted <==**

by sampling _N_ search matrices _Ei ∼ P_ ( _Ei_ ) from a standard matrix normal distribution _N_ (0 _, Im, In_ ) around the mean parameter matrix _M_ , which is updated via stochastic gradient ascent: 

**==> picture [118 x 13] intentionally omitted <==**

## **3 Related Work** 

## **3.1 Evolutionary Algorithms** 

Evolutionary algorithms have long been a compelling alternative to backpropagation-based training methods (e.g., genetic algorithms (Such et al., 2018) or symbolic evolution (Koza, 1994)). Much research in evolution has focused on developing algorithms for deep learning that scale well to distributed parallel computation (Jaderberg et al., 2017; Hansen & Ostermeier, 2001b; Salimans et al., 2017). These approaches have increased in popularity following the application of ES to policy learning in deep RL environments (Salimans et al., 2017). Since then, evolution has been widely applied in other domains, such as meta-learning (e.g., (Lu et al., 2022; Metz et al., 2022; Lange et al., 2023; Goldie et al., 2024; 2025)), hyperparameter tuning (e.g., (Parker-Holder et al., 2021; Tani et al., 2021; Vincent & Jidesh, 2023)), and drug discovery (Towers et al., 2025). ES has also enabled the development of neural network architectures that are unsuitable for backpropagation, such as activation-free models that exploit floating point rounding error as an implicit nonlinearity (Foerster, 2017). Here, we consider how to apply ES at a scale beyond the small networks and population sizes of prior work. For example, Salimans et al. (2017) use a maximum population size of 1440, whereas we use over a million. 

While low-rank structures have been used in prior evolutionary algorithms, they have been applied to different ends, with different trade-offs, relative to EGGROLL. Choromanski et al. (2019) use a low-rank search space found via principal component analysis, which provides a better search direction to more efficiently use small populations. Garbus & Pollack (2025) optimise a low-rank factorisation instead of the full dense matrix with neuroevolution, achieving similar computational gains to EGGROLL but is limited to the low-rank structure regardless of population size. 

5 

## **3.2 Evolution Strategies for LLMs** 

Although gradient backpropagation is typically used for LLM training and fine-tuning, prior work explores ES variants for fine-tuning. In particular, Zhang et al. (2024)’s two-point zeroth-order gradient estimator, which can be viewed as an ES-inspired method using a single perturbation direction and two function queries per update, is used by Malladi et al. (2023) for memory-efficient LLM fine-tuning. Yu et al. (2025) extend this approach by projecting perturbations to a low-rank subspace, improving convergence. Jin et al. (2024) perform ES directly on LoRA matrices. These works focus on supervised fine-tuning and report performance comparable to full fine-tuning, but do not address whether pretraining is possible with two-point zeroth-order methods; we find that large population sizes are necessary for pretraining, indicating such methods are unsuitable here. 

Recent work also explores ES in the context of LLM reasoning. Korotyshova et al. (2025) first train LoRA adapters using supervised fine-tuning (SFT) before decomposing them into fixed SVD bases alongside singular values that are trained using CMA-ES. They achieve comparable performance to GRPO (Shao et al., 2024) in significantly less wall-clock time on maths reasoning benchmarks. Qiu et al. (2025) directly use ES to optimise all LLM parameters for reasoning, with stronger performance than GRPO on the countdown reasoning task. However, both of these approaches use relatively small population sizes, on the order of a hundred unique perturbations per update, and instead collect hundreds of rollouts per perturbation to efficiently use GPUs. By contrast, our approach allows all generations to use different perturbations, such that our maximum population size per update is orders of magnitude larger (equal to the maximum inference batch size), without compromising token generation throughput. 

## **4 EGGROLL** 

We now introduce EGGROLL (Algorithm 1). A practical issue with using a low-rank matrix approximation is that its distribution and score function have no analytic solution except for degenerate cases, so in Section 4.1 we derive the EGGROLL approximate score function from the limiting high-rank Gaussian. Section 4.2 describes how to efficiently implement EGGROLL on modern hardware. 

## **4.1 Low-Rank Evolution Strategies** 

Recall the Gaussian matrix ES update from Eq. (3). Our goal is to introduce a tractable approximation to generating full-rank matrices by using low-rank matrices _AB[⊤]_ as our search matrices instead. Let _p_ ( _A_ ) and _p_ ( _B_ ) denote the distribution of _A ∈_ R _[m][×][r]_ and _B ∈_ R _[n][×][r]_ . 

**Assumption 1** (I.I.D. Sampling) **.** _Assume all elements ai,j ∈ A and bi,j ∈ B are continuous, identically and independently distributed random variables according to some zero-mean, symmetric, absolutely continuous distribution p_ 0( _·_ ) _with finite fourth-order moments and unit variance._ 

**Algorithm 1** EGGROLL( _r, α, σ, T_ max _, N_ workers) 

|**initialise**_M_ and workers with known random seeds_ς_|
|---|
|**for** _T_max timesteps**do**<br>**for**each worker_i ∈{_1_, . . . N_workers_}_in parallel**do**<br>_Ai ∼p_(_Ai_)_, Bi ∼p_(_Bi_)<br>_Ei ←_<br>1<br>~~_√_~~<br>_r AiB⊤_<br>_i_<br>_fi ←f_(_W_ =_M_ +_σEi_)|
|**end for**|
|workers share scalar ftness_fi_ with other workers<br>**for**each worker_i ∈{_1_, . . . N_workers_}_in parallel**do**<br>reconstruct_Ej_ for_j ∈{_1_, . . . N_workers_}_from_ς_<br>_M ←M_ +_α_<br>1<br>_N_Workers<br>�_N_Workers<br>_j_=1<br>_Ejfj_<br>**end for**|
|**end for**|



This assumption is easily satisfied for most perturbation distributions used by ES, including members from the set of generalised Gaussian distributions like Laplace, normal, and uniform distributions. We then form a low-rank search matrix: _E_ = ~~_√_~~ 1 _r[AB][⊤]_[.][The] ~~_√_~~ 1 _r_[scaling ensures the variance of] _[ E]_[remains bounded for] all _r_ . We denote the induced distribution of _E_ as _P_ ( _E_ ). _E_ = ~~_√_~~ 1 _r[AB][⊤]_[maps to the manifold][ M] _[r][⊂]_[R] _[m][×][n]_ of rank- _r_ matrices. Hence, the density _p_ ( _E_ ) is defined with respect to a unit volume on the manifold and cannot be defined with respect to the standard unit volume in Euclidean space. For the corresponding score function, gradients with respect to log _p_ ( _E_ ) are not defined over the usual Euclidean space. Instead, we use an approximation _S_[ˆ] ( _E_ ) : R _[m][×][n] →_ R _[m][×][n]_ for the score function, yielding our low-rank update: 

**==> picture [317 x 21] intentionally omitted <==**

In our experiments, analysis and Algorithm 1, we use a Gaussian approximate score function: 

**==> picture [245 x 13] intentionally omitted <==**

6 

which is the score function for the Gaussian distribution _N_ (0 _, Im, In_ ). This choice is motivated by two theoretical insights from Section 5. The matrix _AB[⊤]_ can be decomposed as a sum of independent, zero-mean vector outer products. Under Assumption 1, the central limit theorem applies to this sum of variables, proving that _P_ ( _E_ ) converges in distribution to a Gaussian _N_ (0 _, Im, In_ ) as rank _r_ increases, recovering the approximate Gaussian score in the limit. Secondly, we investigate the convergence of ES and EGGROLL as the number of parameters grows, proving both updates converge to a linearised form that is consistent with the EGGROLL update using the Gaussian approximate score function. 

EGGROLL is not wedded to any particular score function approximator and we derive and explore a set of mean-field approximators in Appendix D.1 as alternatives. However, our experiments show that the Gaussian approximator has the best overall performance on the tasks we consider. To optimise the ES objective using the EGGROLL update, we adapt the parallelised evolutionary strategies algorithm from Salimans et al. (2017). We make a Monte Carlo estimate of the expectation in Eq. (4) with _N_ workers samples to optimise the mean matrix parameters _M_ using (approximate) stochastic gradient ascent. This yields the Gaussian EGGROLL update: 

**EGGROLL UPDATE:** For each worker _i_ (in parallel), sample _Ai,t ∼ p_ ( _Ai,t_ ) _, Bi,t ∼ p_ ( _Bi,t_ ) and form a low-rank perturbation _Ei,t_ = ~~_√_~~ 1 _r[A][i,t][B] i,t[⊤]_[.][Update matrix parameters using:] 

**==> picture [316 x 30] intentionally omitted <==**

Here we absorb the constant _σ_[1][into the tunable learning rate] _[ α][t]_[.][As each random matrix] _[ E][i,t]_[ in Eq. (][6][) has rank] _r_ almost surely and the matrix is updated using a sum of _N_ worker such matrices, the overall EGGROLL matrix parameter update has rank min( _Nr, m, n_ ) almost surely, i.e., the overall parameter update is not restricted to be low-rank. For all experiments in Section 6, _Nr >_ min( _m, n_ ), i.e., EGGROLL parameter updates are full-rank. 

## **4.2 Hardware-Efficient Implementation** 

A key reason to use EGGROLL over standard ES is that large populations can be simulated in parallel on a GPU thanks to the low-rank perturbations. For the sake of exposition, we write equations from the perspective of a single worker, _i_ , and explain in text how this corresponds to batched GPU operations. Consider the task of computing a batched forward pass over inputs _ui ∈_ R _[d][in]_ for a linear layer with mean parameter _M ∈_ R _[d][out][×][d][in]_ . The standard forward pass is just a regular matrix multiplication, _uiM[T]_ , since _M_ is constant across all threads. In contrast, naïvely applying ES by trying to compute _ui_ ( _M_ + _σEi_ ) _[T]_ becomes a batched matrix multiplication, which is inefficient on GPUs since every element of _M_ + _σEi_ is only used in a single multiplication, yielding poor arithmetic intensity. 

However, with EGGROLL we know that _ui_ ( _M_ + _σEi_ ) _[T]_ = _uiM[T]_ + ~~_√_~~ _[σ] r_[(] _[u][i][B][i]_[)] _[A] i[T]_[, which improves arithmetic] intensity since it preserves the efficient general matrix multiplication used in batched inference while adding some additional cheap work per perturbation. In this context, the bulk of compute is spent on the efficient calculation of _uiM[T]_ using regular matrix multiplication. Meanwhile, when _r_ = 1, _uiBi_ simply becomes an inexpensive batch of _N_ vector-vector dot products of length _din_ to get a batch of _N_ scalars, which is then processed by a batched scalar-vector multiplication when multiplying by _A[T] i_[.][This decomposition is key to] efficient batched LoRA inference, such as those used by vLLM (Kwon et al., 2023), which is why EGGROLL achieves the same speeds as batched LoRA inference systems. The batched LoRA inference enables high arithmetic intensity, enabling us to saturate compute with many unique perturbations per input. Note that this is impossible with naïve ES because each perturbation requires a separate matrix-vector multiplication, setting an upper bound of 1 for arithmetic intensity regardless of population size; see Appendix F for a full derivation. We additionally optimise the update by not explicitly materialising the individual _Ei_ in the computation of � _Ni_ =1 _[E][i][f][i]_[, the key term in the Gaussian approximate score function.][In particular, when the rank is 1, we] reconstruct _A ∈_ R _[N][×][d][out]_ and _B ∈_ R _[N][×][d][in]_ and calculate the expression as (diag( _f_ ) _A_ ) _[T] B_ , a simple matrix multiplication. 

7 

## **5 Analysis** 

_Proofs for all theorems can be found in Appendices A to D_ . 

In this section, we investigate the theoretical properties of the ES and EGGROLL updates. In Section 5.1, we study the convergence properties of the general Gaussian ES update as the parameter dimension _d →∞_ , obtaining the conditions required for convergence to a linearised form. We then extend this analysis to the EGGROLL update in Section 5.2. Finally, in Section 5.3 we provide an analysis investigating the effect that increasing the rank of the EGGROLL approximation, proving convergence to the true ES update in the limit. 

## **5.1 High-Dimensional Gaussian ES** 

We first analyse the general ES update under Gaussian perturbations from Eq. (2): 

**==> picture [176 x 23] intentionally omitted <==**

where _v ∈_ R _[d]_ . In high dimensions, the Gaussian annulus theorem (Vershynin, 2018; Wegner, 2024) proves that the probability mass of standard Gaussian distributions concentrates in thin shells of radius _√d_ , which place probability mass further from the origin as dimension _d_ increases. To counter this, we let _σd_ depend on _d_ and analyse the _critical decay rate_ of _σd_ that yields convergence of the ES updates. We make the following mild regularity assumptions: 

**Assumption 2** (Locally Continuous Fitness) **.** _With probability 1 with respect to the random initialisation of µ, assume there exists a ball Bρ_ ( _µ_ ) := _{x[′] |∥x[′] − µ∥ < ρ} of fixed radius ρ >_ 0 _where f_ ( _x_ ) _is C_[1] _-continuous for all x ∈ Bρ_ ( _µ_ ) _. Within this ball, let ∇f_ ( _x_ ) _be α-Hölder continuous, i.e., ∥∇f_ ( _x_ ) _−∇f_ ( _y_ ) _∥≤ L∥x − y∥[α] for all x, y ∈ Bρ_ ( _µ_ ) _, α ∈_ (0 _,_ 1] _and L_ = _O_ (1) _._ 

Assumption 2 _does not restrict the fitness to be globally continuous_ ; with probability one with respect to the initialisation distribution there must exist an arbitrarily small _C_[1] -continuous ball around _µ_ . In particular, discontinuities, kinks, and non-differentiable regions may exist in the domain, provided they are not encountered with nonzero probability in the local region explored by the algorithm. _α_ -Hölder is the weakest simple, dimension-robust assumption that guarantees vanishing local gradient variation under Gaussian perturbations; it is weaker than Lipschitz continuity, which is recovered with _α_ = 1. 

**Assumption 3** (Global Polynomial Growth) **.** _Assume that there exists some constant_ 0 _< C < ∞ that is O_ (1) _in d and finite polynomial degree p ≥_ 0 _such that |f_ ( _µ_ + _σdv_ ) _| ≤ C_ (1 + _∥µ_ + _σdv∥[p]_ ) _and ∥∇f_ ( _µ_ + _σdv_ ) _∥≤ C_ (1 + _∥µ_ + _σdv∥[p]_ ) _almost surely under v ∼N_ (0 _, Id_ ) _._ 

Unlike Assumption 2, this is a _global_ assumption. Again, discontinuities can exist. The assumption is weaker than boundedness, is satisfied by essentially all fitness functions used in ES, and ensures that both the objective and its gradient are integrable under Gaussian perturbations; objectives violating this condition typically exhibit super-polynomial growth and derivative growth, which leads to ill-defined or highly unstable ES updates. Moreover, if the condition is not satisfied almost surely, then the function and its gradients are undefined in regions that have nonzero Gaussian measure. 

**Assumption 4** (Bounded Derivative) **.** _With probability 1 with respect to the random initialisation of µ, assume that ∥µ∥_ = _O_ (1) _and ∥∇f_ ( _µ_ ) _∥_ = _O_ (1) _, i.e. ∥µ∥ and ∥∇f_ ( _µ_ ) _∥ do not grow with increasing d._ 

This assumption is standard in high-dimensional analysis proving convergence to linearity, as proving convergence to _∇f_ ( _µ_ ) becomes meaningless if _∥∇f_ ( _µ_ ) _∥→∞_ . Moreover, the ES update as a whole can diverge if Assumption 4 is not satisfied. It can be ensured by scaling, typically by scaling networks parameters by _d[−]_ 2[1] or using an appropriate scaled initialisation, commonly Gaussian initialisation _µ ∼N_ �0 _, d_[1] _[I][d]_ �. This is precisely the scaling employed in the neural tangent kernel (NTK) regime (Jacot et al., 2018; Lee et al., 2019; Chizat et al., 2019), where it guarantees dimension-independent gradients and stable training dynamics. 

These assumptions encompass essentially all objectives encountered in modern machine learning, including networks with finitely many ReLU activations, max- and hinge-based losses, and other piecewise-smooth or discontinuous models. Our first theorem proves convergence of a Gaussian ES update to a linearised form, that is to the local first-order derivative _∇f_ ( _µ_ ), with a tight convergence rate for any function satisfying these assumptions: 

8 

**Theorem 1** (Convergence to Linearity) **.** _Let Assumptions 2, 3, and 4 hold and σd_ = _o d[−]_ 2[1] _. Then:_ � � _α ∥∇µJ_ ( _θ_ ) _−∇f_ ( _µ_ ) _∥_ = Θ _σd√d_ = _o_ (1) _, almost surely with respect to the distribution over µ._ �� � � 

To understand the effect that breaching the _σd_ = _o d[−]_ 2[1] rate has on the convergence of Gaussian ES, we � � study the space of functions that can be represented by cubic polynomials of the form: 

**==> picture [297 x 21] intentionally omitted <==**

where _a ∈_ R _[d]_ , _B ∈_ R _[d][×][d]_ is a symmetric matrix and _C_ ( _x, x, x_ ) =[�] _i,j,k[c][i,j,k][x][i][x][j][x][k]_[denotes a symmetric] 3-linear map represented by the symmetric 3-tensor _C ∈_ R _[d][×][d][×][d]_ , which generalises cubic equations of the form _f_ ( _x_ ) = _ax_ + _bx_[2] + _cx_[3] to vector-valued _x_ . These are non-pathological, well-behaved, analytic _C[∞]_ -continuous functions, and include a rich subclass of convex optimisation problems, for instance, cubic perturbations of strictly convex quadratics. Moreover, any convex _C_[3] -continuous objective admits a local third-order Taylor expansion of this form around a minimiser. 

**Theorem 2** (Exact Divergence for Cubic Objectives) **.** _Let f_ ( _x_ ) _denote the cubic polynomial in Eq._ (7) _. Assume ∥a∥_ = _O_ (1) _,∥B∥_ = _O_ (1) _, ∥C∥_ = _O_ (1) _where ∥·∥ denotes operator norm for i-tensor T_ ( _x_ 1 _, . . . xi_ ) _: ∥T ∥_ := sup _∥x_ 1 _∥_ = _···_ = _∥xi∥_ =1 _|T_ ( _x_ 1 _, . . . xi_ ) _|. Let Assumption 4 hold, then:_ 

**==> picture [194 x 23] intentionally omitted <==**

_Moreover:_ 

**==> picture [298 x 42] intentionally omitted <==**

Together, Theorems 1 and 2 prove Gaussian ES has a _critical convergence rate_ of _σd_ = _o d[−]_ 2[1] in high � � dimensions, and operates in three regimes: 

**Regime I (Convergence to Linearity):** For _σd_ = _o d[−]_ 2[1] , ES converges to a linearised form, recovering � � a local first-order gradient update _∇f_ ( _µ_ ). This result is _analogous to neural tangent kernel_ (NTK) type theorems, which prove that neural networks linearise in high dimensions (Jacot et al., 2018) and results from the concentration of the population distribution as _d →∞_ , but applies to a more general set of objectives including discontinuous architectures. Moreover, Theorem 1 proves that the ( _σd√d_ ) _[α]_ rate at which Gaussian ES converges is tight and cannot in general be improved upon without strengthening continuity or introducing specific structure into the objective to ensure the Hölder constant _L_ decays with _d_ ; for the class of cubic functions we consider in Theorem 2, the faster _σd_[2] _[d]_[ convergence rate found in Eq. (][9][) is possible due to the] _C[∞]_ -continuity of this function class, which means the converge rate is governed by third order derivative terms. 

**Regime II (Critical):** For _σd ≍ d[−]_[1] 2 , Gaussian ES converges to a nonlinear limiting update that may retain higher-order derivative terms when they exist; for our cubic example, Eq. (8) proves that at this critical rate, the second-order term associated with the matrix _B_ vanishes due to symmetry and the third-order term associated with the tensor _C_ remains: 

**==> picture [152 x 26] intentionally omitted <==**

As the polynomial form is representative of general Taylor expansions, this implies that the limiting high dimensional update retains third-order derivatives (and higher order odd derivatives) as _d →∞_ . 

9 

**Regime III (Divergence):** For _d[−]_ 2[1] = _o_ ( _σd_ ), Theorem 2 shows that there exist smooth cubic objectives with bounded coefficients for which: 

**==> picture [116 x 12] intentionally omitted <==**

In particular, divergence occurs whenever the cubic tensor has a non-vanishing Gaussian contraction (equivalently, non-zero partial trace), i.e. in non-degenerate cases; only in the exceptional trace-free case does the cubic contribution vanish. 

In practice, _σd_ is often absorbed into the ES update stepsize, and its scale is adjusted automatically as part of the hyperparameter regime to ensure stability. 

## **5.2 High-Dimensional EGGROLL** 

We now extend our high-dimensional analysis to study the EGGROLL update using the Gaussian approximate score function ˆ _g_ LR from Eq. (5). Taking _r_ as fixed, we consider the Gaussian matrix ES setting outlined in Section 2.3. We take _x_ = Vec( _W_ ) where _W ∈_ R _[m][×][n]_ and analyse the effect of increasing the total number of matrix parameters _d_ = _mn_ . Recall the true ES Gaussian matrix update is: 

**==> picture [198 x 22] intentionally omitted <==**

where _M_ is the set of mean matrix parameters associated with the matrix _W_ and _P_ ( _E_ ) is a zero-mean standard normal _p_ ( _E_ ) = _N_ (0 _, Im, In_ ). 

Two key differences between full-rank Gaussian ES and EGGROLL are that ˆ _g_ LR is an approximation to a true gradient and _P_ ( _E_ ) may have heavier tails than a Gaussian. To account for these differences, we require a slightly stricter local continuity control assumption: 

**Assumption 5** (EGGROLL Locally Continuous Fitness) **.** _With probability 1 with respect to the random initialisation of µ, assume there exists a ball Bρ_ ( _µ_ ) := _{x[′] |∥x[′] − µ∥ < ρ} of fixed radius ρ >_ 0 _where f_ ( _x_ ) _is C_[2] _-continuous for all x ∈ Bρ_ ( _µ_ ) _and ∥∇_[2] _f_ ( _µ_ ) _∥ be polynomial bounded in d. Within this ball, let ∇_[2] _f_ ( _x_ ) _be Lipschitz continuous, i.e. ∥∇_[2] _f_ ( _x_ ) _−∇_[2] _f_ ( _y_ ) _∥≤ Ld∥x − y∥ for all x, y ∈ Bρ_ ( _µ_ ) _._ 

This assumption still permits discontinuous objectives. We also assume that _p_ 0( _·_ ) generates sub-Gaussian elements with uniform tail control: 

**Assumption 6** (Sub-Gaussian Tails) **.** _In addition to Assumption 1, assume that p_ 0( _·_ ) _generates variables that have sub-Gaussian tails, i.e. for xi ∼ p_ 0( _xi_ ) _:_ 

**==> picture [116 x 12] intentionally omitted <==**

_for some_ 0 _≤ C < ∞ that does not depend on d._ 

We discuss sub-Gaussian variables and their properties in Section C.3 The assumption is trivially satisfied for Gaussian distributions _a ∼N_ (0 _, Im_ ) and _b ∼N_ (0 _, In_ ), and holds more generally, for example for bounded distributions, uniform distributions and generalised Gaussian distributions with shape parameter greater than two. This flexibility is particularly relevant for the models in Section 6.1, where heavier-shouldered distributions may be preferred over the Gaussian. 

**Theorem 3** (EGGROLL Convergence to Linearity) **.** _Let W ∈_ R _[m][×][n] , d_ = _mn and x_ = _Vec_ ( _W_ ) _. Let Assumptions 3, 4, 5 and 6 hold, σd_ = _o_ ( _d[−]_[1] _[/]_[2] ) _, and Ld_ ( _σdd_ )[2] = _o_ (1) _. Then there exists some K >_ 0 _such that:_ 

**==> picture [389 x 30] intentionally omitted <==**

_and_ 

**==> picture [339 x 19] intentionally omitted <==**

_almost surely with respect to the distribution over µ._ 

10 

Our theory explains the success of EGGROLL in high dimensions with rank as small as _r_ = 1; Eq. (11) proves EGGROLL converges to the true update matrix ES update _∇M J_ ( _θ_ ) as _d →∞_ regardless of _r_ . In addition, Eq. (10) proves that under the same conditions, the EGGROLL update also linearises like the true Gaussian ES update analysed in Section 5.1, recovering a local first-order derivative as _d →∞_ . For high-dimensional neural networks, standard parametrisations place training in the NTK regime, in which the network behaves approximately linearly in its parameters and gradient descent converges to a global minimum (Jacot et al., 2018; Lee et al., 2019; Chizat et al., 2019). Recent results show that the spectral norm of the Hessian decays polynomially with width, and that higher-order derivatives governing the variation of the Hessian also vanish (Liu et al., 2020). Consequently, the Lipschitz constant _Ld_ = _o_ (1), typically at rate _d[−]_[1] 2 or _d[−]_[1] depending on the network architecture. Substituting these rates into our upper bound in Eq. (10) yields convergence rates of 3 _O_ ( _σd_[2] _[d]_ 2 ) or _O_ ( _σd_[2] _[d]_[)][ respectively.] 

## **5.3 Rank Analysis** 

We now analyse how fast the low-rank update from Eq. (4) with Gaussian score approximation converges to the true Gaussian ES matrix gradient in Eq. (3) as the rank of the update _r_ increases. We make notation explicit in _r_ in this subsection, for example writing _E[r]_ = ~~_√_~~ 1 _r[A][r][B][r][⊤]_[.][We introduce the following formal] regularity assumption for the fitness function: 

**Assumption 7** (Bounded Fitness) **.** _Assume that f_ ( _W_ ) _is bounded, that is_ sup _W |f_ ( _W_ ) _| < ∞._ 

Our key theoretical result characterises the error rate between the Gaussian score approximator in the low-rank update ˆ _g_ LR _[r]_[from Eq. (][4][) and the true gradient using the matrix Frobenius norm:] **Theorem 4** (EGGROLL Rank Convergence) **.** _Let Assumptions 1 and 7 hold, then:_ 

**==> picture [281 x 13] intentionally omitted <==**

The convergence rate in Eq. (12) is faster than the r=1 typical _O r[−]_[1] 2 rate dictated by the general para0.4 ~~r=2~~ r=3 metric central limit theorem.� � Our analysis shows that 0.2 r=5r=10 this is due to the symmetry in our problem under Asr=50r=100 sumption 1. To obtain our results, we make an Edge0.0 r worth expansion (Bhattacharya & Ranga Rao, 1976) of the distribution _P_ ( _E[r]_ ), which expands _P_ ( _E[r]_ ) 0.2 as the limiting Gaussian distribution plus a sum of decaying terms that are controlled by the 3rd order 0.4 and higher cumulants of _P_ ( _E[r]_ ). Each _i_ th order cu3 2 1 0 1 2 3 mulant term is multiplied by a factor that decays at Ei, j rate _O_ � _r[−][i][−]_ 2[2] �. For symmetric zero-mean distribuFigure 3: Plot of Marginal Score Multiplied by Density for tions, all odd cumulants are zero (for the same reason Increasing _r_ that all odd moments of a symmetric distribution are zero). Hence, the rate of convergence to the limiting distribution is controlled by the 4th order term, which has rate _O_ � _r[−]_[1][�] . 

Figure 3: Plot of Marginal Score Multiplied by Density for Increasing _r_ 

Although the full distribution _P_ ( _E[r]_ ) has no general closed-form solution, the distribution over marginals _P_ ( _Ei,j_ ) is more amenable to analysis. We derive the density of the marginal distribution _P_ ( _Ei,j_ ) for generalised Gaussian distributed _ai,j_ and _bi,j_ in Section D.1. To illustrate the fast convergence rate, we plot the negative density _×_ score function _p_ ( _Ei,j_ ) _Ei,j_ for the marginal density _p_ ( _Ei,j_ ) in Fig. 3 using Gaussian distributed _ai,j_ and _bi,j_ (see Theorem 6 for a derivation). The figure shows that _p_ ( _Ei,j_ ) _Ei,j_ quickly converges 2 to the limiting function ~~_√_~~ _[E][i]_ 2 _[,j] π_[exp] � _−[E][i]_ 2 _[,j]_ �, recovering the Gaussian form from the true Gaussian ES update. Even at _r_ = 1, the function is not a poor approximation. After _r_ = 10, the function has nearly converged and after _r_ = 50, the function is visually indistinguishable from the limit, providing evidence for the hypothesis that the low-rank approximation is accurate even for very low-rank regimes _r ≪_ min( _m, n_ ). 

11 

**==> picture [428 x 137] intentionally omitted <==**

**----- Start of picture text -----**<br>
Tabula Rasa Reinforcement Learning Countdown — RWKV 7g1.5B<br>1.0<br>0.3<br>0.8<br>0.6<br>0.2<br>0.4<br>0.1<br>0.2<br>0.0 0.0<br>0 1 2 3 4 5 0 1 2 3 4 5 6 7 8<br>Steps 1e8 Relative wall-clock time (hours)<br>EGGROLL (envs=16) OpenES (envs=16) PPO GRPO (n=3) EGGROLL (n=3)<br>Validation Score<br>Normalized Return<br>**----- End of picture text -----**<br>


Figure 4: (a) Comparison of reinforcement learning returns normalised by PPO performance across 16 environments for 10 seeds. The shaded region is the standard error of the mean.(b) Validation score of 3 seeds of EGGROLL v.s. 3 seeds of GRPO in countdown task with an RWKV 7g1.5B model on a single GPU. EGGROLL allows 1024 parallel generations per GPU (618 updates) whereas GRPO only 64 (915 updates). 

## **6 Experiments** 

In the following section we showcase the effectiveness of EGGROLL in a variety of tasks that position it as a strong alternative to back-propagation for the end-to-end training of foundation models. 

## **6.1 Pure Integer Language Model Pretraining** 

To demonstrate the potential of EGGROLL as a general optimisation method, we apply it to language model pretraining. Since EGGROLL does not rely on gradients, we explicitly design a language model architecture to be efficient and hardware-friendly at inference time. To highlight EGGROLL’s flexibility, we train a nonlinear recurrent neural network (RNN) in pure integer datatypes with no explicit activation functions, relying only on the implicit nonlinearity of clipping in int8 operations. We call the resulting language model EGG, the Evolved Generative GRU, an EGGROLL-friendly architecture with all weights in int8. See Appendix G for more details on the architecture and motivation behind EGG. 

We train an EGG model with 6 layers and hidden dimension 256 (6L-256D) to do character-level prediction on the minipile dataset (Kaddour, 2023). We update parameters after 100 tokens for each population member, applying truncated ES by keeping the hidden state and only resetting at document boundaries. We plot the test loss in Fig. 2b over training steps across a range of population sizes with a fixed data batch size of 16 sequences per step, where the best test loss is 3.40 bits/byte. With a sufficiently large population size, EGG outperforms a dense 6L-256D Transformer trained with backprop SGD using the same data batch size. Note that larger population sizes require more parallel compute for the same amount of data; our largest population size of 2[20] = 1048576 requires around 180 times more GPU-hours than the backprop baseline, demonstrating the potential for compute-only scaling in limited data regimes using EGGROLL. 

Moreover, our largest population size of 2[20] is three orders of magnitude larger than the largest experiment done by Salimans et al. (2017) while only requiring a single GPU to train, highlighting EGGROLL’s computational efficiency. We note that large population sizes are critical for pretraining; a population size of 2, analogous to MeZO (Malladi et al., 2023), significantly underperforms larger population sizes despite having access to the same data batch. We conduct more ablations in Appendix I, analysing the tradeoff between population size and data batch size. 

## **6.2 Reinforcement Learning Tasks** 

To verify that low-rank perturbations do not change the optimisation behavior of ES in standard control settings, we benchmark EGGROLL against OpenES (Salimans et al., 2017) across 16 tabula rasa environments spanning Navix, Craftax, Brax, Kinetix, and Jumanji. We use a fixed 3-layer MLP policy (256 hidden units) and perform per-environment hyperparameter optimisation for each method before evaluating the selected configuration over 10 random seeds, reporting mean performance (normalised by PPO) and uncertainty. Overall, EGGROLL is competitive with OpenES on 7/16 environments, underperforms on 2/16, and outperforms on 7/16, while often delivering substantial wall-clock improvements due to its batched low-rank structure (full environment 

12 

**==> picture [430 x 138] intentionally omitted <==**

**----- Start of picture text -----**<br>
GSM8K — RWKV 7g7B Math Reasoning — RWKV 7g7B<br>0.150<br>0.13 0.13 0.13<br>0.80<br>0.125<br>0.75 0.100<br>0.70 0.075 0.07 0.07<br>0.050<br>0.65 0.03<br>0.025<br>0.60<br>0.000<br>0 2 4 6 8 10 AIME24 AIME25<br>Relative wall-clock time (hours)<br>GRPO (n=3) EGGROLL (n=3) Base model GRPO EGGROLL<br>Validation Score Validation Score<br>**----- End of picture text -----**<br>


Figure 5: (a) Comparison of the validation score of 3 seeds of EGGROLL v.s. 3 seeds of GRPO in GSM8K task with an RWKV 7g7B model on 8 GPUs. EGGROLL allows 8192 parallel generations (1024 per GPU with 260 updates) whereas GRPO only 256 (32 per GPU with 340 updates). (b) Performance of our finetuned RWKV 7G 7 billion model on hard reasoning tasks using 128 GPUs for 12 hours. The model was trained using the DeepScaleR dataset and the best checkpoint was chosen by evaluating on AIME24. 

list, learning curves, timing comparisons, and complete HPO ranges/settings are provided in Appendix N.4). Figure 4a shows the averaged normalised return across the 16 environments with 10 seeds per environment. We additionally report MARL results in Section N.1. 

## **6.3 Foundation Model Fine-tuning** 

We apply EGGROLL to finetune an RWKV-7 (Peng et al., 2025) LLM on two reasoning tasks: countdown (Gandhi et al., 2024) and GSM8K (Cobbe et al., 2021). RWKV is a recurrent model that is better suited to parallelisation than transformers because any memory otherwise spent on the KV cache is used to evaluate population members. Figure 4b shows that EGGROLL fine-tuning on an RWKV-7 1.5B model converges to a higher validation accuracy of 35% (vs. 23%) given the same hardware and wall-clock time in the countdown task. Similarly, Figure 5a shows that EGGROLL outperforms GRPO on GSM8K fine-tuning. Our scoring function draws parallels to the group relative advantage of GRPO. In particular, to score a set of noise directions, _E ≡{E_ 1 _, . . . , En}_ , we first compute their accuracies, _{s_ 1 _,qi, . . . , sn,qi}_ , on _|q|_ = _m_ questions, creating a matrix of scores _S ∈_ R _[m][×][n]_ . We then compute the normalised score per question, with the main difference that we use the global variance _σ_ ¯, and average over all the questions to compute a score for the noise direction _Ei_ : 

**==> picture [156 x 30] intentionally omitted <==**

This scoring function weights all questions within the same batch the same across population members. We use this recipe to train a 14 billion parameter RWKV 7 model on the DeepScaleR dataset and evaluate in more challenging maths reasoning tasks. In this regime, GRPO is infeasible due to the extra memory used by the Adam optimiser Kingma & Ba (2014). Using a thinking budget of 5000 tokens for training and evaluation, our fine-tuned 14B model improves from 13% to 30% accuracy on AIME24, from 7% to 33% accuracy on AIME25 and from 11% to 13% accuracy on HMMT25 after training on 32 GPUs for 12 hours (Figure 13b). On 7B models, we outperform GRPO using 128 GPUs for 24 hours (Figure 5b). 

In Section L, we achieve similar performance to GRPO when fine-tuning Qwen Transformer models, and additionally demonstrate that EGGROLL can directly optimise for pass@k, a known limitation of GRPO (Yue et al., 2025). Beyond language models, we also fine-tune a finance world model into an agent for high-frequency trading that directly optimises for PnL; see Section M for more details. 

## **6.4 Fine-tuning Integer Quantised LLMs** 

We follow the same procedure as Jacob et al. (2017) to quantise the RWKV-7 family of models by dividing by the maximum _per-channel_ value on each weight matrix and mapping into the int8 range of [ _−_ 127 _,_ 127]. We then apply EGGROLL with Adam to do model distillation from the original, non-quantised RWKV-7, into the resulting int8 quantised model using examples from GSM8K. See Appendix K for full details about the 

13 

**==> picture [430 x 140] intentionally omitted <==**

**----- Start of picture text -----**<br>
Quantised Distill — RWKV 7g7B Quantised Distill — RWKV 7g7B<br>40<br>0.025<br>30 0.020<br>0.015<br>20<br>0.010<br>10<br>0.005<br>0.000<br>0<br>0 20 40 60 80 100 0 20 40 60 80 100<br>Epoch Epoch<br>Quantised EGGROLL (n=3) Original (n=3) EGGROLL (n=3) Baseline (n=3)<br>Perplexity<br>Validation Score<br>**----- End of picture text -----**<br>


Figure 6: (a) Average per token perplexity (during training) of 3 seeds of a quantised (int8) RWKV 7G 7 billion parameter model on distillation from the non quantised model using examples from GSM8K. (b) Validation score on unseen examples of GSM8K of 3 seeds of a quantised RWKV 7G 7 billion parameter model. Initially the model is unable to solve any problems, but progressively it is capable of solving more problems. The baseline here indicates the validation score of a quantised model without any further training. 

specifics of quantisation and fine-tuning. The distillation is done by matching the distributions between the quantised and non-quantised models on teacher forced examples (with solutions) from the GSM8K dataset. More specifically, the fitness for a given set of parameters, _µi_ , is computed as follows: 

**==> picture [144 x 30] intentionally omitted <==**

where _x_ 1: _T_ is a subsequence of tokens taken from the solutions of GSM8K and KL ( _pt||qt_ ( _·_ ; _µi_ )) is the Kullback-Leibler divergence between the distribution of the non-quantised model, _pt_ , and the distribution of the quantised model _qt_ over the vocabulary at token _t_ . Figure 6a shows the average per token perplexity of 3 seeds of a quantised RWKV 7G 7 billion parameter model compared to that of the original non-quantised model over the same sequence, as a baseline. Progressively, the quantised model recovers the capability to solve a subset of the GSM8K dataset (Figure 6b). 

## **7 Conclusion** 

We introduce EGGROLL, a powerful method for black-box optimisation that scales evolutionary strategies to billion-parameter models and beyond using low-rank search matrices. Our experiments demonstrate that EGGROLL is effective with a rank of 1, giving substantial computational and memory savings for negligible decrease in performance when compared to the full-rank perturbations. Empirically, EGGROLL delivers large speedups over naïve ES in tabula rasa and multi-agent RL, and can power end-to-end training pipelines for foundation models. Our theoretical analysis shows that the EGGROLL update converges towards the Gaussian ES update with increasing rank _r_ and parameter dimension _d_ = _mn_ , and we provide a rigorous study of general ES at high dimensions, deriving necessary and sufficient conditions for convergence and linearisation. 

Looking forward, we can use EGGROLL for other problems beyond the reach of modern first-order gradientbased techniques. In particular, EGGROLL can enable the training of large scale end-to-end neurosymbolic systems (Sarker et al., 2021) with non-differentiable components. For instance, we can train neural networks that interface with symbolic modules for specific functions, like memory or calculations. We can also optimise end-to-end systems of language models, training them to be aware of inference-time harnesses and interactions with other agents in complex systems. 

## **Acknowledgements** 

Compute for this project is graciously provided by the Isambard-AI National AI Research Resource, under the projects “FLAIR 2025 Moonshot Projects” and “Robustness via Self-Play RL.” Some experiments also used compute generously given by JASMIN, the UK’s collaborative data analysis environment (https: //www.jasmin.ac.uk). 

14 

Bidipta Sarkar is supported by the Clarendon Fund Scholarship in partnership with a Department of Engineering Science Studentship for his Oxford DPhil. Mattie Fellows is funded by a generous grant from the UKRI Engineering and Physical Sciences Research Council EP/Y028481/1. Juan Agustin Duque is supported by the St-Pierre-Larochelle Scholarship at the University of Montreal and by Aaron Courville’s CIFAR AI Chair in Representations that Generalize Systematically. Jarek Liesen and Theo Wolf are supported by the EPSRC Centre for Doctoral Training in Autonomous Intelligent Machines & Systems EP/Y035070/1. Jarek Liesen is also supported by Sony Interactive Entertainment Europe Ltd. Uljad Berdica is supported by the EPSRC Centre for Doctoral Training in Autonomous Intelligent Machines & Systems EP/S024050/1 and the Rhodes Scholarship. Lukas Seier is supported by the Intelligent Earth CDT with funding from the UKRI grant number EP/Y030907/1. Alexander D. Goldie is funded by the EPSRC Centre for Doctoral Training in Autonomous Intelligent Machines and Systems EP/S024050/1. Jakob Nicolaus Foerster is partially funded by the UKRI grant EP/Y028481/1 (originally selected for funding by the ERC). Jakob Nicolaus Foerster is also supported by the JPMC Research Award and the Amazon Research Award. 

We thank Andreas Kirsch for discovering an emergent log-linear scaling law for EGG loss with respect to int8 OPs in this tweet along with other community members for their comments and recommendations during the first arXiv release of this work. 

## **References** 

- Agentica Organization, Michael Luo, Sijun Tan, and Justin Wong. Deepscaler-preview-dataset. https:// huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset, 2025. Accessed: 2025-01-14. 

- R. Askey and R. (eds.) Roy. Nist digital library of mathematical functions, chapter 5: Gamma function. Online: https://dlmf.nist.gov/5, 2020-2026. Section 5.11 (Stirling / asymptotic expansions), release 1.1.16. 

- Anne Auger and Nikolaus Hansen. Theory of evolution strategies: A new perspective. In Anne Auger and Benjamin Doerr (eds.), _Theory of Randomized Search Heuristics: Foundations and Recent Developments_ , pp. 289–325. World Scientific, Singapore, 2011. 

- Mislav Balunovi´c, Jasper Dekoninck, Ivo Petrov, Nikola Jovanovi´c, and Martin Vechev. Matharena: Evaluating llms on uncontaminated math competitions, 2026. URL https://arxiv.org/abs/2505.23281. 

- A. B. Basset. _A Treatise on Hydrodynamics: with numerous examples_ , volume 2. Deighton, Bell, and Co., Cambridge, UK, 1888. 

- Yoshua Bengio, Réjean Ducharme, and Pascal Vincent. A neural probabilistic language model. In T. Leen, T. Dietterich, and V. Tresp (eds.), _Advances in Neural Information Processing Systems_ , volume 13. MIT Press, 2000. URL https://proceedings.neurips.cc/paper_files/paper/2000/ file/728f206c2a01bf572b5940d7d9a8fa4c-Paper.pdf. 

- Hans-Georg Beyer. Toward a theory of evolution strategies: Self-adaptation. _Evolutionary Computation_ , 3: 311–347, 1995. URL https://api.semanticscholar.org/CorpusID:17416734. 

- Hans-Georg Beyer and Hans-Paul Schwefel. Evolution strategies –a comprehensive introduction. _Natural Computing_ , 1(1):3–52, 2002. 

- R. N. Bhattacharya and R. Ranga Rao. _Normal approximation and asymptotic expansions_ . Wiley series in probability and mathematical statistics. Wiley, New York, 1976. ISBN 047107201X. 

- Clément Bonnet, Daniel Luo, Donal Byrne, Shikha Surana, Sasha Abramowitz, Paul Duckworth, Vincent Coyette, Laurence I. Midgley, Elshadai Tegegn, Tristan Kalloniatis, Omayma Mahjoub, Matthew Macfarlane, Andries P. Smit, Nathan Grinsztajn, Raphael Boige, Cemlyn N. Waters, Mohamed A. Mimouni, Ulrich A. Mbou Sob, Ruan de Kock, Siddarth Singh, Daniel Furelos-Blanco, Victor Le, Arnu Pretorius, and Alexandre Laterre. Jumanji: a diverse suite of scalable reinforcement learning environments in jax, 2024. URL https://arxiv.org/abs/2306.09884. 

15 

- Jean-Philippe Bouchaud, Julius Bonart, Jonathan Donier, and Martin Gould. _Trades, quotes and prices: financial markets under the microscope_ . Cambridge University Press, 2018. 

- James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/jax-ml/jax. 

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020. URL https://arxiv.org/abs/2005.14165. 

- Lénaïc Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), _Advances in Neural Information Processing Systems_ , volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper_files/paper/2019/file/ ae614c557843b1df326cb29c57225459-Paper.pdf. 

- Krzysztof M Choromanski, Aldo Pacchiano, Jack Parker-Holder, Yunhao Tang, and Vikas Sindhwani. From complexity to simplicity: Adaptive es-active subspaces for blackbox optimization. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), _Advances in Neural Information Processing Systems_ , volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper_files/paper/2019/file/ 88bade49e98db8790df275fcebb37a13-Paper.pdf. 

- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sashank Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways. _J. Mach. Learn. Res._ , 24(1144), 2023. URL https://jmlr.org/papers/volume24/22-1144/22-1144.pdf. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- A. P. Dawid. Some matrix-variate distribution theory: Notational considerations and a bayesian application. _Biometrika_ , 68(1):265–274, 1981. ISSN 0006-3444. 

- Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten P. Bosma, Zongwei Zhou, Tao Wang, Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc Le, Yonghui Wu, Zhifeng Chen, and Claire Cui. Glam: Efficient scaling of language models with mixture-of-experts. In _Proceedings of the 39th International Conference on Machine Learning_ , volume 162 of _Proceedings of Machine Learning Research_ , pp. 5547–5569, Jul 2022. URL https://proceedings.mlr.press/v162/du22c.html. 

- William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: scaling to trillion parameter models with simple and efficient sparsity. _J. Mach. Learn. Res._ , 23(1):1–39, January 2022. ISSN 1532-4435. URL https://jmlr.org/papers/volume23/21-0998/21-0998.pdf. 

16 

Maxim Fishman, Brian Chmiel, Ron Banner, and Daniel Soudry. Scaling fp8 training to trillion-token llms, 2025. URL https://arxiv.org/abs/2409.12517. 

Jakob Nicolaus Foerster. Nonlinear computation in deep linear networks, sep 2017. URL https://blog. openai.com/nonlinear-computation-in-linear-networks/. Accessed: 2025-11-20. 

- Gerald B. Folland. _Real Analysis: Modern Techniques and Their Applications_ . John Wiley & Sons, New York, 2nd edition, 1999. See Theorem 8.22 (Riemann–Lebesgue Lemma). 

- Catherine Forbes, Merran Evans, Nicholas Hastings, and Brian Peacock. _Statistical Distributions_ . Wiley Series in Probability and Statistics. John Wiley & Sons, Hoboken, NJ, USA, 4th edition, 2011. ISBN 9780470390634. 

- C. Daniel Freeman, Erik Frey, Anton Raichuk, Sertan Girgin, Igor Mordatch, and Olivier Bachem. Brax – a differentiable physics engine for large scale rigid body simulation, 2021. URL https://arxiv.org/ abs/2106.13281. 

- Sascha Yves Frey, Kang Li, Peer Nagy, Silvia Sapora, Christopher Lu, Stefan Zohren, Jakob Foerster, and Anisoara Calinescu. Jax-lob: A gpu-accelerated limit order book simulator to unlock large scale reinforcement learning for trading. In _Proceedings of the Fourth ACM International Conference on AI in Finance_ , pp. 583–591, 2023. 

- Kevin Galim, Wonjun Kang, Yuchen Zeng, Hyung Il Koo, and Kangwook Lee. Parameter-efficient fine-tuning of state space models, 2025. URL https://arxiv.org/abs/2410.09016. 

- Matteo Gallici, Mattie Fellows, Benjamin Ellis, Bartomeu Pou, Ivan Masmitja, Jakob Nicolaus Foerster, and Mario Martin. Simplifying deep temporal difference learning. In _The Thirteenth International Conference on Learning Representations_ , 2025. URL https://openreview.net/forum?id=7IzeL0kflu. 

- Kanishk Gandhi, Denise Lee, Gabriel Grand, Muxin Liu, Winson Cheng, Archit Sharma, and Noah D. Goodman. Stream of search (sos): Learning to search in language, 2024. URL https://arxiv.org/ abs/2404.03683. 

- Jack Garbus and Jordan Pollack. Low rank factorizations are indirect encodings for deep neuroevolution. In _Proceedings of the Genetic and Evolutionary Computation Conference Companion_ , GECCO ’25 Companion, pp. 2371–2379, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400714641. doi: 10.1145/3712255.3734297. URL https://doi.org/10.1145/3712255.3734297. 

- Alexander D. Goldie, Chris Lu, Matthew T. Jackson, Shimon Whiteson, and Jakob N. Foerster. Can Learned Optimization Make Reinforcement Learning Less Difficult? In _Advances in Neural Information Processing Systems_ , volume 37, pp. 5454–5497, 2024. 

- Alexander David Goldie, Zilin Wang, Jaron Cohen, Jakob Nicolaus Foerster, and Shimon Whiteson. How Should We Meta-Learn Reinforcement Learning Algorithms? May 2025. URL https://openreview. net/forum?id=jKzQ6af2DU. 

Ian Goodfellow, Yoshua Bengio, and Aaron Courville. _Deep Learning_ . MIT Press, 2016. http://www. deeplearningbook.org. 

- Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K.Q. Weinberger (eds.), _Advances in Neural Information Processing Systems_ , volume 27. Curran Associates, Inc., 2014. URL https://proceedings.neurips.cc/paper_ files/paper/2014/file/f033ed80deb0234979a61f95710dbe25-Paper.pdf. 

- Martin D Gould, Mason A Porter, Stacy Williams, Mark McDonald, Daniel J Fenn, and Sam D Howison. Limit order books. _Quantitative Finance_ , 13(11):1709–1742, 2013. 

17 

- I. S. (Izrail Solomonovich) Gradshte˘ın, I. M. (Iosif Moiseevich) Ryzhik, Daniel Zwillinger, Victor Moll, and Inc Scripta Technica. _Table of integrals, series, and products_ . Academic Press, San Diego ; Tokyo, 8 edition, 2015. ISBN 0123849330. 

- G R Grimmett and D R Stirzaker. Probability and random processes. _Journal of the Royal Statistical Society. Series A, Statistics in society_ , 156(3):503–503, 1993. ISSN 0964-1998. 

- Peter Hall. _The bootstrap and Edgeworth expansion_ . Springer series in statistics. Springer-Verlag, New York, 1992. ISBN 9780387945088. 

- Nikolaus Hansen. The cma evolution strategy: A tutorial, 2023. URL https://arxiv.org/abs/1604. 00772. 

- Nikolaus Hansen and Andreas Ostermeier. Completely derandomized self-adaptation in evolution strategies. _Evolutionary Computation_ , 9(2):159–195, 2001a. 

- Nikolaus Hansen and Andreas Ostermeier. Completely Derandomized Self-Adaptation in Evolution Strategies. _Evolutionary Computation_ , 9(2):159–195, June 2001b. ISSN 1063-6560. doi: 10.1162/ 106365601750190398. URL https://ieeexplore.ieee.org/document/6790628. 

- Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024. URL https://arxiv.org/abs/2402.14008. 

- Joel Heck and Fathi M. Salem. Simplified minimal gated unit variations for recurrent neural networks, 2017. URL https://arxiv.org/abs/1701.03452. 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021. URL https: //arxiv.org/abs/2103.03874. 

- Sepp Hochreiter and Jürgen Schmidhuber. Lstm can solve hard long time lag problems. In M.C. Mozer, M. Jordan, and T. Petsche (eds.), _Advances in Neural Information Processing Systems_ , volume 9. MIT Press, 1996. URL https://proceedings.neurips.cc/paper_files/paper/1996/ file/a4d2f0d23dcc84ce983ff9157f8b7f88-Paper.pdf. 

- Mark Horowitz. 1.1 computing’s energy problem (and what we can do about it). In _2014 IEEE International Solid-State Circuits Conference Digest of Technical Papers (ISSCC)_ , pp. 10–14, 2014. doi: 10.1109/ISSCC. 2014.6757323. 

- Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In _ICLR_ . OpenReview.net, 2022. 

- Ruihong Huang and Tomas Polak. LOBSTER: Limit order book reconstruction system. _Available at SSRN 1977207_ , 2011. 

- Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference, 2017. URL https://arxiv.org/abs/1712.05877. 

- Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), _Advances in Neural Information Processing Systems_ , volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper_files/paper/2018/file/ 5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf. 

- Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M. Czarnecki, Jeff Donahue, Ali Razavi, Oriol Vinyals, Tim Green, Iain Dunning, Karen Simonyan, Chrisantha Fernando, and Koray Kavukcuoglu. Population Based Training of Neural Networks, November 2017. URL http://arxiv.org/abs/ 1711.09846. arXiv:1711.09846 [cs]. 

18 

- Feihu Jin, Yifan Liu, and Ying Tan. Derivative-free optimization for low-rank adaptation in large language models. _IEEE/ACM Trans. Audio, Speech and Lang. Proc._ , 32:4607–4616, October 2024. ISSN 2329-9290. doi: 10.1109/TASLP.2024.3477330. URL https://doi.org/10.1109/TASLP.2024.3477330. 

- Jean Kaddour. The minipile challenge for data-efficient language models. _arXiv preprint arXiv:2304.08442_ , 2023. 

- Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 

- Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In Yoshua Bengio and Yann LeCun (eds.), _2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings_ , 2014. URL http://arxiv.org/abs/1312.6114. 

- Daria Korotyshova, Boris Shaposhnikov, Alexey Malakhov, Alexey Khokhulin, Nikita Surnachev, Kirill Ovcharenko, George Bredis, Alexey Gorbatovski, Viacheslav Sinii, and Daniil Gavrilov. Essa: Evolutionary strategies for scalable alignment, 2025. URL https://arxiv.org/abs/2507.04453. 

- John R. Koza. Genetic programming as a means for programming computers by natural selection. _Statistics and Computing_ , 4(2):87–112, June 1994. ISSN 1573-1375. doi: 10.1007/BF00175355. URL https: //doi.org/10.1007/BF00175355. 

- Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In F. Pereira, C.J. Burges, L. Bottou, and K.Q. Weinberger (eds.), _Advances in Neural Information Processing Systems_ , volume 25. Curran Associates, Inc., 2012. URL https://proceedings.neurips.cc/paper_files/paper/2012/file/ c399862d3b9d6b76c8436e924a68c45b-Paper.pdf. 

- Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In _Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles_ , 2023. 

- Robert Tjarko Lange, Tom Schaul, Yutian Chen, Tom Zahavy, Valentin Dallibard, Chris Lu, Satinder Singh, and Sebastian Flennerhag. Discovering Evolution Strategies via Meta-Black-Box Optimization, March 2023. URL http://arxiv.org/abs/2211.11260. arXiv:2211.11260 [cs]. 

- Pierre-Simon Laplace. Mémoire sur les intégrales définies et leur application aux probabilités, et spécialement à la recherche du milieu qu’il faut choisir entre les résultats des observations. _Mémoires de la Classe des Sciences Mathématiques et Physiques de l’Institut Impérial de France,_ 1[re] série, 11(1[re] partie):297–347, 1811. 

- Jaehoon Lee, Lechao Xiao, Samuel S. Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models under gradient descent. In _Proceedings of the 33rd International Conference on Neural Information Processing Systems_ , Red Hook, NY, USA, 2019. Curran Associates Inc. 

- Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models, 2022. URL https://arxiv.org/abs/2206.14858. 

- Junjie Li, Yang Liu, Weiqing Liu, Shikai Fang, Lewen Wang, Chang Xu, and Jiang Bian. Mars: a financial market simulation engine powered by generative foundation model. In _The Thirteenth International Conference on Learning Representations_ , 2025. URL https://openreview.net/forum?id= Yqk7EyT52H. 

19 

- Oscar Li, James Harrison, Jascha Sohl-Dickstein, Virginia Smith, and Luke Metz. Variance-reduced gradient estimation via noise-reuse in online evolution strategies. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. 

- Elliott H. Lieb and Michael Loss. _Analysis_ . Graduate studies in mathematics ; volume 14. American Mathematical Society, Providence, Rhode Island, 2nd ed. edition, 2010 - 2010. ISBN 1-4704-1143-1. 

- Jarek Liesen, Chris Lu, and Robert Lange. rejax, 2024. URL https://github.com/keraJLi/rejax. 

- Chaoyue Liu, Libin Zhu, and Mikhail Belkin. On the linearity of large non-linear models: when and why the tangent kernel is constant. In _Proceedings of the 34th International Conference on Neural Information Processing Systems_ , NeurIPS 2020, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546. 

- Jun S. Liu. Siegel’s formula via stein’s identities. _Statistics and probability letters_ , 21(3):247–251, 1994. ISSN 0167-7152. 

- Zichen Liu, Anya Sims, Keyu Duan, Changyu Chen, Simon Yu, Xiangxin Zhou, Haotian Xu, Shaopan Xiong, Bo Liu, Chenmien Tan, Chuen Yang Beh, Weixun Wang, Hao Zhu, Weiyan Shi, Diyi Yang, Michael Shieh, Yee Whye Teh, Wee Sun Lee, and Min Lin. Gem: A gym for agentic llms, 2025. URL https://arxiv.org/abs/2510.01051. 

- Ryan Lowe, Aviv Tamar, Jean Harb, OpenAI Pieter Abbeel, and Igor Mordatch. Multi-agent actor-critic for mixed cooperative-competitive environments. _Advances in neural information processing systems_ , 30, 2017. 

- Chris Lu, Jakub Kuba, Alistair Letcher, Luke Metz, Christian Schroeder de Witt, and Jakob Foerster. Discovered policy optimisation. _Advances in Neural Information Processing Systems_ , 35:16455–16468, 2022. 

- H. M. Macdonald. Zeroes of the bessel functions. _Proceedings of the London Mathematical Society_ , 30: 165–179, 1899. doi: 10.1112/plms/s1-30.1.165. 

- Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D. Lee, Danqi Chen, and Sanjeev Arora. Fine-tuning language models with just forward passes. In _Proceedings of the 37th International Conference on Neural Information Processing Systems_ , NIPS ’23, Red Hook, NY, USA, 2023. Curran Associates Inc. 

- Michael Matthews, Michael Beukman, Benjamin Ellis, Mikayel Samvelyan, Matthew Jackson, Samuel Coward, and Jakob Foerster. Craftax: A lightning-fast benchmark for open-ended reinforcement learning. _arXiv preprint arXiv:2402.16801_ , 2024. 

- Michael T. Matthews, Michael Beukman, Chris Lu, and Jakob Nicolaus Foerster. Kinetix: Investigating the training of general agents through open-ended physics-based control tasks. In _ICLR_ , 2025. URL https://openreview.net/forum?id=zCxGCdzreM. 

- William Merrill, Jackson Petty, and Ashish Sabharwal. The illusion of state in state-space models. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), _Proceedings of the 41st International Conference on Machine Learning_ , volume 235 of _Proceedings of Machine Learning Research_ , pp. 35492–35506. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/merrill24a.html. 

- Luke Metz, James Harrison, C. Daniel Freeman, Amil Merchant, Lucas Beyer, James Bradbury, Naman Agrawal, Ben Poole, Igor Mordatch, Adam Roberts, and Jascha Sohl-Dickstein. VeLO: Training Versatile Learned Optimizers by Scaling Up, November 2022. URL http://arxiv.org/abs/2211.09760. arXiv:2211.09760 [cs, math, stat]. 

- Valentin Mohl, Sascha Frey, Reuben Leyland, Kang Li, George Nigmatulin, Mihai Cucuringu, Stefan Zohren, Jakob Foerster, and Anisoara Calinescu. Jaxmarl-hft: Gpu-accelerated large-scale multi-agent reinforcement learning for high-frequency trading. In _Proceedings of the 6th ACM International Conference on AI in Finance_ , pp. 18–26, 2025. URL https://doi.org/10.1145/3768292.3770416. 

20 

- Peer Nagy, Sascha Frey, Silvia Sapora, Kang Li, Anisoara Calinescu, Stefan Zohren, and Jakob Foerster. Generative ai for end-to-end limit order book modelling: A token-level autoregressive generative model of message flow using a deep state space network. In _Proceedings of the Fourth ACM International Conference on AI in Finance_ , ICAIF ’23, pp. 91–99, 2023. 

- Peer Nagy, Sascha Yves Frey, Kang Li, Bidipta Sarkar, Svitlana Vyetrenko, Stefan Zohren, Ani Calinescu, and Jakob Nicolaus Foerster. LOB-bench: Benchmarking generative AI for finance - an application to limit order book data. In _Forty-second International Conference on Machine Learning_ , 2025. URL https://openreview.net/forum?id=CXPpYJpYXQ. 

- Brian Ning, Franco Ho Ting Lin, and Sebastian Jaimungal. Double deep q-learning for optimal execution. _Applied Mathematical Finance_ , 28(4):361–380, 2021. 

- Jack Parker-Holder, Vu Nguyen, and Stephen Roberts. Provably Efficient Online Hyperparameter Optimization with Population-Based Bandits, June 2021. URL http://arxiv.org/abs/2002.02518. arXiv:2002.02518 [cs]. 

- Bo Peng, Ruichong Zhang, Daniel Goldstein, Eric Alcaide, Xingjian Du, Haowen Hou, Jiaju Lin, Jiaxing Liu, Janna Lu, William Merrill, Guangyu Song, Kaifeng Tan, Saiteja Utpala, Nathan Wilce, Johan S. Wind, Tianyi Wu, Daniel Wuttke, and Christian Zhou-Zheng. Rwkv-7 "goose" with expressive dynamic state evolution, 2025. URL https://arxiv.org/abs/2503.14456. 

- K. B. Petersen and M. S. Pedersen. The matrix cookbook, nov 2012. URL http://localhost/pubdb/ p.php?3274. Version 20121115. 

- Eduardo Pignatelli, Jarek Liesen, Robert Tjarko Lange, Chris Lu, Pablo Samuel Castro, and Laura Toni. Navix: Scaling minigrid environments with jax, 2024. URL https://arxiv.org/abs/2407.19396. 

- Xin Qiu, Yulu Gan, Conor F. Hayes, Qiyao Liang, Elliot Meyerson, Babak Hodjat, and Risto Miikkulainen. Evolution strategies at scale: Llm fine-tuning beyond reinforcement learning, 2025. URL https:// arxiv.org/abs/2509.24372. 

- I. Rechenberg. Evolutionsstrategien. In Berthold Schneider and Ulrich Ranft (eds.), _Simulationsmethoden in der Medizin und Biologie_ , pp. 83–114, Berlin, Heidelberg, 1978. Springer Berlin Heidelberg. ISBN 978-3-642-81283-5. 

- V. K. Rohatgi. _An introduction to probability theory and mathematical statistics_ . Wiley series in probability and mathematical statistics. Wiley, New York, 1976. ISBN 0471731358. 

- Frank. Rosenblatt. _Principles of neurodynamics : perceptrons and the theory of brain mechanisms._ Spartan Books, Washington, 1962. 

- Alexander Rutherford, Benjamin Ellis, Matteo Gallici, Jonathan Cook, Andrei Lupu, Garðar Ingvarsson, Timon Willi, Ravi Hammond, Akbir Khan, Christian Schroeder de Witt, Alexandra Souly, Saptarashmi Bandyopadhyay, Mikayel Samvelyan, Minqi Jiang, Robert Tjarko Lange, Shimon Whiteson, Bruno Lacerda, Nick Hawes, Tim Rocktäschel, Chris Lu, and Jakob Nicolaus Foerster. JaxMARL: Multi-agent RL environments and algorithms in JAX. In _The Thirty-eighth Conference on Neural Information Processing Systems Datasets and Benchmarks Track_ , 2024. 

- Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, and Ilya Sutskever. Evolution strategies as a scalable alternative to reinforcement learning, 2017. URL https://arxiv.org/abs/1703.03864. 

- John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw. Parallel random numbers: As easy as 1, 2, 3. In _SC ’11: Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis_ , pp. 1–12, 2011. doi: 10.1145/2063384.2063405. 

- Md Kamruzzaman Sarker, Lu Zhou, Aaron Eberhart, and Pascal Hitzler. Neuro-symbolic artificial intelligence: Current trends, 2021. URL https://arxiv.org/abs/2105.05330. 

21 

Hans-Paul Schwefel. _Evolution and Optimum Seeking_ . John Wiley & Sons, New York, 1995. 

- Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300. 

- Zhihong Shao, Yuxiang Luo, Chengda Lu, Z. Z. Ren, Jiewen Hu, Tian Ye, Zhibin Gou, Shirong Ma, and Xiaokang Zhang. Deepseekmath-v2: Towards self-verifiable mathematical reasoning, 2025. URL https: //arxiv.org/abs/2511.22570. 

- Jimmy TH Smith, Andrew Warrington, and Scott W Linderman. Simplified state space layers for sequence modeling. In _International Conference on Learning Representations_ , 2023. 

- Jasper Snoek, Hugo Larochelle, and Ryan P. Adams. Practical bayesian optimization of machine learning algorithms, 2012. URL https://arxiv.org/abs/1206.2944. 

- Charles Stein. A bound for the error in the normal approximation to the distribution of a sum of dependent random variables. In _Proceedings of the Sixth Berkeley Symposium on Mathematical Statistics and Probability_ , volume 2, pp. 583–602, Berkeley, CA, 1972. University of California Press. 

- Felipe Petroski Such, Vashisht Madhavan, Edoardo Conti, Joel Lehman, Kenneth O. Stanley, and Jeff Clune. Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning, April 2018. URL http://arxiv.org/abs/1712.06567. arXiv:1712.06567 [cs]. 

- Laurits Tani, Diana Rand, Christian Veelken, and Mario Kadastik. Evolutionary algorithms for hyperparameter optimization in machine learning for application in high energy physics. _The European Physical Journal C_ , 81(2):170, February 2021. ISSN 1434-6044, 1434-6052. doi: 10.1140/epjc/s10052-021-08950-y. URL http://arxiv.org/abs/2011.04434. arXiv:2011.04434 [hep-ex]. 

- Nico M Temme. _Bessel Functions_ , chapter 9, pp. 219–255. John Wiley and Sons, Ltd, 1996. ISBN 9781118032572. doi: https://doi.org/10.1002/9781118032572.ch9. URL https://onlinelibrary. wiley.com/doi/abs/10.1002/9781118032572.ch9. 

- Sebastian Towers, Aleksandra Kalisz, Philippe A. Robert, Alicia Higueruelo, Francesca Vianello, MingHan Chloe Tsai, Harrison Steel, and Jakob N. Foerster. ADIOS: Antibody Development via Opponent Shaping, June 2025. URL http://arxiv.org/abs/2409.10588. arXiv:2409.10588 [q-bio]. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), _Advances in Neural Information Processing Systems_ , volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper_ files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf. 

- Roman Vershynin. _High-Dimensional Probability: An Introduction with Applications in Data Science_ . Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, UK, 2018. ISBN 9781108415194. Foundational text covering concentration of norms and high-dimensional Gaussian phenomena. 

- Amala Mary Vincent and P. Jidesh. An improved hyperparameter optimization framework for AutoML systems using evolutionary algorithms. _Scientific Reports_ , 13(1):4737, March 2023. ISSN 2045-2322. doi: 10.1038/s41598-023-32027-3. URL https://doi.org/10.1038/s41598-023-32027-3. 

- Martin J. Wainwright. _Basic tail and concentration bounds_ , pp. 21–57. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2019. 

- Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, and Furu Wei. Bitnet: Scaling 1-bit transformers for large language models, 2023. URL https://arxiv.org/abs/2310.11453. 

22 

- G. N. Watson. _A Treatise on the Theory of Bessel Functions_ . Cambridge University Press, Cambridge, 2 edition, 1944. Reprinted with corrections, various later printings. 

- Sven A. Wegner. Gaussian random vectors in high dimensions. In _Mathematical Introduction to Data Science_ , pp. 139–149. Springer, Berlin, Heidelberg, 2024. doi: 10.1007/978-3-662-69426-8_10. Chapter proving and discussing the Gaussian annulus theorem. 

- G. B. Whitham. _Linear and nonlinear waves_ . Pure and applied mathematics. Wiley-Interscience, New York, 1999. ISBN 9786613306241. 

- Daan Wierstra, Tom Schaul, Tobias Glasmachers, Yi Sun, and Jürgen Schmidhuber. Natural evolution strategies, 2011. URL https://arxiv.org/abs/1106.4487. 

- Samuel Webb Williams. _Auto-tuning performance on multicore computers_ . PhD thesis, USA, 2008. AAI3353349. 

- C.S. Withers. A simple expression for the multivariate hermite polynomials. _Statistics and Probability Letters_ , 47(2):165–169, 2000. ISSN 0167-7152. doi: https://doi.org/10.1016/S0167-7152(99)00153-4. URL https://www.sciencedirect.com/science/article/pii/S0167715299001534. 

- Ke Xue, Chao Qian, Ling Xu, and Xudong Fei. Evolutionary gradient descent for non-convex optimization. In Zhi-Hua Zhou (ed.), _Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21_ , pp. 3221–3227. International Joint Conferences on Artificial Intelligence Organization, 8 2021. doi: 10.24963/ijcai.2021/443. URL https://doi.org/10.24963/ijcai.2021/443. Main Track. 

- An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388. 

- Ziming Yu, Pan Zhou, Sike Wang, Jia Li, Mi Tian, and Hua Huang. Zeroth-order fine-tuning of llms in random subspaces. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , pp. 4475–4485, October 2025. 

- Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? _arXiv preprint arXiv:2504.13837_ , 2025. 

- Yihua Zhang, Pingzhi Li, Junyuan Hong, Jiaxiang Li, Yimeng Zhang, Wenqing Zheng, Pin-Yu Chen, Jason D. Lee, Wotao Yin, Mingyi Hong, Zhangyang Wang, Sijia Liu, and Tianlong Chen. Revisiting zeroth-order optimization for memory-efficient llm fine-tuning: A benchmark, 2024. 

23 

## **Appendix** 

|**A **|**Notation**|**Notation**|||**26**|
|---|---|---|---|---|---|
|**B**|**ES Matrix Gradient Deviations**||||**26**|
|**C **|**High-Dimensional Analysis**||||**27**|
||C.1|High-Dimensional Gaussian ES and Convergence . . . . . . . . . . . . . . . . . . . . . . .|||27|
||C.2|Critical Convergence Rate||. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|32|
||C.3|EGGROLL Linearisation|.|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|34|
|**D **|**Asymptotic Rank Analysis**||||**42**|
||D.1|Mean Field Score Function Approximator . . . . . . . . . . . . . . . . . . . . . . . . . . .|||46|
||D.2|Derivation of Mean-feld Approximators . . . . . . . . . . . . . . . . . . . . . . . . . . . .|||47|
|**E**|**EGGROLL Speed**||||**52**|
|**F**|**Arithmetic Intensity Analysis**||||**53**|
||F.1|Arithmetic Intensity of Standard Batched Inference . . . . . . . . . . . . . . . . . . . . . .|||53|
||F.2|Arithmetic Intensity of Gaussian Matrix ES . . . . . . . . . . . . . . . . . . . . . . . . . .|||53|
||F.3|Arithmetic Intensity of EGGROLL . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|||54|
|**G **|**EGG Architecture**||||**55**|
||G.1|Motivation . . . . . . . .|.|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|55|
||G.2|Notation and Operations|.|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|55|
||G.3|Parameter Initialisation .|.|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|56|
||G.4|Matrix Multiplication . .|.|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|56|
||G.5|Embedding<br>. . . . . . .|.|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|57|
||G.6|Layer Normalisation (LN)||. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|57|
||G.7|MLP . . . . . . . . . . .|.|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|57|
||G.8|GRU . . . . . . . . . . .|.|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|57|
||G.9|Fitness Calculation in Integer Types . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|||58|
|**H **|**EGG Pretraining with Integer**||**EGGROLL**||**58**|
||H.1|Adding EGGROLL Perturbations<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|||58|
||H.2|Fitness Shaping . . . . .|.|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|58|
||H.3|Parameter Update . . . .|.|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|58|
|**I**|**EGG Ablations**||||**59**|
|**J**|**Distributed EGGROLL Framework**||||**60**|
||J.1|Base-3 Fitness Packing and||Bandwidth Effciency . . . . . . . . . . . . . . . . . . . . . . .|60|
||J.2|System Architecture. . .|.|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|60|
|**K **|**Fine-tuning of Integer Quantised Models**||||**60**|
||K.1|Quantisation Procedure .|.|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|60|



24 

||K.2|Integrating integer-quantised EGGROLL with Adam|. . . . . . . . . . . . . . . . . . . . .|60|
|---|---|---|---|---|
|**L**|**Fine-tuning Pretrained Transformer LLMs with Verifable Rewards**|||**61**|
||L.1|Results. . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . . . . . . . . . . . .|61|
||L.2|Training Infrastructure for Large-Scale Transformer LLMs . . . . . . . . . . . . . . . . . .||62|
|**M **|**Fine-tuning Time Series Foundation Model: High-Frequency Trading**|||**64**|
|**N **|**Experimental Details**|||**66**|
||N.1|Multi Agent Reinforcement Learning Experiments|. . . . . . . . . . . . . . . . . . . . . .|66|
||N.2|Reasoning Fine-tuning Experiments: Countdown .|. . . . . . . . . . . . . . . . . . . . . .|68|
||N.3|Reasoning Fine-tuning Experiments: GSM8K . . .|. . . . . . . . . . . . . . . . . . . . . .|69|
||N.4|Reinforcement Learning Experiments<br>. . . . . . .|. . . . . . . . . . . . . . . . . . . . . .|69|



25 

## **A Notation** 

**==> picture [434 x 113] intentionally omitted <==**

so mat(vec( _M_ )) = _M_ . We will use the fact that the Frobenius norm becomes the _ℓ_ 2 norm in vector space: 

**==> picture [332 x 32] intentionally omitted <==**

Our proofs make use of Fourier analysis. For a vector-valued function _f_ ( _v_ ) : R _[d] →_ R, we define the Fourier transform as: 

**==> picture [212 x 75] intentionally omitted <==**

and the inverse Fourier transform as: 

## **B ES Matrix Gradient Deviations** 

Let _µM_ = vec( _M_ ) _∈_ R _[mn]_ be the vector of mean parameters associated with the matrix _M_ . Let _vM ∈_ R _[mn]_ denote the corresponding search vector associated with _µM_ . As each element of _v_ is generated independently from a standard normal _N_ (0 _,_ 1), the search vector _vM_ is generated from the standard multivariate norm: _vM ∼N_ (0 _, Imn_ ). From Eq. (2), the update for _µM_ is: 

**==> picture [326 x 42] intentionally omitted <==**

where _E_ = mat( _vM_ ) and we have used the fact that sampling _vM ∼N_ (0 _, Imn_ ) is equivalent to sampling _E ∼N_ (0 _, Im, In_ ) and applying _vM_ = vec( _E_ ). Now 

**==> picture [272 x 84] intentionally omitted <==**

26 

## **C High-Dimensional Analysis** 

## **C.1 High-Dimensional Gaussian ES and Convergence** 

We use insights from the Gaussian annulus theorem when investigating the convergence properties of highdimensional ES: our proof relies on the fact that all probability mass converges to the interior of the ball _Bϵ_ ( _µ_ ) := _{x[′] |∥x[′] − µ∥ < ϵ}_ where _ϵ_ = _[ρ]_ 2[in the limit] _[ d][→∞]_[, where] _[ ρ]_[ is the radius of the local ball from] Assumption 2, meaning we only need to consider the smooth region around _µ_ in this limit. Our first result proves that the mass outside of the ball for any polynomially bounded function tends to zero at an exponential rate. 

**Lemma 1** (Polynomial Tail Bounds) **.** _Let g_ ( _x_ ) _be polynomial bounded as:_ 

**==> picture [174 x 11] intentionally omitted <==**

_for some finite polynomial of orders p and q and constant C >_ 0 _. Let Ad_ := _{∥σdv∥≥ ϵ} denote the event that a mutation lies outside the a local ball of radius ϵ around µ. Assume σd_ = _o_ ( _d[−]_[1] _[/]_[2] ) _. Then for some constant K >_ 0 _:_ 

**==> picture [276 x 30] intentionally omitted <==**

_and in particular the right-hand side is o_ (1) _as d →∞._ 

_Proof._ We start by bounding the integrand using the polynomial bound. Denote P( _Ad_ ) := E _v∼N_ (0 _,Id_ )[1( _Ad_ )]. Then, by Jensen’s inequality in the first line, polynomial boundedness in the second and _∥a_ + _b∥[p] ≤_ 2 _[p][−]_[1] ( _∥a∥[p]_ + _∥b∥[p]_ ) in the third: 

**==> picture [430 x 60] intentionally omitted <==**

where _C[′]_ = _C_ (1 + 2 _[p][−]_[1] _∥µ∥[p]_ ) and _C[′′]_ = _C_ 2 _[p][−]_[1] are constants independent of _d_ . Applying the Cauchy– Schwarz inequality to the second expectation gives: 

**==> picture [268 x 19] intentionally omitted <==**

Now, the variable _∥v∥_ is _χd_ -distributed. Using the formula for the _i_ -th central moment of _∥v∥_ about the origin (Forbes et al., 2011, Chapter 11.3) yields: 

**==> picture [158 x 27] intentionally omitted <==**

Applying the identity[Γ] Γ([(] _[z] z_[+] + _[a] b_ )[)] _[∼][z][a][−][b]_[ (][Askey & Roy][,][ 2020-2026][, Eq.][5.11.12):] 

**==> picture [296 x 29] intentionally omitted <==**

where _∼_ denotes asymptotic equivalence. For _i_ = 2( _p_ + _q_ ), this yields the bound: 

**==> picture [144 x 13] intentionally omitted <==**

hence: 

**==> picture [369 x 46] intentionally omitted <==**

27 

We use the Gaussian concentration inequality for the Euclidean norm (Vershynin, 2018, Theorem 3.1.1), which states that for _x ∼N_ (0 _, Id_ ) there exists an absolute constant _K >_ 0 such that for all _t ≥_ 0, 

**==> picture [306 x 74] intentionally omitted <==**

In our setting, we need to bound: 

Setting _t_ = _σϵd[−] √d_ , the assumption _√dσd_ = _o_ (1) implies for sufficiently large _d_ that _√dσd ≤ ϵ_ and therefore _t ≥_ 0, so we can apply the concentration bound to obtain: 

**==> picture [402 x 59] intentionally omitted <==**

Now, as _√dσd_ = _o_ (1), it follows _[σ][d] ϵ_ ~~_√_~~ _d_ = _o_ (1), yielding: 

**==> picture [211 x 98] intentionally omitted <==**

_p −p p_ Applying these results to Eq. (15) , along with _σd[p][d]_ 2 = _O_ ( _d_ 2 ) _d_ 2 = _O_ (1), yields our desired result: 

**==> picture [332 x 99] intentionally omitted <==**

where we have absorbed the factor of[1] 2[into the constant] _[ K]_[.] 

Our proof in Lemma 1 reveals the necessity of the condition _σd√d_ = _o_ (1) for convergence as we can only apply the Gaussian concentration inequality in Eq. (16) for _σd√d_ = _o_ (1); this is a direct consequence of the Gaussian annulus theorem, as for slower rates 1 = _o_ ( _σd√d_ ), the Gaussian probability mass will exit any local ball around _µ_ and flood the tail, meaning that the tail probability will grow with increasing _d_ . Having bounded the tail, convergence to linearity follows by proving convergence within the ball, which allows us to exploit the local _C_[1] smoothness of _f_ ( _x_ ): 

**Theorem 1** (Convergence to Linearity) **.** _Let Assumptions 2, 3 and 4 hold and σd_ = _o d[−]_ 2[1] _. Then:_ � � 

**==> picture [194 x 20] intentionally omitted <==**

_almost surely with respect to the distribution over µ._ 

28 

_Proof._ We start with the definition of the ES update: 

**==> picture [176 x 23] intentionally omitted <==**

Now let _ϵ_ = _[ρ]_[Consider the hinge function:] 2[where] _[ ρ]_[ is the radius of the ball from Assumption][ 2][.] 

**==> picture [155 x 43] intentionally omitted <==**

which interpolates between 1 and 0 in the region _ϵ < ∥x∥ <_ 2 _ϵ_ . Our first goal is to use _ϕ_ ( _x_ ) to generate a function _f_[˜] ( _x_ ) that is absolutely continuous and has integrable derivatives outside of _Bρ_ ( _µ_ ) to allow us to apply Stein’s lemma (Stein, 1972). We define _f_[˜] ( _x_ ) as: 

**==> picture [98 x 12] intentionally omitted <==**

Consider the closed ball _Bϵ_ ( _µ_ ) := _{x[′] |∥x[′] − µ∥≤ ϵ}_ . We note that within the ball _f_ ( _µ_ + _σdv_ ) remains unchanged: 

**==> picture [353 x 49] intentionally omitted <==**

The derivative of the function with respect to _v_ is: 

**==> picture [418 x 49] intentionally omitted <==**

where the gradient fails to exist only on the sets _∥σdv∥∈{ϵ,_ 2 _ϵ}_ , which have Lebesgue measure zero. We start by using this function to decompose _J_ ( _µ_ ) into a smoothed part and a remainder: 

**==> picture [384 x 39] intentionally omitted <==**

Hence: 

**==> picture [333 x 14] intentionally omitted <==**

Consider the smoothed part: 

**==> picture [182 x 23] intentionally omitted <==**

Our goal is to apply Stein’s lemma (Stein, 1972) in its multivariate form (Liu, 1994, Lemma 1). The assumptions of (Liu, 1994, Lemma 1) require that the partial derivatives _∂vif_[˜] ( _µ_ + _σdv_ ) are absolutely continuous almost everywhere and: 

**==> picture [148 x 13] intentionally omitted <==**

These two conditions are satisfied by construction. Indeed, under Assumption 2, _f_ ( _·_ ) is _C_[1] continuous on _Bρ_ ( _µ_ ), hence from Eq. (17), _f_[˜] ( _·_ ) coincides with a compactly supported, piecewise _C_[1] function whose gradient (Eq. (18)) exists almost everywhere. Moreover, under Assumption 3. both _f_ ( _µ_ + _σdv_ ) and _∇f_ ( _µ_ + _σdv_ ) are polynomially bounded, and since _∇f_[˜] ( _µ_ + _σdv_ ) is supported on _∥σdv∥≤_ 2 _ϵ_ , it follows that: 

**==> picture [148 x 13] intentionally omitted <==**

29 

Applying (Liu, 1994, Lemma 1): 

**==> picture [369 x 72] intentionally omitted <==**

Let _{µ_ + _σdv ∈ Bϵ_ ( _µ_ ) _}_ = _{∥σdv∥≤ ϵ}_ denote the event that a mutation lies within the ball _Bϵ_ ( _µ_ ). We now split the integral into two regions, the first within the ball and the second outside: 

**==> picture [407 x 59] intentionally omitted <==**

Consider the region inside the ball, _I_ loc. From Eq. (18), _∇f_[˜] ( _µ_ + _σdv_ ) = _∇f_ ( _µ_ + _σdv_ ) within this region. Using the local _α_ -Hölder continuity from Assumption 2: 

**==> picture [246 x 41] intentionally omitted <==**

**==> picture [272 x 47] intentionally omitted <==**

We now bound the tail region outside the ball: 

**==> picture [254 x 30] intentionally omitted <==**

Now, asbounded under Assumption _∥∇f_ ( _µ_ ) _∥_ = _O_ (1) from Assumption 3 when applying Stein’s lemma, it follows that 4 and we have established that�� _∇ ∥∇f_ ˜( _µf_[˜] +( _µ σ_ + _d σv_ ) _d −∇v_ ) _∥_ is polynomial _f_ ( _µ_ )�� is also polynomial bounded, that is there exists some constant _C >_ 0 and finite polynomial order _p_ such that: 

**==> picture [206 x 14] intentionally omitted <==**

Applying Lemma 1, it follows: 

**==> picture [142 x 31] intentionally omitted <==**

for some constant _K >_ 0. Together, this yields: 

**==> picture [286 x 48] intentionally omitted <==**

As exp( _−x_ ) = _o_ ( _x[−][a]_ ) for any _a >_ 0, we take _a_ = _α/_ 2 to obtain a weakened bound matching the first term: 

**==> picture [224 x 31] intentionally omitted <==**

30 

This yields the upper bound: 

**==> picture [300 x 20] intentionally omitted <==**

Returning to Eq. (19), we must bound the remainder term: 

**==> picture [262 x 50] intentionally omitted <==**

˜ Again, from Assumption 3, it follows that ��( _f_ ( _µ_ + _σdv_ ) _− f_ ( _µ_ + _σdv_ ))�� is polynomially bounded, that is there exists some constant _C[′] >_ 0 and finite polynomial order _p[′]_ such that: 

��( _f_ ( _µ_ + _σdv_ ) _− f_ ˜( _µ_ + _σdv_ ))�� _≤ C ′_ (1 + _∥µ_ + _σdv∥p_ ) _._ 

Applying Lemma 1 with _q_ = 1: 

**==> picture [174 x 31] intentionally omitted <==**

Now, as exp( _−x_ ) = _o_ � _x[−]_[1][�] for _x →∞_ , it follows: 

**==> picture [330 x 104] intentionally omitted <==**

where the final line follows from the fact _√dσd_ = _o_ (1). Assembling our bounds using Ineq. 19 yields our desired result: 

**==> picture [346 x 34] intentionally omitted <==**

We now show that the bound is tight. Consider the function _f_ ( _x_ ) = _[L]_ 2 � _di_ =1 _[x][i][|][x][i][|]_[ +] _[ a][⊤][x]_[ where] _[ ∥][a][∥]_[=] _[ O]_[(1)][.] Taking partial derivatives: 

**==> picture [261 x 11] intentionally omitted <==**

hence: 

**==> picture [176 x 37] intentionally omitted <==**

Applying the reverse triangle inequality _||xi| −|yi|| ≤|xi − yi|_ = _⇒_ ( _|xi| −|yi|_ )[2] _≤_ ( _xi − yi_ )[2] : 

**==> picture [220 x 37] intentionally omitted <==**

31 

We have thus shown that _f_ ( _x_ ) is _C_[1] -continuous and its gradient has Lipschitz constant _L_ , i.e. _α_ = 1 with Hölder constant _L_ . It is also bounded by a polynomial of order 2. Without loss of generality, we take a deterministic initialisation _µ_ = 0 to simplify algebra, yielding; 

**==> picture [178 x 11] intentionally omitted <==**

_f_ ( _x_ ) thus satisfies Assumptions 2, 3 and 4. Using _f_ ( _x_ ) as the fitness: 

**==> picture [214 x 46] intentionally omitted <==**

Taking expectations element-wise and using Eq. (23): 

**==> picture [210 x 27] intentionally omitted <==**

Applying Eq. (14): 

**==> picture [148 x 28] intentionally omitted <==**

Hence: 

**==> picture [156 x 25] intentionally omitted <==**

thereby attaining the upper bound rate of _σd√d_ . 

## **C.2 Critical Convergence Rate** 

To show that our rate is critical, we investigate the space of functions that can be represented by cubic polynomials of the form: 

**==> picture [296 x 22] intentionally omitted <==**

where _a ∈_ R _[d]_ , _B ∈_ R _[d][×][d]_ is a symmetric matrix and _C_ [ _x, x, x_ ] =[�] _i,j,k[c][i,j,k][x][i][x][j][x][k]_[denotes a symmetric] 3-linear map represented by the 3-tensor _C ∈_ R _[d][×][d][×][d]_ . 

Since our theory depends on analysing the local stability of a smooth ball for a fitness function, stability over this class is necessary for convergence on more general objectives. We show that once _σd_ decays slower than the critical rate, divergence already occurs within this subclass, establishing the sharpness of the rate. **Theorem 2** (Exact divergence for cubic objectives) **.** _Let f_ ( _x_ ) _denote the cubic polynomial in Eq._ (24) _. Assume ∥a∥_ = _O_ (1) _,∥B∥_ = _O_ (1) _, ∥C∥_ = _O_ (1) _where ∥·∥ denotes operator norm for i-tensor T_ ( _x_ 1 _, . . . xi_ ) _: ∥T ∥_ := sup _∥x_ 1 _∥_ = _···_ = _∥xi∥_ =1 _|T_ ( _x_ 1 _, . . . xi_ ) _|. Let Assumption 4 hold, then:_ 

**==> picture [194 x 23] intentionally omitted <==**

_Moreover:_ 

**==> picture [162 x 41] intentionally omitted <==**

32 

_Proof._ We start by taking derivatives of _f_ ( _x_ ): 

**==> picture [134 x 22] intentionally omitted <==**

Substituting this into the definition of _∇µJ_ ( _θ_ ) and using Eq. (20): 

**==> picture [380 x 189] intentionally omitted <==**

where we have used the fact _C_ ( _v, µ, ·_ ) = _C_ ( _µ, v, ·_ ) by definition of the symmetry of C. As _C_ ( _v, µ, ·_ ) is linear in _v_ , its expectation under zero-mean _N_ (0 _, Id_ ) is zero, hence: 

**==> picture [194 x 23] intentionally omitted <==**

proving our first result. Now, it follows that _∥C_ ( _v, v, ·_ ) _∥≤∥C∥∥v∥_[2] and as _∥C∥_ = _O_ (1): 

**==> picture [210 x 30] intentionally omitted <==**

Now as _v_ is unit Gaussian: E _v∼N_ (0 _,Id_ ) � _∥v∥_[2][�] = _d_ , hence: 

**==> picture [138 x 12] intentionally omitted <==**

We now show that the bound is tight. Consider the function _f_ ( _x_ ) = _u[⊤] x∥x∥_[2] for _u[⊤]_ = ~~_√_~~ 1 _d_[[1] _[, . . .]_[ 1]][.][The] factor of ~~_√_~~ 1 _d_[ensures][that][the][gradient][of][the][function] _[∇][x][f]_[(] _[x]_[)][=] _[O]_[(1)][.][We][can][write] _[∥][x][∥]_[2][as][the][tensor] contraction: 

**==> picture [70 x 12] intentionally omitted <==**

where _Id_ is the identity matrix and: 

**==> picture [62 x 13] intentionally omitted <==**

hence we write _f_ ( _x_ ) as a tensor contraction as: 

**==> picture [78 x 11] intentionally omitted <==**

where _C_ := Sym( _u ⊗ Id_ ). Using this function: 

**==> picture [307 x 76] intentionally omitted <==**

33 

hence _∥_ E _v∼N_ (0 _,Id_ ) [ _C_ ( _v, v, ·_ )] _∥_ = _d_ + 2, achieving the upper bound rate of _O_ ( _d_ ) which implies: 

**==> picture [138 x 12] intentionally omitted <==**

Our final result follows immediately: 

**==> picture [270 x 23] intentionally omitted <==**

## **C.3 EGGROLL Linearisation** 

We now study the effect of EGGROLL in high dimensions. We introduce the notation _v_ = vec( _E_ ) to denote the vectorisation of the low-rank matrix perturbation _E_ = ~~_√_~~ 1 _r[AB][⊤]_[and work in vector space.][The EGGROLL] vector update _v_ can thus be written as sum of independent variables: 

**==> picture [60 x 28] intentionally omitted <==**

with: 

**==> picture [72 x 13] intentionally omitted <==**

where recall _ai_ and _bi_ are the _i_ th column vectors of _A_ and _B_ . We write _µ_ = vec( _M_ ). Using Eq. (13), we can convert between results in vector space and matrix space as: 

**==> picture [214 x 27] intentionally omitted <==**

To extend our analysis, we need to ensure that all polynomial moments of _P_ ( _v_ ) are finite and grow at most polynomially in the dimension _d_ = _mn_ . In particular, such tail bounds are sufficient to dominate polynomial error terms in our analysis. To introduce sub-Gaussian variables, we follow the exposition of Vershynin (2018) and results therein. A random variable _xi ∈_ R is sub-Gaussian if there exists some finite constant _C >_ 0 such that for all _t >_ 0: 

**==> picture [116 x 12] intentionally omitted <==**

meaning their their tails decay like Gaussians. This is equivalent to any of the following three properties holding (Vershynin, 2018, 2.6.1): There exist constants _C_ 1 _, C_ 2 _, C_ 3 _>_ 0 that differ at most by an absolute constant factor such that: 

**==> picture [150 x 43] intentionally omitted <==**

and if E[ _xi_ ] = 0: 

**==> picture [160 x 12] intentionally omitted <==**

A random vector _x ∈_ R _[d]_ is sub-Gaussian if all one-dimensional marginals of _x_ are sub-Gaussian, i.e. _x[⊤] u_ is sub-Gaussian for all _u ∈_ R _[d]_ . The sub-Gaussian norm is defined as: 

**==> picture [320 x 31] intentionally omitted <==**

which returns the smallest universal sub-Gaussian constant for all marginals. 

34 

A key property of sub-Gaussian vectors that we use in our proofs is the sub-Gaussian concentration inequality for the Euclidean norm (Vershynin, 2018, Theorem 3.1.1), which states that for if _x_ is a sub-Gaussian vector with E[ _x_[2] _i_[] = 1][ and] _[ K]_[=] _[ ∥][x][∥][ψ]_ 2[, there exists an absolute constant] _[ C][>]_[ 0][ such that for all] _[ t][ ≥]_[0][,] 

**==> picture [301 x 25] intentionally omitted <==**

We also use a weaker form of control, that replaces the Gaussian-like tail decay with an exponential decay, but all other properties are defined similarly. In this paper, we use the definition that a variable _x_ is known as sub-exponential if there exists a _K >_ 0 such that for all _t ≥_ 0: 

**==> picture [120 x 25] intentionally omitted <==**

Our first result derives a bound on the expected value of the norms _∥a∥[i]_ and _∥b∥[i]_ : **Lemma 2.** _Let Assumption 6 hold. Let P_ ( _a_ ) _denote the distribution over columns of A and P_ ( _b_ ) _denote the distribution over columns of B. Then:_ 

**==> picture [218 x 15] intentionally omitted <==**

_i i Proof._ It suffices to prove E _a∼P_ ( _a_ )[ _∥a∥[i]_ ] = _O_ ( _m_ 2 ) as E _b∼P_ ( _b_ )[ _∥b∥[i]_ ] = _O_ ( _n_ 2 ) follows automatically from the same assumptions. We start by using the ‘layer cake’ representation of the expectation Lieb & Loss (2010 - 2010, Theorem 1.13): 

**==> picture [172 x 24] intentionally omitted <==**

Let _tm_ = _C[√] m_ for any _C >_ 1. We split the integral into two regions: 

**==> picture [318 x 26] intentionally omitted <==**

For the first integral: 

**==> picture [166 x 56] intentionally omitted <==**

For the second integral, we wish to bound P( _∥a∥ > t_ ) for the region _t ≥ tm_ = _C[√] m_ . Setting _t[′]_ = _t−[√] m >_ 0, the assumption _C >_ 1 implies _t[′] ≥_ 0 in this region, hence 

**==> picture [238 x 12] intentionally omitted <==**

We bound this using the sub-Gaussian concentration inequality from Eq. (25). Under Assumption 6, _a_ is a sub-Gaussian vector with _∥x∥ψ_ 2 _≤∞_ , hence there exists an absolute constant _C[′] >_ 0 such that for all _t[′] ≥_ 0, 

**==> picture [172 x 19] intentionally omitted <==**

This implies: 

**==> picture [162 x 13] intentionally omitted <==**

for all _t ≥ tm_ . Substituting yields: 

**==> picture [254 x 25] intentionally omitted <==**

35 

**==> picture [131 x 11] intentionally omitted <==**

**==> picture [284 x 26] intentionally omitted <==**

_x_ Now, _[√] m ≤ C−_ 1[for all] _[ x][ ≥√] m_ ( _C −_ 1), hence: 

**==> picture [324 x 135] intentionally omitted <==**

Combining the two bounds yields: 

**==> picture [106 x 14] intentionally omitted <==**

as required. 

Using this result, we now bound the whole vector _v_ = ~~_√_~~ 1 _r_ � _ri_ =1[vec][(] _[a][i][b] i[⊤]_[)] **Lemma 3.** _Let i ≥_ 1 _. Under Assumption 6:_ 

**==> picture [132 x 19] intentionally omitted <==**

_Proof._ For any vectors _a, b_ : 

**==> picture [264 x 37] intentionally omitted <==**

**==> picture [127 x 13] intentionally omitted <==**

Applying Lemma 2 under Assumption 6 for each summand of _v_ = 

~~_√_~~ 1 _r_ � _rl_ =1[vec][(] _[a][l][b] l[⊤]_[)][:] 

**==> picture [258 x 35] intentionally omitted <==**

Applying the triangle inequality: 

**==> picture [238 x 116] intentionally omitted <==**

36 

Now, as _i ≥_ 1, we can apply Jensen’s inequality: 

**==> picture [190 x 33] intentionally omitted <==**

yielding: 

**==> picture [370 x 29] intentionally omitted <==**

Our proof borrows techniques used to prove linearisation of the ES update in Section C.1 by bounding the tail probability of any polynomial under the low-rank distribution outside of the ball _Bρ_ ( _µ_ ). To apply the concentration inequality that would generalise Lemma 1, we show that _v_ has an exponentially decaying tail: 

**Lemma 4** (Exponential Tail Bound) **.** _Let r < ∞ and Assumption 6 hold. Then all elements of v are sub-exponential and for √dσd_ = _o_ (1) _there exists some constant C >_ 0 _such that:_ 

**==> picture [162 x 25] intentionally omitted <==**

_Proof._ In matrix form: 

**==> picture [78 x 29] intentionally omitted <==**

The elements of _E_ are thus: 

**==> picture [94 x 29] intentionally omitted <==**

As _aij_ and _bik_ are independent sub-Gaussian random variables with zero mean, it follows from Vershynin (2018, Lemma 2.8.6) that their product _aijbik_ is a zero-mean sub-exponential variable with a uniform norm _∥aijbik∥ψ_ 1 _< ∞_ . Finally, a finite sum of sub-exponential variables is sub-exponential (Wainwright, 2019, Eq. (2.18)) with a uniform norm, so all elements of _E_ and hence _v_ = vec( _E_ ) are sub-exponential and zero-mean with a uniform _ψ_ 1-norm _K < ∞_ . 

We now bound P( _∥σdv∥≥ ρ_ ) = P( _∥v∥≥ σρd_[)][.][For the vector] _[ v]_[, it follows for] _[ t][ ≥]_[0][:] 

**==> picture [126 x 23] intentionally omitted <==**

This is easily proven via the contrapositive: if max _j|vj| <_ ~~_√_~~ _td_[then] 

**==> picture [112 x 31] intentionally omitted <==**

implying _∥v∥ < t_ . This means for _t ≥_ 0: 

**==> picture [292 x 60] intentionally omitted <==**

37 

As _vj_ is a sub-exponential variable with finite uniform sub-exponential norm, by definition (Vershynin, 2018, Proposition 2.8.1) there exists a finite _K_ such that for all _j_ : 

**==> picture [156 x 25] intentionally omitted <==**

Applying to Eq. (26) yields: 

**==> picture [140 x 25] intentionally omitted <==**

Now, using _t_ = _σρd_[and] _[ C]_[=] _K_[1][yields:] 

**==> picture [162 x 26] intentionally omitted <==**

We now use these results to assemble into our key polynomial tail bound: 

**Lemma 5** (EGGROLL Polynomial Tail Bounds) **.** _Let Assumption 6 hold. Let g_ ( _x_ ) _be polynomial bounded as:_ 

**==> picture [98 x 11] intentionally omitted <==**

_for some finite polynomial of order p and constant C >_ 0 _. Consider the ball Bρ_ ( _µ_ ) := _{x[′] |∥x[′] − µ∥ < ρ}. Let {µ_ + _σdv ∈ Bρ_ ( _µ_ ) _}_ = _{∥σdv∥ < ρ} denote the event that a mutation lies outside the ball. Assume σd_ = _o_ ( _d[−]_[1] _[/]_[2] ) _. Then for some constant K >_ 0 _independent of d:_ 

**==> picture [258 x 25] intentionally omitted <==**

_and in particular the right-hand side is o_ (1) _as d →∞._ 

_Proof._ Let 

**==> picture [110 x 11] intentionally omitted <==**

and denote P( _Ad_ ) := E _v∼P_ ( _v_ )[1( _Ad_ )]. Our proof proceeds as in Lemma 1 to obtain: 

**==> picture [304 x 13] intentionally omitted <==**

where _C[′]_ = _C_ (1 + 2 _[p][−]_[1] _∥µ∥[p]_ ) and _C[′′]_ = _C_ 2 _[p][−]_[1] are constant in _d_ . Applying the Cauchy–Schwarz inequality to the second expectation gives: 

**==> picture [220 x 19] intentionally omitted <==**

Applying Lemma 3 with fixed _r_ and _d_ = _mn_ : 

**==> picture [126 x 19] intentionally omitted <==**

Now, P( _Ad_ ) = P( _∥σdv∥≥ ρ_ ). From Lemma 4, there exists some _K >_ 0 such that: 

**==> picture [191 x 53] intentionally omitted <==**

where we have absorbed the factor of[1] 2[into] _[ K]_[, hence:] 

**==> picture [226 x 25] intentionally omitted <==**

38 

**==> picture [179 x 13] intentionally omitted <==**

**==> picture [230 x 25] intentionally omitted <==**

Applying our bounds yields our desired result: 

**==> picture [288 x 25] intentionally omitted <==**

where the _o_ (1) bound follows from the fact that the exponential factor dominates _√d_ and _√dσd_ = _o_ (1). **Theorem 3** (EGGROLL Convergence to Linearity) **.** _Let Assumptions 3, 4, 5 and 6 hold and σd_ = _o_ ( _d[−]_[1] _[/]_[2] ) _and Ld_ ( _σdd_ )[2] = _o_ (1) _. Then there exists some K >_ 0 _such that:_ 

**==> picture [321 x 31] intentionally omitted <==**

3 _∥_ vec(ˆ _gLR_ ) _−∇µJ_ ( _θ_ ) _∥_ = _O σd√d ·_ 1 + _Ldσdd_ 2 = _o_ (1) _._ � � �� _almost surely with respect to the distribution over µ._ 

_Proof._ We start with the definition of the vectorised EGGROLL update: 

**==> picture [404 x 162] intentionally omitted <==**

where we have used the fact that the expectation of an odd function under a symmetric, zero mean distribution is always zero, and _P_ ( _v_ ) satisfies this under Assumption 6, hence E _v∼P_ ( _v_ )[ _vv[⊤] ∇_[2] _f_ ( _µ_ ) _v_ ] = 0, and E _v∼P_ ( _v_ )[ _vv[⊤]_ ] = _Id_ from Lemma 6. Consider the ball _Bρ_ ( _µ_ ) := _{x[′] |∥x[′] − µ∥ < ρ}_ . We now split the integral into two regions, the first within the ball and the second outside: 

**==> picture [329 x 76] intentionally omitted <==**

Consider the region inside the ball: 

**==> picture [318 x 48] intentionally omitted <==**

39 

Within this region, _f_ ( _µ_ + _σdv_ ) is _C_[2] continuous under Assumption 5. We can thus write _f_ ( _µ_ + _σdv_ ) using a first-order Taylor expansion about _µ_ with a Hessian (second order derivative) remainder within the ball: 

**==> picture [324 x 84] intentionally omitted <==**

Applying the Lipschitz bound on the Hessian from Assumption 5: 

**==> picture [262 x 137] intentionally omitted <==**

Using this to bound Eq. (27): 

**==> picture [190 x 46] intentionally omitted <==**

Now, (for fixed _r_ ) we apply the identity E _v∼P_ ( _v_ ) � _∥v∥_[4][�] = _O_ �( _mn_ )[2][�] with _mn_ = _d_ from Lemma 3: _∥I_ loc _∥_ = _O_ ( _Ld_ ( _σdd_ )[2] ) _._ 

We now bound the tail region outside the ball: 

**==> picture [200 x 75] intentionally omitted <==**

Now under Assumptions 3, 4 and 5, _f_ ( _µ_ + _σdv_ ) is polynomial bounded, _∥∇f_ ( _µ_ ) _∥_ = _O_ (1) and _∥∇_[2] _f_ ( _µ_ ) _∥_ is polynomial bounded hence there exists some finite constant _C >_ 0 and finite polynomial order _p_ such that: 

**==> picture [152 x 11] intentionally omitted <==**

We thus apply Lemma 5: 

**==> picture [307 x 64] intentionally omitted <==**

40 

~~_√_~~ _d_ Now, as _σd√d_ = _o_ (1), the exponential term dominates the prefactor _[d] dσd_[2][, we conclude:] 

**==> picture [197 x 24] intentionally omitted <==**

Our final result follows from: 

**==> picture [288 x 27] intentionally omitted <==**

We have already shown _∥_ vec(ˆ _g_ LR) _−∇f_ ( _µ_ ) _∥_ = _o_ (1) and under the assumptions for this theorem, Theorem 1 holds and so _∥∇f_ ( _µ_ ) _−∇µJ_ ( _θ_ ) _∥_ = _o_ (1). 

41 

## **D Asymptotic Rank Analysis** 

For convenience, we work with random vectors in our analysis. We analyse the vector _v[r]_ = vec( _E[r]_ ), which is the vectorisation of the low-rank matrix _E[r]_ . We denote _v_ = vec( _E_ ), which is the vectorisation of the full rank matrix _E_ . Note _v ∼N_ (0 _, Id_ ) which we denote as _P_ ( _v_ ). We write _v[r]_ as a standardised sum of _r_ independent, zero-mean random vectors. Let 

**==> picture [253 x 14] intentionally omitted <==**

where recall _ai_ and _bi_ are the _i_ th column vectors of _A_ and _B_ so: 

**==> picture [68 x 28] intentionally omitted <==**

Denoting the covariance matrix of _p_ ( _u_ ) as Σ _u_ , the central limit theorem proves that the distribution of _v[r]_ converges in distribution to a zero-mean Gaussian _N_ (0 _,_ Σ _r_ ). In Lemma 6, we derive the covariance matrix for Σ _u_ , which we prove is the identity. Our analysis uses an Edgeworth expansion (Bhattacharya & Ranga Rao, 1976) to characterise precisely the rate at which _P_ ( _v[r]_ ) converges to the limiting Gaussian distribution. In Lemma 7, we make an Edgeworth expansion of _P_ ( _v[r]_ ) to show that it is dominated by _O_ � _r[−]_[1][�] terms and higher. These are then used to prove Lemma 8, which allows us to bound the integral of the remainder of the Edgeworth expansion, thereby characterising how fast _P_ ( _v[r]_ ) converges to the limiting Gaussian distribution. 

**Lemma 6.** _Let Assumption 1 hold and ui be defined in Eq._ (28) _. Then the variable ui has identity covariance matrix:_ 

**==> picture [120 x 13] intentionally omitted <==**

_has finite_ 4 _th-order absolute moments:_ 

**==> picture [100 x 14] intentionally omitted <==**

_and the vector v[r]_ = _vec_ ( _E[r]_ ) _is zero-mean and has identity covariance matrix:_ 

**==> picture [124 x 14] intentionally omitted <==**

_Proof._ Under the vec operator, the vector _ui_ can be written element wise as: 

**==> picture [182 x 12] intentionally omitted <==**

We note that all elements in the vector _ui_ have zero mean, and so the covariance matrix is the expectation of the outer product: 

**==> picture [106 x 14] intentionally omitted <==**

The diagonal elements of Σ _u_ are: 

**==> picture [301 x 13] intentionally omitted <==**

As all elements of _a_ , _b_ and _ϵ_ are zero-mean, off-diagonal elements are zero: 

**==> picture [304 x 12] intentionally omitted <==**

Using Eqs. (29) and (30), our first result follows: 

**==> picture [38 x 9] intentionally omitted <==**

Now, as _ui_ is a vector of elements which are sums and products of variables which all have finite 4th order moments from Assumption 1, it immediately follows that _u_ has finite 4th order absolute moments. 

42 

For our final result, we can write _v[r]_ as sum of independent variables: 

**==> picture [120 x 28] intentionally omitted <==**

1 where _xi_ := ~~_√_~~ _r[u][i]_[.][As] _[v][r]_[is][a][sum][of][zero-mean][vectors,][it][is][also][zero-mean.][We][use][the][fact][that][the] covariance of _r_ i.i.d. random variables is equal to the sum of the individual covariances, hence 

**==> picture [116 x 71] intentionally omitted <==**

as required. 

Using Lemma 6, we see the asymptotic Gaussian density of _v[r]_ is a standard normal: 

**==> picture [289 x 27] intentionally omitted <==**

which is the density of _P_ ( _v_ ), where recall _v_ = vec( _E_ ), is the vectorisation of the full rank matrix _E_ . 

Although _P_ ( _v[r]_ ) does not have a density in the usual sense for low-rank _r_ , we can still approximate it with a distribution _p_ ˆ( _v[r]_ ) by making a Taylor series expansion of its characteristic function, which always exists regardless of whether _P_ ( _v[r]_ ) has a well-defined density or not. We now derive the 4th order Edgeworth expansion for _P_ ( _v[r]_ ). Our proof reveals that 3rd order cumulants control all terms in the expansion that decay at rate _O r[−]_ 2[1] . As 3rd order cumulants are all zero due to symmetry in Assumption 1, the overall decay rate � � is controlled by _O_ � _r[−]_[1][�] terms associated with 4th order cumulants. It is for this reason that we obtain a faster convergence rate than the standard central limit theorem. **Lemma 7.** _Let Assumption 1 hold and let v[r]_ = _vec_ ( _E[r]_ ) _and ui be defined in Eq._ (28) _. Let g_ ( _v[r]_ ) _denote the limiting Gaussian density in Eq._ (31) _. Then, the 2nd order Edgeworth expansion of v[r] is a distribution P_[ˆ] ( _v[r]_ ) _defined by the approximate density:_ 

_where:_ 

**==> picture [250 x 81] intentionally omitted <==**

_is a 4th order Hermite polynomial associated with g_ ( _v[r]_ ) _(Laplace, 1811; Hall, 1992; Withers, 2000)._ 

_Proof._ We denote the characteristic function of _P_ ( _ui_ ) as: 

**==> picture [140 x 23] intentionally omitted <==**

and the characteristic function of _P_ ( _v[r]_ ) as: 

**==> picture [142 x 23] intentionally omitted <==**

Recall _v[r]_ = ~~_√_~~ 1 _r_ � _ri_ =1 _[u][i]_[is][the][sum][of] _[r]_[i.i.d.][copies][of] ~~_√_~~ 1 _r[u][i]_[.][Using][the][scaling][property][of][the][Fourier] transform, the characteristic function of ~~_√_~~ 1 _r[u][i]_[ is] _[ φ][U]_ � ~~_√_~~ _ωr_ �. The distribution of a sum of _r_ independent random 

43 

variables is given by the _r_ -fold convolution of the individual distributions. As convolution in the spatial domain corresponds to multiplication in the frequency domain, the characteristic function of _v[r]_ is (Bhattacharya & Ranga Rao, 1976): 

**==> picture [106 x 26] intentionally omitted <==**

Taking logarithms yields the log-characteristic function: 

**==> picture [136 x 53] intentionally omitted <==**

where _KU_ ( _ω_ ) := log _φU_ ( _ω_ ). The cumulants are defined by 

**==> picture [148 x 26] intentionally omitted <==**

The Edgeworth expansion proceeds by a Taylor expansion of _rKU_ � ~~_√_~~ _ωr_ � about _ω_ = 0. A 4th order expansion yields: 

**==> picture [288 x 60] intentionally omitted <==**

where _KU_ (0) = 0. Under Assumption 8, _ui_ is symmetric, hence all odd-order cumulants vanish: _κ_[1] = _κ_[3] = 0. The second-order cumulant satisfies 

**==> picture [108 x 23] intentionally omitted <==**

and from Lemma 6 we have Σ _u_ = _I_ . Substituting yields: 

**==> picture [216 x 29] intentionally omitted <==**

Exponentiating and expanding the exponential to first-order in 1 _/r_ gives: 

**==> picture [246 x 65] intentionally omitted <==**

Taking the inverse Fourier transform (with the convention _F[−]_[1] ( _f_ )( _v_ ) = (2 _π_ ) _[−][d]_[ �] _e[iω][⊤][v] f_ ( _ω_ ) _dω_ ) yields: 

**==> picture [226 x 30] intentionally omitted <==**

and using the identity _Hi,j,k,l_ ( _v[r]_ ) = _g_ ( _v[r]_ ) _[−]_[1] _∂vi[r][∂v] j∂[r][∂v]_[4] _k[r][∂v] l[r][g]_[(] _[v][r]_[)][, we recover the stated Edgeworth density.] 

44 

We now apply key results from Bhattacharya & Ranga Rao (1976) to bound the difference in expectation between the low-rank distribution and the Edgeworth approximation as well as the difference in expectation between the true ES Gaussian distribution and the Edgeworth approximation. 

**Lemma 8.** ˆ _Let f_ ( _v_ ) := _f_ ( _M_ = _µ_ + _σmat_ ( _v_ )) _, let P_ ( _v_ ) = _N_ (0 _, Id_ ) _, P_ ( _v[r]_ ) _be the distribution of v[r] and P_ ( _v[r]_ ) _be the 2nd order Edgeworth expansion of P_ ( _v[r]_ ) _. Let Assumptions 1 and 7 hold and let v[r]_ = _vec_ ( _E[r]_ ) _and ui be defined in Eq._ (28) _. Then:_ 

**==> picture [256 x 41] intentionally omitted <==**

_Proof._ From Lemma 7, we have shown that the Edgeworth expansion for _P_ ( _v[r]_ ) is controlled by 4th order cumulants and higher, that is; 

**==> picture [321 x 28] intentionally omitted <==**

We show that the three assumptions needed to apply Bhattacharya & Ranga Rao (1976, Theorem 20.1) to obtain our result using Eq. (32) hold. Firstly, the boundedness assumption of the integrand holds: 

**==> picture [146 x 25] intentionally omitted <==**

Secondly, the sampling regularity assumption that _ui_ (as defined in Eq. (28)) is zero-mean i.i.d. (satisfied under Assumption 1) with finite 4th order moments (satisfied from Lemma 6) holds. Let _φU_ ( _ω_ ) denote the characteristic function of _p_ ( _u_ ), then the final assumption we need to verify is the Cramer condition: lim sup _∥ω∥→∞ φU_ ( _ω_ ) _<_ 1, which is satisfied from the Riemann-Lebesgue lemma Folland (1999, Theorem 8.22) because _p_ 0( _·_ ) is absolutely continuous under Assumption 1 and hence _|φU_ ( _ω_ ) _| →_ 0 as _∥ω∥→_ 0. Our first result thus follows from applying Bhattacharya & Ranga Rao (1976, Theorem 20.1): 

**==> picture [256 x 19] intentionally omitted <==**

We now derive our second result. 

**==> picture [306 x 68] intentionally omitted <==**

hence 

**==> picture [386 x 68] intentionally omitted <==**

Now by definition, _Hi,j,k,l_ ( _v_ ) is a 4th order Hermite polynomial and under Assumption 7, _|f_ ( _v_ ) _|_ is bounded, hence _∥v∥· |f_ ( _v_ ) _|_ 4![1] _r_ � _i,j,k,l[|][κ] i,j,k,l_[4] _[H][i,j,k,l]_[(] _[v]_[)] _[|]_[ has polynomial growth of order 5 and is bounded by:] 

**==> picture [220 x 27] intentionally omitted <==**

45 

for some finite _C >_ 0. As the expectation of a finite order polynomial under _N_ (0 _, Id_ ) is bounded, it thus follows: 

**==> picture [336 x 23] intentionally omitted <==**

as required. 

Using Lemma 8, we have all ingredients needed derive our main about the convergence result, which follows after some simple algebra on the norm: 

**Theorem 4.** _Let Assumptions 1 and 7 hold, then:_ 

**==> picture [128 x 13] intentionally omitted <==**

_Proof._ We start by converting the Frobenius norm to vector form using Eq. (13): 

**==> picture [414 x 81] intentionally omitted <==**

where _f_ ( _v_ ) := _f_ ( _M_ = _µ_ + _σ_ mat( _v_ )) and _v_ = vec( _E_ ) is the vectorisation of variable _E_ , which is distributed as _v ∼ P_ ( _v_ ) := _N_ (0 _, Id_ ). Let _P_[ˆ] ( _v_ ) be the distribution for the 2nd order Edgeworth expansion, which we derived in Lemma 7. Since _P_[ˆ] ( _v[r]_ ) and _P_[ˆ] ( _v_ ) are identified as the same Edgeworth-expanded distribution on R _[d]_ , we may equivalently write: 

**==> picture [172 x 14] intentionally omitted <==**

hence: 

**==> picture [412 x 61] intentionally omitted <==**

Applying Lemma 8 to each bound yields our desired result: 

**==> picture [128 x 14] intentionally omitted <==**

## **D.1 Mean Field Score Function Approximator** 

We will use _n_ th order Bessel functions of the second kind _Kn_ ( _z_ ) (Basset, 1888; Macdonald, 1899; Watson, 1944), which are conveniently represented by the integral equations: 

**==> picture [176 x 25] intentionally omitted <==**

Bessel functions are the solutions to systems of differential equations that occur naturally in phenomena where there is strong radial symmetry, typically involving the propagation of spherical waves from points like the ripples formed from water droplets (Whitham, 1999). For our setting, Bessel functions describe the 

46 

probability density of the product of rotationally invariant random variables, whose solution is analogous to the interference pattern of two spherical wave propagators. 

Using the representation, we find the derivative of the zeroth order function takes the recursive form: 

**==> picture [333 x 24] intentionally omitted <==**

More generally, the derivative of the _n_ th order Bessel function is Watson (1944, Section 3.71, Eq. 4): 

**==> picture [284 x 22] intentionally omitted <==**

## **D.2 Derivation of Mean-field Approximators** 

To derive a mean-field approximation, we assume that the elements of _A_ and _B_ are drawn independently from the set of generalised Gaussian distributions (GGDs): 

**Assumption 8.** _Assume each element ai,j ∼GG_ ( _s, p_ ) _and bi,j ∼GG_ ( _s, p_ ) _of A and B is independently distributed according to the zero-mean generalised Gaussian distribution GG_ ( _s, p_ ) _with density:_ 

**==> picture [160 x 30] intentionally omitted <==**

_where_ 0 _< s < ∞ is the scale parameter, p >_ 0 _the shape parameter and_ Γ( _·_ ) _is the gamma function._ 

We observe common distributions emerge from the set of GGDs including the Laplace for _p_ = 1, the Gaussian for _p_ = 2 and the uniform over [ _−s,_ + _s_ ] in the limit _p →∞_ . 

If we make the assumption that all elements of _E_ are independent (this is true as _r_ grows) then we can write _p_ ( _E_ ) _≈ p_ ˆ( _E_ ) :=[�] _[m] i_ =1 � _nj_ =1 _[p]_[(] _[E][i,j]_[)][ as the product of the marginal distributions.][Under this approximation,] the score function can be defined element-wise as: 

**==> picture [196 x 14] intentionally omitted <==**

Using this approximation we apply the score function _S_[ˆ] ( _·_ ) element-wise to the matrix _E_ : 

**==> picture [244 x 21] intentionally omitted <==**

For _r_ = 1, _S_[ˆ] ( _·_ ) has a convenient analytic form for all members of the set of GGDs: 

**Theorem 5.** _Let Assumption 8 hold and r_ = 1 _. Then the distribution over marginals p_ ( _Ei,j_ ) _is:_ 

**==> picture [299 x 38] intentionally omitted <==**

_where K_ 0 ( _·_ ) _is the zeroth-order modified Bessel function of the second kind and the marginal score function is defined element-wise as:_ 

**==> picture [214 x 52] intentionally omitted <==**

_Proof._ For _r_ = 1, we denote the elements of vector _A_ as _ai_ and elements of vector _B_ as _bj_ , then the elements of matrix _E_ = _AB[⊤]_ are: _Ei,j_ = _aibj_ . We now derive the distribution of the unnormalised variables: _Ei,j_ 

47 

using the formula for the distribution of the product of two independent random variables (Rohatgi, 1976; Grimmett & Stirzaker, 1993): 

**==> picture [298 x 112] intentionally omitted <==**

where we have used symmetry of the integrand about 0 to derive the final line. Now, making the substitution _x_ = � _asi_ � _p_ , we have: 

hence: 

**==> picture [238 x 84] intentionally omitted <==**

Now, we use the identity (Temme, 1996, Theorem 9.42): 

**==> picture [162 x 26] intentionally omitted <==**

_p_ 2 with _z_ =[2] _[|][E] s[i][p][,j][|]_ to yield: 

**==> picture [162 x 37] intentionally omitted <==**

as required for Eq. (35). Now we derive the marginal score function by applying the chain rule: 

**==> picture [284 x 208] intentionally omitted <==**

where we have used the identity _∂zK_ 0( _x_ ) = _−K_ 1( _x_ ) from Eq. (33). 

48 

For _r >_ 1 we can derive _S_[ˆ] ( _·_ ) for the Gaussian sampling case: 

**Theorem 6.** _Let Assumption 8 hold and p_ = 2 _. Then the distribution over marginals p_ ( _Ei,j_ ) _is:_ 

**==> picture [210 x 29] intentionally omitted <==**

_and the score function is (for Ei,j_ = 0 _):_ 

**==> picture [222 x 40] intentionally omitted <==**

_Proof._ Each element _Ei,j_ is the sum of _r_ independent variables _ui,j,l_ := _ai,lbj,l_ distributed according to Eq. (35) with _p_ = 2: 

**==> picture [160 x 29] intentionally omitted <==**

Let _Zi,j_ = _[√] rEi,j_ , hence: 

**==> picture [68 x 28] intentionally omitted <==**

We first find the density _p_ ( _Zi,j_ ). From Eq. (35), the distribution of each _ui,j,l_ is: 

**==> picture [126 x 25] intentionally omitted <==**

We use the fact that the PDF of a sum of _r_ independent random variables (i.e. _Zi,j_ ) is given by the _r_ -fold convolution of the individual PDFs. As convolution in the spatial domain is equal to multiplication in the frequency domain, the PDF _p_ ( _Zi,j_ ) follows by taking Fourier transform of _p_ ( _ui,j,l_ ), taking the power _r_ and then taking the inverse Fourier transform: 

**==> picture [210 x 25] intentionally omitted <==**

where recall from Sectionand _F[−]_[1] [ _f_[˜] ]( _x_ ) := 21 _π_ � _f_ ˜( A _ω_ ) exp( with _diωx_ =) _dω,_ 1, _F_ the inverse Fourier transform.[ _f_ ]( _ω_ ) := � _f_ ( _x_ ) exp( _−iωx_ )Taking the Fourier transform of the _dx_ denotes the Fourier transform Bessel function: 

**==> picture [380 x 108] intentionally omitted <==**

2 _|x|_ where we have used the fact thatsecond line is zero. Using a standard result, we can evaluate the integral in Eq. ( _K_ 0 � _s_[2] � is an even function of _x_ and so its integral with36) Gradshte sin(˘ın et al. _ωx_ ) ( in the2015, 6.671 Integral 14): 

**==> picture [158 x 32] intentionally omitted <==**

49 

hence: 

**==> picture [343 x 265] intentionally omitted <==**

where we have used the fact that the integrand is an even function and so its integral with sin( _ωZi,j_ ) is zero to derive the penultimate line. To evaluate the integral in Eq. (37) we apply Gradshte˘ın et al. (2015, 3.771 Integral 2): 

**==> picture [266 x 62] intentionally omitted <==**

Using the transformation of variables _Ei,j_ = ~~_√_~~ 1 _r[Z][i,j]_[yields our desired results:] 

**==> picture [210 x 45] intentionally omitted <==**

Now, we derive the score function: 

**==> picture [318 x 110] intentionally omitted <==**

50 

Now, from Eq. (34) for _Ei,j_ = 0: 

**==> picture [363 x 192] intentionally omitted <==**

as required. 

51 

## **E EGGROLL Speed** 

All timings were done on a single GPU on a GH200 (equivalent to a single H100) for a linear model with dimension 8192 in bfloat16, allowing a maximum batch size of 1024. For the graph in Fig. 2a, we pre-generate the noises instead of integrating the noise generation into the forward pass. 

**==> picture [174 x 208] intentionally omitted <==**

**----- Start of picture text -----**<br>
Normalised Training Speeds<br>100<br>91 (69)<br>80<br>60<br>40<br>34<br>20<br>0.41 (0.054)<br>0<br>EGGROLL PPO OpenES<br>Normalised Speed<br>**----- End of picture text -----**<br>


Figure 7: Relative speed of EGGROLL, when including jax noise regeneration. 

In Fig. 7, we consider the impact of regenerating noises on-the-fly using jax PRNG. The darker area and value in parenthesis for EGGROLL and OpenES indicate the speed when regenerating noises on-the-fly, while the full bar indicates the speed when the noises are already generated. 

We regenerate noises on the fly in our primary jax codebase, but pre-generating the EGGROLL perturbations beforehand is also a practical possibility since low-rank perturbations only require a small amount of memory, proportional to the square root of the size of the original parameter matrices. 

52 

## **F Arithmetic Intensity Analysis** 

In this section, we derive the arithmetic intensity of standard batched inference, Gaussian matrix ES, and EGGROLL. We calculate arithmetic intensity as the number of operations divided by the total number of bytes read from or written to. For context, for the (b)float16 datatype on an H100 GPU, there are approximately 1000 teraFLOPS of compute (without sparsity) and 3.35 TB/s of GPU memory bandwidth, meaning that the roofline threshold is approximately 300 ops/byte, defined as the minimum for computation needed for it to be the bottleneck instead of memory movement. 

In the following subsections, we are considering a single linear layer with mean parameter _M ∈_ R _[d][out][×][d][in]_ and a batch of inputs _u ∈_ R _[B][×][d][in]_ . All operations occur with a precision of _s_ bytes per element. 

## **F.1 Arithmetic Intensity of Standard Batched Inference** 

In standard batched inference, we wish to simply calculate _uM[T]_ . The total bytes read as input are _B × din × s_ (for _u_ ) and _dout × din × s_ (for _M_ ), and the total bytes written as output are _B × dout × s_ . The total number of operations are _B × din × dout ×_ 2 since matrix multiplication requires both multiplications and additions for each element of _u_ across all of _dout_ . Therefore, the arithmetic intensity is: 

**==> picture [81 x 9] intentionally omitted <==**

**==> picture [194 x 11] intentionally omitted <==**

When _s_ = 2 (for (b)float16) and _dout_ = _din_ = _m_ , the arithmetic intensity simplifies to 

**==> picture [39 x 22] intentionally omitted <==**

The batch size needed to achieve a desired arithmetic intensity of _A_ is derived as follows: 

**==> picture [112 x 51] intentionally omitted <==**

Therefore, achieving an arithmetic intensity of 300 ops/byte with _m_ = 8192 requires a minimum batch size of 324. 

## **F.2 Arithmetic Intensity of Gaussian Matrix ES** 

In Gaussian matrix ES, we assume access to pre-generated perturbations of shape R _[B][×][d][out][×][d][in]_ . The total bytes read as input are _B × din × s_ (for _u_ ) and _B × dout × din × s_ (for _M_ ), and the total bytes written as output are _B × dout × s_ . Otherwise, the total number of operations is identical to standard batched inference, giving us an arithmetic intensity of 

**==> picture [379 x 23] intentionally omitted <==**

When _s_ = 2 (for (b)float16) and _dout_ = _din_ = _m_ , the arithmetic intensity simplifies to 

**==> picture [31 x 20] intentionally omitted <==**

This means that arithmetic intensity is always strictly less than 1, regardless of batch size or dimensionality. The common way to increase arithmetic intensity is to bring it closer to standard batched inference, reusing the same perturbation across multiple inputs. For instance, when _m_ = 8192, achieving an arithmetic intensity of 300 ops/byte requires that each perturbation is reused at least 324 times, and smaller values of _m_ need to be reused even more often. 

53 

## **F.3 Arithmetic Intensity of EGGROLL** 

For EGGROLL, we assume access to the pre-generated decomposed perturbations _A ∈_ R _[B][×][d][out][×][r]_ and _B ∈_ R _[B][×][d][in][×][r]_ . Therefore, the bytes read as pure input are _B×din×s_ + _B×_ ( _din_ + _dout_ ) _×r×s_ + _dout×din×s_ and the bytes written as pure output are _B × dout × s_ . However, the efficient low-rank perturbation calculation requires writing and reading an intermediate matrix of shape _B × r_ , so the total bytes read are 

**==> picture [284 x 11] intentionally omitted <==**

The total number of operations includes the amount for standard batch inference, _B × din × dout ×_ 2, along with the rank- _r_ perturbations, _B ×_ ( _din_ + _dout_ ) _× r ×_ 2, and the final sum between the main calculation and perturbation _B × dout_ . Therefore, the arithmetic intensity is 

**==> picture [285 x 24] intentionally omitted <==**

When _s_ = 2 (for (b)float16) and _dout_ = _din_ = _m_ , the arithmetic intensity simplifies to 

**==> picture [110 x 58] intentionally omitted <==**

The batch size needed to achieve a desired arithmetic intensity of _A_ is derived as follows: 

**==> picture [254 x 74] intentionally omitted <==**

Note that the only difference with the critical batch size of standard batched inference is the additional 2 _r_ +[1] 2 _[−][rA]_[(2 +] _m_[2][)][ in the denominator.][Therefore, achieving an arithmetic intensity of 300 ops/byte with] _m_ = 8192 and _r_ = 1 requires a minimum batch size of 352, compared to 324 for standard batched inference. This means that EGGROLL can saturate compute with unique perturbations per input, unlike Gaussian matrix ES. Note that there is an overhead of _Bm_ (4 _r_ + 1) flops relative to standard batched inference, resulting in an additional compute rate of _[Bm]_ 2 _Bm_[(][4] _[r]_[+][2][1][)] =[4] 2 _[r] m_[+1][, which is effectively negligible for large enough matrices.] 

54 

## **G EGG Architecture** 

In the following section, we detail the design of our EGG model, which follows the high-level structure of modern pre-layernorm decoder-only language models, but replaces self-attention with a modified minGRU and standard layernorms with a custom variant to enable pure integer training. See Algorithm 2 for an overview of the forward pass of the EGG architecture. 

## **Algorithm 2** EGG forward pass 

**Require:** Input token _t ∈_ U8, input state _s ∈_ I _[l]_ 8 _[×][D]_ , network parameters _θ_ **Ensure:** Output vector _y ∈_ I _[D]_ 8[and output state] _[ s][′][∈]_[I] _[l]_ 8 _[×][D] s[′] ←_ I _[l]_ 8 _[×][D]_ initialised to 0 _y ←_ EMBED( _θ_ emb _, t_ ) **for** _i ∈{_ 0 _, . . . , l −_ 1 _}_ **do** _y[′] , s[′] i[←]_[GRU][(] _[θ]_[gru] _[,i][,]_[ LN][(] _[θ]_[ln1] _[,i][, y]_[)] _[, s][i]_[)] _y ←_ I8(I32( _y[′]_ ) + I32( _y_ )) _y[′] ←_ MLP( _θ_ mlp _,i,_ LN( _θ_ ln2 _,i, y_ )) _y ←_ I8(I32( _y[′]_ ) + I32( _y_ )) **end for** _y ←_ LN( _θ_ lnout _,i, y_ )@ _θ_ head _[T]_ 

## **G.1 Motivation** 

Since EGGROLL does not rely on gradients, we can explicitly design a language model architecture to be efficient and hardware-friendly at inference time. In particular, we design EGG under the following constraints to emphasise the flexibility of EGGROLL: 

**Pure Integer Training:** On H100 systems, int8 is the fastest datatype and int8 matrix multiplication with int32 accumulation is the fastest tensor core operation. Furthermore, integer datatypes are much simpler to implement in hardware, providing massive energy savings for high-throughput systems (Horowitz, 2014). Therefore, we keep all weights in int8 and all activations in integer formats, _never_ casting to floating point at any point during training. This stands in contrast to the standard approach for language model quantisation through "quantisation aware training" with backpropagation, where floating point activations are still necessary (Wang et al., 2023). 

**Nonlinear RNN:** Modern language models use sequence-parallel architectures like Transformers and SSMs, since they enable stable gradients without backpropagation through time. However, most of these architectures cannot handle simple state tracking (Merrill et al., 2024), whereas classic recurrent networks like LSTMs and GRUs can do so with a single layer. Since EGGROLL does not require backpropagation through time, we can train on unbounded sequence lengths (Li et al., 2023) with nonlinear RNNs of broader complexity classes. Specifically, we develop a variant of the minGRU model (Heck & Salem, 2017) that performs all operations in integer formats. 

**Removal of all Activation Functions:** Inspired by Foerster (2017), we remove all activation functions, like the rectified linear unit and hyperbolic tangent, due to the nonlinearity present in the int8 datatype. Specifically, the saturated addition of int8 values provides sufficient nonlinearity due to the implicit clipping of values to the int8 dynamic range, which evolution strategies can exploit. 

## **G.2 Notation and Operations** 

We use the constant _l ∈_ Z[+] to denote the number of layers of the model and _D_ = 4 _[d]_ as the hidden dimension of the model, where _d ∈_ Z[+] . 

We use I _n_ to denote an _n_ -bit signed integer and U _n_ to denote an _n_ -bit unsigned integer. We denote casting vector _⃗u_ to format I _n_ as I _n_ ( _⃗u_ ), which implicitly includes clipping to the bounds of the datatype. To ensure symmetry between positive and negative values of each datatype, we consider the value _−_ 2 _[n][−]_[1] to be invalid for datatype I _n_ ; for instance, for 8-bit signed integers we only allows value from -127 to 127. 

We use the following operations: 

55 

- _⃗u_ @ _M_ indicating scaled vector-matrix multiplication of I _[n]_ 8 _[×]_[I] _[n,m]_ 8 _→_ I _[m]_ 8[, corresponding to int8 tensor] core multiplication with int32 accumulation and scaling. The details of this operation are described in Section G.4. 

- _a · b_ indicates dot product with int32 accumulation, I _[n]_ 8 _[×]_[ I] _[n]_ 8 _[→]_[I][32][, and] _[ a][ ⊙][b]_[ indicates the Hadamard] (elementwise) product. 

- Standard integer operations: + for addition, _−_ for subtraction, and _⊙_ for element-wise multiplication. 

- _|u|_ indicates taking the element-wise absolute value of _u_ , I _[n] →_ I _[n]_ . 

- sign( _u_ ) indicates taking the element-wise sign of _u_ , giving 1 for positive values, -1 for negative values, and 0 for zero. 

- sum( _u_ ) indicates taking the sum of all elements in _u_ (casting to I32 to prevent overflow): I _[n] →_ I32. 

- _u ≫ n_ indicates an elementwise bitwise right shift by _n_ , which is typically equivalent to 2 _[−][n] u_ . Similarly, _u ≪ n_ indicates a bitwise left shift by _n_ , which is typically equivalent to 2 _[n] u_ . 

- Square-bracket indexing. For instance _M_ [ _i, j_ ] extracts the element at index _i_ in axis 0 and index _j_ in axis 1, following the zero-based indexing convention. 

## **G.3 Parameter Initialisation** 

The standard initialisation for matrix parameters in our model is rounding 16 times a sample from the standard normal, and casting to I8. This can be precomputed on a CPU since this is only done once at the start of training. 

The egg model has the following parameters (where an additional subscript of _i_ indicates that there is a version of this parameter for each layer of the model): 

- _θ_ emb _∈_ I[256] 8 _[×][D]_ , following standard initialisation. 

- _θ_ head _∈_ I[256] 8 _[×][D]_ , following standard initialisation. 

- _θ_ lnout _∈_ I _[D]_ 8[, initialised to 16 for each element.] 

- _θ_ ln1 _,i, θ_ ln2 _,i ∈_ I _[D]_ 8[, initialised to 16 for each element] 

- _θ_ mlp _,i,_ 1 _∈_ I[4] 8 _[D][×][D]_ and _θ_ mlp _,i,_ 2 _∈_ I _[D]_ 8 _[×]_[4] _[D]_ , following standard initialisation. 

- _θ_ GRU _,i,_ [Wf,Uf,Wh,Uh] _∈_ I _[D]_ 8 _[×][D]_ , following standard initialisation. 

- _θ_ GRU _,i,_ [bfm bh] _∈_ I _[D]_ 8[, initialised to 0 for each element.] 

In total there are 513 _D_ + _l_ (4 _D_ + 12 _D_[2] ) parameters in the model. 

## **G.4 Matrix Multiplication** 

Tensor cores in GPUs are able to calculate fast vector-matrix multiplications with int32 accumulation as _uM ∈_ I _[m]_ 32[where] _[ u][ ∈]_[I] _[n]_ 8[and] _[ M][∈]_[I] _[n]_ 8 _[×][m]_ . For our purposes, we define _u_ @ _M_ as a scaled multiplication: 

**==> picture [96 x 25] intentionally omitted <==**

Note that when _n_ = 4 _[d]_ , the division operation just becomes a right-shift by 4 + _d_ , which is fast to calculate. 

We choose this scaled matrix multiplication because we initialise _M_ to 16 times standard normal samples for each element, so dividing by 16 _[√] n_ preserves the magnitude of _u_ for the output. In particular, if all elements of _u_ and _M_ are drawn from independently from the standard normal distribution multiplied by 16, the central limit theorem tells us that the expected value per element of the output will be 256 _[√] n_ , so dividing by 16 _[√] n_ preserves the standard deviation of 16. 

56 

## **G.5 Embedding** 

Our embedding function takes as input an embedding matrix _θ_ emb _∈_ I[256] 8 _[×][D]_ and an input token _t ∈_ U8, and simply outputs the vector corresponding to that token: _θ_ emb[ _t_ ] _∈_ I _[D]_ 8[.] 

## **G.6 Layer Normalisation (LN)** 

Our layer normalisation operation involves multiplying our input _u ∈_ I _[D]_ 8[with][a][weight] _[θ]_[ln] _[∈]_[I] _[D]_ 8[before] dividing by the mean absolute value of _u_ . 

We decide to divide by the mean absolute value of the input instead of the more common root-mean-squared since square roots are expensive on integers. Note that the _L_ 1 norm after dividing the input by the mean absolute value (when using real numbers) is _D_ instead of 1, which we intentionally choose to preserve more bits of information given the limited range of I8. 

We calculate the mean absolute value of input _u_ as: 

**==> picture [120 x 11] intentionally omitted <==**

Note that we can safely cast the mean absolute value to an I8 without overflow given the properties of the mean of a set, though we lose precision due to truncating the fractional component. 

The output of layernorm is calculated as: 

**==> picture [136 x 11] intentionally omitted <==**

Since division is an expensive operation, we precompute it using a lookup table. Note that the product of two I8 values will always remain in the dynamic range of I16, so our lookup table will be of shape 2[16] _×_ 2[8] . 

## **G.7 MLP** 

Each MLP block consists of two weight parameters: _θ_ 1 _∈_ I[4] 8 _[D][×][D]_ and _θ_ 2 _∈_ I _[D]_ 8 _[×]_[4] _[D]_ . Given an input _u ∈_ I _[D]_ 8[,] we calculate the output as: 

( _u_ @ _θ_ 1 _[T]_[)@] _[θ]_ 2 _[T][.]_ 

Note that we do not use an activation function, because the @ operation is already nonlinear due to the saturated conversion from I32 to I8 

## **G.8 GRU** 

Each GRU block accepts an input vector and state _u, s ∈_ I _[D]_ 8 consists of 6 weight parameters: _θ_ Wf _, θ_ Uf _, θ_ Wh _, θ_ Uh _∈_ I _[D]_ 8 _[×][D]_ and _θ_ bf _, θ_ bh _∈_ I _[D]_ 8[.] 

Using these weight matrices, we calculate the following vectors: 

**==> picture [228 x 64] intentionally omitted <==**

where _h_ is the output and the new hidden state. In the typical GRU, _σ_ stands for the sigmoid function while _ϕ_ stands for the hyperbolic tangent, but we find that setting these as identity operations is sufficient due to the nonlinearity already present in the clipped addition. One can view this clipped addition operation as scaled and shifted version of the “hard" tanh and sigmoid operators. 

To explain why we perform these operations, we can analyse this relative to the original GRU. The _f_ vector for the standard GRU has all elements between 0 and 1 due to the sigmoid, but our elements are between -127 and 127. Therefore, to calculate _f_[ˆ] (which is typically just _f ⊙ s_ ), we first add 127 to _f_ , getting the range between 0 and 254 before multiplying by _s_ before bit-shifting right by 8 again to bring our values back to the I8 dynamic range. We apply similar logic to calculate the final _h_ , which is typically just _h_ = _s_ + _f ⊙_ ( _h_[ˆ] _− s_ ) but needs to be rescaled to keep the int8 dynamic range. 

57 

## **G.9 Fitness Calculation in Integer Types** 

The “fitness” used in language model pretraining is the log-likelihood of correctly generating the next token, treating the outputs of the language model as logits (unnormalised log probabilities). If _t[′] ∈_ U8 is the next token to predict and _y ∈_ I[256] 8 are the logits, we can calculate the log likelihood as follows: 

**==> picture [150 x 27] intentionally omitted <==**

where _o_ is the loss for one token. We implement EXP2 and LOG2 as lookup tables, where 

**==> picture [118 x 27] intentionally omitted <==**

Note that each element in EXP2 for any U8 input requires at most 20 bits, so the sum of exponents across all possible choices is at most 28 bits, meaning we have to precompute LOG2 for 2[28] values. 

## **H EGG Pretraining with Integer EGGROLL** 

The core ideas of EGGROLL still apply in this integer-based training setting, but we have to make some modifications to ensure it only uses integer operations. 

## **H.1 Adding EGGROLL Perturbations** 

For parameter _θ ∈_ I _[m]_ 8 _[×][n]_ that represents a matrix multiplication, we first sample rank-1 perturbation vectors for each index in the batch: _A ∈_ I _[m]_ 8[and] _[ B][∈]_[I] _[n]_ 8[.][We sample these vectors from the standard random normal] multiplied by 16 and rounded to the nearest I8 (clipping if necessary). To prevent the use of floating-point arithmetic on the accelerator, we pre-generate a large matrix of these random values, randomly indexing into it to get the perturbation vectors. 

Given an input _u ∈_ I _[n]_ 8[, instead of calculating] _[ u]_[@] _[θ][T]_[ , we calculate] 

**==> picture [172 x 26] intentionally omitted <==**

ˆ The value of _σ_ is a hyperparameter, related to the _σ_ in the main paper as _σ_ = 2 _[−][σ]_[ˆ] . Note that the batched forward pass remains efficient since it still simply performs a batched vector-vector dot product in int8 (with int32 accumulate) and a batched vector-scalar product in int32. 

We apply this same logic to the embedding matrix, since we can interpret _θ_ [ _t_ ] as one_hot( _t_ ) _θ_ and still apply our rank-1 updates in that context. In practice, this means replacing _u · B_ with _B_ [ _t_ ]. 

## **H.2 Fitness Shaping** 

We employ a simple fitness shaping scheme based on antithetical pairs. Specifically, given raw fitnesses _s_[+] _, s[−]_ , for the positive and negative sample of the antithetical pair respectively, the transformed fitness for the noise is: 

sign( _s_[+] _− s[−]_ ) _,_ 

Note that the only possible values for the fitness after shaping are _{−_ 1 _,_ 0 _,_ 1 _}_ . 

## **H.3 Parameter Update** 

For parameter _θ ∈_ I _[m]_ 8 _[×][n]_ that represents a matrix multiplication (or embedding vector), suppose the sampled batch of rank-1 perturbation vectors are _A ∈_ I _[N]_ 8 _[×][m]_ and _B ∈_ I _[N]_ 8 _[×][n]_ , and let the fitnesses after shaping be _F ∈_ I _[N]_ 8[.][Then we calculate an intermediate value] _[ E][∈]_[I] _[m]_ 32 _[×][n]_ as: 

_E_ = (diag( _F_ ) _A_ ) _[T] B._ 

58 

We use _E_ to determine if each element of _θ_ should be increased or decreased. In particular, when the absolute value of _E_ is above a pre-specified threshold we move _θ_ by one discrete bin in the direction of the sign of _E_ . Since there are only 255 unique values for each element in I8, restricting updates to single bins improves stability without compromising the ability for a parameter to get to any other value with relatively few updates. In particular, we have a real-valued hyperparameter, _α ∈_ (0 _,_ 1) such that the threshold equals 

**==> picture [154 x 25] intentionally omitted <==**

where Φ is the normal cumulative distribution function. Note that this threshold can be precalculated on a CPU. We observe that _α_ approximately equals the fraction of parameters that are updated at each step. 

We currently do not incorporate any momentum or other optimiser states, but this remains critical future work to improve the speed of convergence for pure integer training. 

Across model sizes and population size, we find that setting _σ_ ˆ to 4 and letting _α_ decay over training steps as 1 _._ 015 _t_ +1[gives consistently strong results.] 

## **I EGG Ablations** 

In our main experiments, we use a fixed data batch size of 16 sequences for population sizes 2 and powers of 4 ranging from 4 to 4[10] = 1048576. In this section, we vary the batch size by powers of 4, ranging from 4 to 4[5] = 1024, while varying population size by powers of 4 from 16 to 1048576. When the batch size, _b_ is greater than half of the population size, _N_ , we give each antithetical pair[2] _N[b]_[sequences, functionally giving a cleaner] fitness signal to each member of the population. This also means that the number of parallel "inferences" required is max(2 _b, N_ ). 

**==> picture [324 x 195] intentionally omitted <==**

**----- Start of picture text -----**<br>
Pure Integer Pretraining: Data Batch Size Impact<br>10 [6]<br>Backprop Transformer (fp32)<br>5.5<br>EGGROLL EGG (int8)<br>10 [5]<br>5.0<br>10 [4]<br>4.5<br>10 [3]<br>4.0 10 [2]<br>3.5 10 [1]<br>10 [0]<br>4 [1] 4 [2] 4 [3] 4 [4] 4 [5]<br>Data Batch Size<br>Population Size<br>Final Test Loss (bits/byte)<br>**----- End of picture text -----**<br>


Figure 8: Test loss curves when varying data batch size and population size. 

In Fig. 8, we observe that the final test loss for each population size is relatively constant beyond a specific data batch size threshold. At the top right of the figure, we observe a decrease in loss for small population sizes after _b >[N]_[Ignoring] 2[, which is an artifact of the increased compute usage necessary to use the full data batch.] this artifact, the minimum batch size for near-optimal performance at a given population size _N_ appears to be 4 _N_[6][.][We see that large population sizes need larger data batches for improved performance, since a batch size] of 4 results in nearly identical performance for population sizes 4[9] = 262144 and 4[10] = 1048576, but this diverges as data batch size increases. 

59 

## **J Distributed EGGROLL Framework** 

To facilitate the large-scale experiments, where we scale population sizes beyond 1M, we develop a lightweight distributed training framework designed to minimise network overhead. 

## **J.1 Base-3 Fitness Packing and Bandwidth Efficiency** 

A key bottleneck in distributed training is the communication of gradients or results. We address this via a custom base-3 packing scheme for fitness vectors. Since workers evaluate perturbations in antithetic pairs, the raw signal is discretised into ternary values _{_ +1 _,_ 0 _, −_ 1 _}_ . These are mapped to _{_ 0 _,_ 1 _,_ 2 _}_ and packed five at a time into a single byte: 

**==> picture [71 x 30] intentionally omitted <==**

This yields an effective bitrate of 1 _._ 6 bits per value (near the log2 3 _≈_ 1 _._ 585 theoretical limit). Consequently, the network payload per chunk is approximately 52 + chunk_size _/_ 10 bytes, rendering bandwidth usage independent of model size. 

## **J.2 System Architecture** 

The system employs a Coordinator-Worker topology. The Coordinator maintains the global state and assigns population chunks to Workers. Workers calculate fitness on GPU, apply signal shaping (chunk mean filtering, adaptive thresholding), and return only the packed ternary fitness, minimising traffic significantly compared to standard gradient transmission. 

## **K Fine-tuning of Integer Quantised Models** 

## **K.1 Quantisation Procedure** 

To maximise population throughput and reduce device memory during EGGROLL fine-tuning, we represent the large matrix-multiplication parameters of RWKV in an int8 weight format while keeping non-matmul parameters (e.g., small biases / bookkeeping tensors) in floating point, bf16. Following Jacob et al. (2017), for each weight matrix _W ∈_ R _[d]_[in] _[×][d]_[out] , we use symmetric per-channel int8 quantisation with an absmax scale. For each output channel we first compute: 

**==> picture [122 x 25] intentionally omitted <==**

where _ϵ_ is some small scalar. Then, we store each _si_ in bf16, and quantise weights as 

**==> picture [196 x 25] intentionally omitted <==**

Every matrix parameter is stored as a dictionary containing the quantised weight matrix _Q_ , the scale parameters per channel _{si}∀i ∈_ 1 _, . . . , d_ out and an input scale factor _sx_ in bf16 precision. At runtime, the forward pass is computed by scaling the input vector by _sx_ and the quantised matrix _Q_ with the scales per channel, [ _s_ 1 _, . . . , sd_ out], 

**==> picture [170 x 13] intentionally omitted <==**

## **K.2 Integrating integer-quantised EGGROLL with Adam** 

EGGROLL performs black-box (ES) optimisation directly over the parameter representation used in the forward pass, including integer quantised weights. We integrate this with the Adam optimiser (Kingma & Ba, 2014) by maintaining Adam’s moment estimates in bf16, while enforcing that all quantised tensors remain on the int8 lattice. 

60 

**ES gradients.** EGGROLL estimates gradients via antithetic ES perturbations and score-weighted averaging. This yields a bf16 gradient estimate for: (i) floating-point parameters (when present), (ii) quantised matrix parameters via a low-rank perturbation pathway, and (iii) scale parameters _{si}∀i ∈_ 1 _, . . . , d_ out and _sx_ via explicit scale perturbations. We then pass these gradients to Adam (Optax), which produces an update tensor _u_ for each parameter leaf. 

**Adam updates for int8 tensors (discretised).** For integer parameters (notably int8), Adam produces a real-valued proposal _u_ (stored in bf16). Since the parameter itself must remain int8, we convert this proposal into a sparse unit-step update using a normalised thresholding rule. Let _Q ∈_ Z _[m]_ 8 _[×][n]_ be an int8 tensor and _u ∈_ R _[m][×][n]_ be Adam’s proposed update. We compute a per-tensor z-score normalisation 

**==> picture [76 x 25] intentionally omitted <==**

then apply a threshold _τ_ to form the integer step 

**==> picture [188 x 11] intentionally omitted <==**

Finally we update by unit increments and clip to the valid int8 range: 

**==> picture [126 x 11] intentionally omitted <==**

Intuitively, Adam supplies a magnitude- and history-aware proposal, while the discretisation enforces the integer constraint and yields a stable, sparse update pattern (only entries with sufficiently large normalised updates are modified). 

**Memory considerations.** We store Adam’s optimiser state (moments) in bf16 for all array-valued leaves to reduce memory footprint, while keeping scalar bookkeeping in full precision. This keeps the dominant memory cost of optimisation close to that of the parameters themselves, which is particularly important when fine-tuning large models with large ES populations. 

**Model distillation.** We distil a non-quantised model into the quantised RWKV-7 model by matching the two distributions in teacher forced examples from GSM8k. More specifically, the fitness for a given set of parameters, _µi_ , is computed as follows: 

**==> picture [144 x 30] intentionally omitted <==**

where _x_ 1: _T_ is a subsequence of tokens taken from the solutions of GSM8K and KL ( _pt||qt_ ( _·_ ; _µi_ )) is the Kullback-Leibler divergence between the distribution of the non-quantised model, _pt_ , and the distribution of the quantised model _qt_ over the vocabulary at token _t_ . 

## **L Fine-tuning Pretrained Transformer LLMs with Verifiable Rewards** 

This section describes compares EGGROLL to standard RL from Verifiable Rewards (RLVR). We first describe our experimental results, before including details of the infrastructure used to run these experiments. 

## **L.1 Results** 

Here we demonstrate that EGGROLL can be used to fine-tune pre-trained LLMs on verifiable rewards. We use the vLLM library Kwon et al. (2023) for efficient inference. More infrastructure detail is given in Section L.2. 

We first fine-tune the Qwen3-4B-Base model Yang et al. (2025) on the DeepScaleR Agentica Organization et al. (2025), a dataset of 40k maths questions. As in standard RLVR, the model generates a chain-of-thought (CoT) followed by a final answer. Fitness is then simply calculated by extracting the final answer and comparing it to the ground truth answer Shao et al. (2025). We evaluate performance on MATH500 Hendrycks et al. (2021), OlympiadBench He et al. (2024), AIME24 Balunovi´c et al. (2026), AMC, and MinervaMath Lewkowycz et al. (2022). Training curves are shown in Figure 9. Here we see that fine-tuning with EGGROLL significantly 

61 

improves performance over the base model. In Section L.1 we show final accuracies with EGGROLL and with the equivalent RL experiment. The RL values are taken from Liu et al. (2025), and we match all the relevant shared hyperparameters and setup, such as maximum response length and prompt phrasing. We see that EGGROLL is able to match the RL optimisation with very minimal hyperparameter tuning, a LoRA rank of 1 and a moderately small population size of 2048. Full hyperparameter details are given in Table 3. 

**==> picture [390 x 163] intentionally omitted <==**

Figure 9: Training curves for fine-tuning Qwen3-4B-Base on the DeepScaleR math dataset. Similar to RL from Verifiable Rewards (RLVR), we see that optimising with EGGROLL is able to improve chain-of-thought reasoning performance on a range of math benchmarks. 

||MATH500|OlympiadBench|AIME24|AMC|MinervaMath|**Average**|
|---|---|---|---|---|---|---|
|_Qwen3-4B-Base_|50.2|24.4|10.0|33.7|21.7|28.0|
|+EGGROLL|**75.8**|**37.3**|13.3|**49.4**|31.3|**41.4**|
|+RL|67.4|33.5|**16.7**|**49.4**|**40.1**|**41.4**|



Table 1: Final test accuracies when training on the DeepScaleR dataset to optimise verifiable rewards with EGGROLL and RL. We see that EGGROLL significantly boosts performance from the base model and is able to match the equivalent RL experiment. 

Since EGGROLL can be used to optimise non-differentiable objectives we next try optimising for pass@k. While zero-shot (pass@1) is differentiable, the pass@k objective is not as it depends on multiple samples from the model. This means it cannot be optimised easily with RL. In Figure 10 we fine-tune the Qwen3-1.7B model on the DeepScaleR dataset with a population size of 256, LoRA rank 1, and _K_ = 4. We see that EGGROLL successfully optimises both the pass@1 (differentiable) and pass@k (non-differentiable) objectives. In Figure 10 _(right)_ we plot the number of distinct answers in 4 samples from the model. We see then when optimising for pass@k the answer diversity sampled by the model increases over training, whereas when optimising for zero-shot (pass@1) the model collapses towards a single final answer. 

## **L.2 Training Infrastructure for Large-Scale Transformer LLMs** 

EGGROLL facilitates the fine-tuning of transformer-based LLMs at scale. We achieve this by repurposing the vLLM inference engine, leveraging its high-throughput kernel implementations and native support for multi-LoRA serving. The system utilises vLLM’s native Tensor Parallelism (TP) to shard the model weights across the GPUs within a node, while cross-node parallelisation is employed for the concurrent evaluation of the LoRA population. 

To render ES-based optimisation feasible and efficient across a wide range of model sizes, we implement several critical systems-level optimisations: 

**Custom WorkerExtension and Sharding-Aware Updates** By implementing a custom WorkerExtension, we effectively convert the vLLM inference engine into a training-capable runtime. This extension allows the optimisation logic to reside within the GPU process space, enabling direct, 

62 

**==> picture [260 x 123] intentionally omitted <==**

Figure 10: Using EGGROLL to optimise non-differentiable objectives. _Left_ : Fitness curves comparing training with pass@1 (differentiable) versus pass@k (non-differentiable), where _K_ = 4. _Right_ : The mean number of unique final answers generated per 4-sample set. We observe that when optimizing for pass@k increases answer diversity, whereas optimizing for zero-shot accuracy (pass@1) reduces it. 

in-place manipulation of the model’s weights. A significant complexity of this integration is vLLM’s internal tensor parallelism, which frequently fuses weights (e.g. combining q_proj, k_proj, and v_proj into a single qkv_proj tensor). Our update mechanism is explicitly “sharding-aware”; it constructs a dictionary which maps individual LoRA updates to the specific fused slices held by each local GPU rank. This ensures that the global ES update is mathematically consistent across all distributed shards. 

**Layer-wise Memory Management** To prevent out-of-memory (OOM) errors during the update phase, the WorkerExtension performs the ES weight application in a streaming, layer-wise fashion. By processing one layer at a time and clearing temporary buffers, the memory overhead of the update remains independent of the total model depth. This allows for the fine-tuning of models of very different sizes with a VRAM footprint barely exceeding that of standard inference. 

**Direct GPU-to-GPU Weight Synchronization** After computing the ES update on the primary rank, we broadcast the updated parameters to all model instances using NCCL via PyNcclCommunicator. This approach bypasses CPU-based communication and instead uses hardware interconnects to transfer weights directly between GPUs, preventing synchronization from becoming a bottleneck when scaling to more nodes. 

**Meta-Device Blueprinting** To initialise models that exceed the physical RAM of the control node, we employ Meta-Device Initialisation. Using accelerate’s init_empty_weights, we instantiate a “meta” version of the model to derive the weight shapes and sharding requirements for the LoRA adapters. This allows the system to generate a complete parameter blueprint for models of arbitrary size without ever allocating the full weight tensors in system memory. 

**vLLM Engine Settings** Throughout the different experiments with vLLM, we use the following engine settings. These generally allow for high throughput across model sizes (e.g. at least 800 tokens/second), but we haven’t performed hyperparameter sweeps, so potentially faster, more memory-efficient settings may be used for improved results. 

63 

|Parameter|Value|
|---|---|
|Tensor parallel size|2,4|
|Data type|auto|
|Enable prefx caching|True|
|Enforce eager execution|True|
|Enable LoRA|True|
|Max LoRAs|_⌈_population_size_/_num_engines_⌉_|
|GPU memory utilisation|0.90|
|Max number of sequences|384|
|Max model length|max(1024_,_512 +max_tokens)|
|Max batched tokens|prompt_batch_size_×_1024|
|Load format|auto|



Table 2: vLLM engine configuration parameters to allow for high throughput EGGROLL training on large-scale transformer LLMs. 

|Parameter|Value|
|---|---|
|Population size|256, 2048|
|Sigma|0.001|
|Learning Rate|0.001|
|Max Response Length|4096|
|Temperature|0.0, 0.7|
|Samples Per Prompt|1, 4|
|Pass at K|True, False|
|LoRA Rank|1|
|LoRA Reuse Steps|4|



Table 3: Hyperparameters for the verifiable reward transformer fine-tuning experiments in Section L.1. 

## **M Fine-tuning Time Series Foundation Model: High-Frequency Trading** 

The preceding experiments demonstrate the effectiveness of EGGROLL on natural language reasoning tasks. We now investigate whether EGGROLL can effectively fine-tune pretrained foundation models on a fundamentally different data modality: structured time series. We focus on high-frequency trading (HFT) for two reasons. First, HFT generates data at an unprecedented scale. The S&P 500 constituents alone produced approximately 3.8 trillion tokens of order flow data between 2016 and 2021, comparable to the largest natural language corpora. Second, the domain presents a well-defined downstream task (order execution) with a natural reward signal: the realised profit and loss, also known as PnL, making it amenable to fine-tuning via evolution strategies. 

Order execution takes place in limit order books (LOBs), which are the mechanism upon which modern financial exchanges operate (Gould et al., 2013; Bouchaud et al., 2018). They allow market participants to submit limit orders that specify the details of intended transactions. Specifically, each limit order contains the order type, direction, price, and quantity. The continuous stream of these orders is known as the order flow. LOBs aggregate the limit orders that have not been matched yet. Unlike natural language, where tokens are purely symbolic, order flow messages comprise both categorical values (e.g., order type, direction) and numerical values (e.g., price, quantity) in which magnitude carries semantic meaning. This structure provides a distinct test of EGGROLL’s ability to fine-tune foundation models on time series sequential data. 

A central objective in this context is order execution, which consists of buying or selling a specified quantity of an asset within a given time window. The goal is to maximise profit by transacting at favourable prices. In prior reinforcement learning approaches to this problem, the action space is usually simplified (Frey et al., 2023; Mohl et al., 2025; Ning et al., 2021). In contrast, we aim to give the model full flexibility in choosing 

64 

**==> picture [282 x 108] intentionally omitted <==**

**----- Start of picture text -----**<br>
12000 Baseline Baseline<br>EGGROLL 3000 EGGROLL<br>11000<br>2500<br>10000<br>9000 2000<br>8000 1500<br>7000 1000<br>6000<br>500<br>5000<br>0<br>0 1000 2000 3000 4000 5000 6000 7000 0 1000 2000 3000 4000 5000 6000 7000<br>Epoch Epoch<br>PnL Mean PnL Std<br>**----- End of picture text -----**<br>


Figure 11: Training curves for order execution with EGGROLL. **Left** : Mean PnL over training epochs for the baseline ( _σ_ = 0, orange dashed) and EGGROLL ( _σ_ = 0 _._ 01, blue solid). **Right** : PnL standard deviation over training epochs. Shaded regions indicate the interquartile range across runs. 

limit orders, i.e., to freely choose the order type, direction, price, and quantity. We achieve this by tokenising the limit order book messages and providing the model with a token-level action space. 

Foundation models have recently been used to generate synthetic order flow (Nagy et al., 2023; Li et al., 2025) and have been shown to replicate realistic market behaviour (Nagy et al., 2025) through next-token prediction. We therefore first pretrain a foundation model on tokenised limit order book messages, and then fine-tune it using EGGROLL for the order execution task. The pretraining follows the approach of Nagy et al. (2023): we employ an S5 model architecture (Smith et al., 2023) that generates next-token probabilities, with cross-entropy as the training loss. The pretraining is conducted on the LOBSTER data set (Huang & Polak, 2011) for the Google stock (GOOG) in 2022, which contains around 25B tokens. 

Subsequently, we fine-tune the model using EGGROLL. The training parameters are summarised in Table 4. The task is to execute a sell order of _Q_ = 30 shares within a horizon of _T_ = 10 steps. In each episode, the LOB is initialised based on a LOB snapshot followed by 10 warm-up background messages. In each step, the population members generate their messages, which are then followed by 50 real background data messages. The orders are executed using the Jax-LOB (Frey et al., 2023) simulator. We perform the fine-tuning on a fixed time window for GOOG in January 2023. Following (Galim et al., 2025), we apply LoRA with rank 4 on all projection matrices while freezing SSM parameters and layer norms. Performance is evaluated using PnL based on the executed prices and the initial mid price. Specifically, for a sell task of total quantity _Q_ , the PnL is computed as 

**==> picture [76 x 30] intentionally omitted <==**

where _qi_ and _pi_ denote the quantity and price of the _i_ -th executed trade and _P_ mid[init][is][the][mid-price][at][the] beginning of the execution window. If the agent does not execute the entire quantity by the end of the episode, an automatic market order is submitted selling the remaining quantity. To improve robustness to outliers, fitness is defined as a rank-based transformation of the PnL. Specifically, for a population of size _M_ , the PnL values 

**==> picture [108 x 11] intentionally omitted <==**

are mapped to the interval [ _−_ 0 _._ 5 _,_ 0 _._ 5], where rank(PnL _i_ ) _∈{_ 0 _, . . . , M −_ 1 _}_ denotes the rank of the _i_ -th individual’s PnL: 

**==> picture [98 x 22] intentionally omitted <==**

Training curves over 6,500 epochs are shown in Figure 11. The baseline policy ( _σ_ = 0), corresponding to the pretrained model, achieves a mean PnL of approximately 4,700. In contrast, EGGROLL fine-tuning ( _σ_ = 0 _._ 01) improves the mean PnL to around 12,000, corresponding to a roughly 155% improvement over the baseline. The right panel of Figure 11 depicts the PnL standard deviation during fine-tuning: it initially increases to around 3,100 during the first 2,500 epochs, which corresponds to an exploration phase where the population tries out diverse strategies, before decreasing to approximately 400 by the end of training, indicating that the population concentrates around a high-performing policy. 

65 

|Hyperparameter|Value|
|---|---|
|Model|LOBS5-360M|
|Parallel generations per GPU|2,048|
|Total parallel generations|65,536|
|LoRA rank|4|
|Sigma|0.01|
|Learning rate_η_|0.001|
|Epochs|6,500|



Table 4: Model and EGGROLL fine-tuning settings for high-frequency trading. 

**==> picture [360 x 206] intentionally omitted <==**

**----- Start of picture text -----**<br>
Simple Spread Simple Speaker Listener Simple Reference<br>−40<br>−40 −50<br>−60 −100 −50<br>IPPO<br>−80 −60 OpenES<br>−150<br>EGGROLL<br>0 2 4 0 2 4 0 2 4<br>Steps 1e8 Steps 1e8 Steps 1e8<br>15<br>5<br>2.0<br>4<br>10<br>1.5<br>3<br>2 1.0<br>5<br>1 0.5<br>0 0 0.0<br>IPPO OpenES EGGROLL IPPO OpenES EGGROLL IPPO OpenES EGGROLL<br>(Batch Size = 128) (Batch Size = 512) (Batch Size = 4096)<br>Return<br>Wall Clock Time (mins)<br>**----- End of picture text -----**<br>


Figure 12: Training curves and wall clock times for cooperative Multi Particle Environments. Hyperparameter optimisation yielded equal batch sizes for all algorithms on the same environment. All EGGROLL runs used rank 1 perturbations. Shaded regions are standard errors of mean values. 

## **N Experimental Details** 

## **N.1 Multi Agent Reinforcement Learning Experiments** 

Table 5: Hyperparameter Ranges Used in MPE Sweeps for EGGROLL and OpenES 

Table 6: Hyperparameter Ranges Used in MPE Sweeps for IPPO 

|Hyperparameter<br>Values<br>activation<br>pqn, tanh<br>pop_size<br>128, 512, 1024, 2048, 4096<br>learning_rate<br>0.01, 0.05, 0.1, 0.5<br>lr_decay<br>0.3, 0.7, 1.0<br>sigma<br>0.1, 0.2, 0.3, 0.4, 0.5<br>rank_transform<br>true,false||
|---|---|
||Hyperparameter<br>Values|
||activation<br>relu, tanh<br>pop_size<br>128, 512, 1024, 2048, 4096<br>learning_rate<br>5e-5, 1e-4, 2.5e-4, 1e-3<br>entropy_coef<br>0.001,0.005,0.01|
|||



66 

Table 7: MPE Simple Spread v3 

Table 8: MPE Simple Speaker Listener v4 

|Hyperparameter<br>eggroll<br>open_es<br>ippo<br>activation<br>tanh<br>tanh<br>tanh<br>deterministic_policy<br>true<br>true<br>false<br>learning_rate<br>0.01<br>0.01<br>0.001<br>lr_decay<br>0.7<br>0.7<br>linear<br>layer_size<br>64<br>64<br>64<br>n_layers<br>3<br>3<br>3<br>pop_size<br>128<br>128<br>128<br>optimizer<br>adamw<br>adamw<br>adam<br>rank<br>1<br>1<br>-<br>rank_transform<br>false<br>false<br>-<br>sigma<br>0.5<br>0.5<br>-<br>n_minibatches<br>-<br>-<br>4<br>update_epochs<br>-<br>-<br>4<br>gamma<br>-<br>-<br>0.99<br>gae_lambda<br>-<br>-<br>0.95<br>epsilon_clip<br>-<br>-<br>0.2<br>entropy_coef<br>-<br>-<br>0.01<br>value_coef<br>-<br>-<br>0.5<br>max_grad_norm<br>-<br>-<br>0.5|Hyperparameter<br>eggroll<br>open_es<br>ippo|
|---|---|
||activation<br>tanh<br>tanh<br>relu<br>deterministic_policy<br>true<br>true<br>false<br>learning_rate<br>0.01<br>0.01<br>0.001<br>lr_decay<br>0.7<br>0.3<br>linear<br>layer_size<br>64<br>64<br>64<br>n_layers<br>3<br>3<br>64<br>pop_size<br>512<br>512<br>512<br>optimizer<br>adamw<br>adamw<br>adam<br>rank<br>1<br>1<br>-<br>rank_transform<br>true<br>true<br>-<br>sigma<br>0.5<br>0.5<br>-<br>n_minibatches<br>-<br>-<br>4<br>update_epochs<br>-<br>-<br>4<br>gamma<br>-<br>-<br>0.99<br>gae_lambda<br>-<br>-<br>0.95<br>epsilon_clip<br>-<br>-<br>0.2<br>entropy_coef<br>-<br>-<br>0.005<br>value_coef<br>-<br>-<br>0.5<br>max_grad_norm<br>-<br>-<br>0.5|



Table 9: MPE Simple Reference v3 

|Hyperparameter|eggroll|open_es|ippo|
|---|---|---|---|
|activation|pqn|tanh|relu|
|deterministic_policy|true|true|false|
|learning_rate|0.01|0.01|0.001|
|lr_decay|0.3|0.3|linear|
|layer_size|64|64|64|
|n_layers|3|3|3|
|pop_size|4096|4096|4096|
|optimizer|adamw|adamw|adam|
|rank|1|1|-|
|rank_transform|false|true|-|
|sigma|0.1|0.3|-|
|n_minibatches|-|-|4|
|update_epochs|-|-|4|
|gamma|-|-|0.99|
|gae_lambda|-|-|0.95|
|epsilon_clip|-|-|0.2|
|entropy_coef|-|-|0.01|
|value_coef|-|-|0.5|
|max_grad_norm|-|-|0.5|



We train on three cooperative Multi Particle Environments (MPEs) (Lowe et al., 2017) implemented in JaxMARL (Rutherford et al., 2024) with feed-forward networks of width 64 and depth 3, performing Bayesian hyperparameter optimisation for each environment and algorithm. All runs were executed on NVIDIA A100-SXM4-40GB GPUs. We find that the optimal batch size is consistent across algorithms on the same environment. Figure 12 shows that EGGROLL with rank 1 trains up to 2.4 times faster than OpenES for large batch sizes while staying competitive in performance. 

67 

**==> picture [433 x 159] intentionally omitted <==**

**----- Start of picture text -----**<br>
Math Reasoning — RWKV 7g14B<br>Countdown — RWKV 7g7B<br>0.33<br>0.7   OpenES: Qwen-2.5-7B (0.668) 0.30<br>0.3<br>0.6<br>  GRPO: Qwen-2.5-7B (0.528)<br>0.5<br>0.2<br>0.4 0.13 0.13<br>  Original: Qwen-2.5-7B (0.312) 0.11<br>0.3 0.1 0.07<br>0.2<br>0.0<br>0 100 200 300 400 500 AIME24 AIME25 HMMT25<br>Epoch<br>EGGROLL: RWKV-g0:7g7B Base model EGGROLL<br>(a) (b)<br>Validation Score Validation Score<br>**----- End of picture text -----**<br>


Figure 13: (a) Comparison of our finetuned RWKV 7G 7 billion parameter model using 8 GPUS with the results reported by Qiu et al. (2025) on similarly sized Qwen models. (b) Performance of our finetuned RWKV 7G 14 billion parameter model on hard reasoning tasks using 32 GPUs for 12 hours. The model was trained using the DeepScaleR dataset and the best checkpoint was chosen by evaluating on AIME24. Due to the size of the model we were not able to run similar baseline experiments using GRPO. 

## **N.2 Reasoning Fine-tuning Experiments: Countdown** 

We ran a Bayesian hyper-parameter sweep (Snoek et al., 2012) for both GRPO and EGGROLL and used the best set found to run the experiments in figure 4b. For GRPO we swept over sampling temperature and learning rate, whereas for EGGROLL we swept over the standard deviation of the ES sampling ( _σ_ ) and the learning rate scale. The best hyper-parameters found are detailed on tables 10 (EGGROLL) and 11 (GRPO). All of the experiments run in 8 hours on a NVIDIA H200 GPU. 

|Hyperparameter|Value|
|---|---|
|Model|RWKV 7g1.5B|
|Optimiser|Gradient descent|
|ES standard deviation_σ_|7_×_10_−_4|
|Rank_r_|1|
|Learning-rate scale_η_scale|0.125|
|Population size|256|
|Parallel generations per GPU|1536|
|Prompts per epoch|6|
|Generation / thinking length|1000 tokens|
|Train / val temperature|0 / 0|
|Parallel validations|128|



Table 10: Key hyperparameters for EGGROLL training on Countdown with FastRWKV-7g1.5B. 

We also run an experiment where we increase the number of GPUs to 8 and use a bigger model, RWKV 7g7B, on the Countdown task, allowing for stronger final performance. Notably, we compare to the results reported by Qiu et al. (2025) on Countdown. Figure 13a shows that starting from our significantly weaker model (RWKV 7g7B v.s. Qwen 2.5-7B), we are able to train to a higher validation accuracy (72.9%), v.s. the ones reported for training with GRPO (52.8%) and Open ES (66.8%). Qiu et al. (2025) do not report the wall clock time or the hardware used for their experiments which makes it difficult to establish a fair comparison. 

68 

|Hyperparameter|Value|
|---|---|
|Model|RWKV 7g1.5B|
|Optimiser|Radam|
|Learning rate_η_|3_×_10_−_6|
|Generations per prompt_G_|8|
|Parallel generations per GPU|64|
|Prompts per epoch|8|
|Generation length|1000 tokens|
|Number of minibatches|4|
|PPO clip parameter_ϵ_clip|0.2|
|Train / val temperature|1 / 0|
|Parallel validations|128|



Table 11: Key hyperparameters for GRPO training on Countdown with AssociativeScanRWKV-7g1.5B. 

## **N.3 Reasoning Fine-tuning Experiments: GSM8K** 

We used the hyper-parameters found for Countdown as a starting point and reduced the learning rates for both GRPO and EGGROLL using linear search until we found the best performing one on the validation set. Our experiments for GSM8K run on 8 NVIDIA H200 GPUS for 8 hours each. We also increase the standard deviation, _σ_ , parameter for ES (from 7 _×_ 10 _[−]_[4] to 2 _×_ 10 _[−]_[3] ) as the significantly bigger population sizes (8096 v.s. 512) allow for much more stable training and aggressive exploration. 

|Hyperparameter|Value|
|---|---|
|Model|RWKV 7g7B|
|ES standard deviation_σ_|2_×_10_−_3|
|Rank_r_|1|
|Learning-rate scale_η_scale|0.06|
|Generations per prompt_G_|512|
|Parallel generations per GPU|1024|
|Total parallel generations|8192|
|Prompts per epoch|16|
|Generation length|1000 tokens|
|Noise reuse factor|1|
|Freeze non-LoRA params|True|
|Train / val temperature|0 / 0|
|Parallel validations|128|



Table 12: Key hyperparameters for multi-GPU EGGROLL training on GSM8K with FastRWKV-7g7B. 

## **N.4 Reinforcement Learning Experiments** 

Next, we compare the performance of EGGROLL against standard OpenES as implemented in Salimans et al. (2017) on reinforcement learning tasks. Given the small network sizes, we can use OpenES at this scale, but as network sizes increase, the use of vanilla OpenES becomes computationally infeasible. We use the standard formulation of simply optimising for the final return in the environment. For both EGGROLL and OpenES, we perform hyperparameter optimisation (HPO) separately for each environment. For each algorithm–environment pair, we define plausible ranges for all key hyperparameters based on prior work and preliminary experiments. We then perform 20 random search trials, where each trial corresponds to a single training run with a randomly sampled hyperparameter configuration. Each configuration is evaluated based on the final return achieved by the mean policy parameters at the end of training. After all trials, we select the configuration that yields the highest final return. Using this best configuration, we then run 10 independent seeds to evaluate performance and report the mean and standard error of the mean across these seeds. 

69 

**==> picture [432 x 433] intentionally omitted <==**

**----- Start of picture text -----**<br>
CartPole-v1 Pendulum-v1 Brax Ant Brax Humanoid<br>1.0 1.0 1.0 1.0<br>0.8 0.8 0.8 0.8<br>0.6 0.6 0.6 0.6<br>0.4 0.4 0.4 0.4<br>EGGROLL<br>0.2 OpenES 0.2 0.2 0.2<br>PPO 0.0<br>0.0 0.0 0.0<br>0 2 4 0 2 4 0 2 4 0 2 4<br>Steps 1e8 Steps 1e8 Steps 1e8 Steps 1e8<br>Brax Inverted Double Pendulum Craftax Classic Craftax Symbolic Jumanji 2048<br>1.2 1.0 1.0 1.0<br>1.0<br>0.8 0.8 0.8<br>0.8<br>0.6 0.6 0.6<br>0.6<br>0.4 0.4 0.4 0.4<br>0.2 0.2 0.2 0.2<br>0.0 0.0 0.0<br>0 2 4 0 2 4 0 2 4 0 2 4<br>Steps 1e8 Steps 1e8 Steps 1e8 Steps 1e8<br>Jumanji Knapsack Jumanji Snake Kinetix Hard Pinball (l) Kinetix Thrust Control Left (m)<br>1.0 1.0 1.00<br>1.75<br>0.75<br>0.8 0.8 1.50<br>0.50<br>1.25<br>0.6 0.6 0.25<br>1.00<br>0.4 0.4 0.00 0.75<br>−0.25 0.50<br>0.2<br>0.2 −0.50 0.25<br>0.0 −0.75 0.00<br>0 2 4 0 2 4 0 2 4 0 2 4<br>Steps 1e8 Steps 1e8 Steps 1e8 Steps 1e8<br>Kinetix Thrust Over Ball (s) Navix DoorKey (8x8) Navix Dynamic Obstacles Random (6x6) Navix FourRooms (8x8)<br>1.0 1.0 1.0<br>1.04<br>0.8 0.8 0.8<br>1.02<br>0.6 0.6<br>1.00 0.6<br>0.4 0.4<br>0.98 0.4<br>0.2 0.2<br>0.96<br>0.2<br>0.0 0.0<br>0 2 4 0 2 4 0 2 4 0 2 4<br>Steps 1e8 Steps 1e8 Steps 1e8 Steps 1e8<br>Normalized Return Normalized Return Normalized Return Normalized Return<br>Normalized Return Normalized Return Normalized Return Normalized Return<br>Normalized Return Normalized Return Normalized Return Normalized Return<br>Normalized Return Normalized Return Normalized Return Normalized Return<br>**----- End of picture text -----**<br>


Figure 14: Comparison of reinforcement learning results: Mean returns for each environment and algorithm across 10 random seeds. The returns are evaluated using the mean of the parameters. HPO was conducted for each algorithm/environment pair. The shaded region is the standard error of the mean. 

70 

|Hyperparameter|Value|
|---|---|
|Model|RWKV 7g7B|
|Learning rate_η_|1_×_10_−_6|
|Generations per prompt_G_|8|
|Parallel generations per GPU|32|
|Total parallel generations|256|
|Prompts per epoch|32|
|Generation length|1000 tokens|
|Number of minibatches|16|
|Number of workers (processes)|8|
|PPO clip parameter_ϵ_clip|0.2|
|Train / val temperature|1 / 0|
|Parallel validations|128|



Table 13: Key hyperparameters for multi-GPU GRPO training on GSM8K with AssociativeScanRWKV-7g7B. 

|Hyperparameter|Value|
|---|---|
|Model|RWKV 7g7B|
|Optimiser|EGGROLL (Quantised))|
|ES standard deviation_σ_|0.4|
|Rank_r_|1|
|Learning-rate scale_η_scale|3_×_10_−_7|
|Population size|8192|
|Parallel generations per GPU|256|
|Prompts per epoch|1|
|Generation / thinking length|256 tokens|
|Train / val temperature|0 / 0|
|Parallel validations|128|



Table 14: Key hyperparameters for quantised EGGROLL training on GSM8K (teacher-forced) with RWKV-7g7B. 

We use policy networks with 3 layers of 256 neurons and a range of environments that demonstrate different capabilities. We evaluate across the Navix (Pignatelli et al., 2024), Craftax (Matthews et al., 2024), Brax (Freeman et al., 2021), Kinetix (Matthews et al., 2025), and Jumanji (Bonnet et al., 2024) suites of environments. We evaluate 16 environments in total. We choose environments that are not trivial or impossible for PPO to solve, according to the original papers. We also choose environments that belong to different categories (e.g., environment size in Kinetix or categories in Jumanji). 

We show a subsample of the evaluated environments in Fig. 4a with the remaining results and hyperparameter details in Appendix N.4. Our findings show that EGGROLL is competitive with Open ES on 7/16 environments, underperforms on 2/16, and outperforms on 7/16. This does not take into account the speed-ups when compared to using OpenES with full-rank updates (see Figure 15). We postulate that the reason for this performance increase is that the large networks are difficult to optimise for OpenES and lend themselves well to low-rank updates. 

We present here the hyperparameter ranges we used for hyperparameter optimisation, as well as all hyperparameter settings for all the experiments. All RL experiments were run on an NVIDIA L40S GPU. For PPO, we use the same methodology to tune the hyperparameters as we did for OpenES and EGGROLL as described in Section 6.2. We report the ranges and the final hyperparameters here. We train PPO agents using Rejax (Liesen et al., 2024). We use the activation function from Gallici et al. (2025) in our experiments, which we refer to as the “pqn” activation function in our hyperparameter tables. 

71 

**==> picture [432 x 433] intentionally omitted <==**

**----- Start of picture text -----**<br>
CartPole-v1 Pendulum-v1 Brax Ant Brax Humanoid<br>1.83x faster 7.81x faster 2.47x faster 1.80x slower<br>10000<br>400<br>300<br>300 10000 7500<br>200<br>200 5000<br>5000<br>100 100 2500<br>0 0 0 0<br>EGGROLL OpenES EGGROLL OpenES EGGROLL OpenES EGGROLL OpenES<br>Brax Inverted Double Pendulum Craftax Classic Craftax Symbolic Jumanji 2048<br>15000 30000<br>1.65x slower 1.60x faster 1.29x slower 5.26x faster<br>3000 600<br>10000 20000<br>2000 400<br>5000 10000<br>1000 200<br>0 0 0 0<br>EGGROLL OpenES EGGROLL OpenES EGGROLL OpenES EGGROLL OpenES<br>Jumanji Knapsack Jumanji Snake Kinetix Hard Pinball (l) Kinetix Thrust Control Left (m)<br>11.17x faster 40.68x faster 28.54x faster 4000 1.60x faster<br>1500<br>15000<br>40000<br>3000<br>1000 10000<br>2000<br>20000<br>500 5000<br>1000<br>0 0 0 0<br>EGGROLL OpenES EGGROLL OpenES EGGROLL OpenES EGGROLL OpenES<br>Kinetix Thrust Over Ball (s) Navix DoorKey (8x8) Navix Dynamic Obstacles Random (6x6) Navix FourRooms (8x8)<br>4000<br>1.96x slower 1.61x faster 3.14x faster 2.25x faster<br>400 3000 1500<br>3000<br>2000 1000<br>2000<br>200<br>1000 1000 500<br>0 0 0 0<br>EGGROLL OpenES EGGROLL OpenES EGGROLL OpenES EGGROLL OpenES<br>Time (seconds) Time (seconds) Time (seconds) Time (seconds)<br>Time (seconds) Time (seconds) Time (seconds) Time (seconds)<br>Time (seconds) Time (seconds) Time (seconds) Time (seconds)<br>Time (seconds) Time (seconds) Time (seconds) Time (seconds)<br>**----- End of picture text -----**<br>


Figure 15: Comparison of reinforcement learning results: Mean and standard deviation of training time. Note that some of the timing difference is due to the differences in episode lengths, which is why the total time for EGGROLL sometimes appears longer than OpenES despite EGGROLL being faster on a per-timestep basis. 

Table 15: Hyperparameter Ranges for EGGROLL and OpenES 

|**Hyperparameter**|**Values**|
|---|---|
|pop_size|512, 1024, 2048, 4096|
|n_parallel_evaluations|1, 4, 8|
|rank|1, 2, 4|
|optimizer|adamw, sgd, adam|
|learning_rate|1e-3, 1e-2, 1e-1|
|lr_decay|0.995, 0.999, 0.9995, 1.0|
|sigma|0.05, 0.2, 0.5|
|sigma_decay|0.995, 0.999, 0.9995, 1.0|
|rank_transform|true, false|
|deterministic_policy|true,false|



72 

Table 16: Hyperparameter Ranges for PPO 

|**Hyperparameter**|**Values**|
|---|---|
|clip_eps|0.1, 0.2, 0.3|
|ent_coef|0, 0.0001, 0.001|
|gae_lambda|0.9, 0.95, 0.98|
|gamma|0.95, 0.99, 0.995, 0.999|
|learning_rate|0.0001, 0.0003, 0.001|
|max_grad_norm|0.5, 1, 2|
|layer_size|256|
|n_layers|3|
|normalize_observations|true|
|normalize_rewards|false|
|num_envs|64, 128, 256|
|num_epochs|4, 8, 16|
|num_minibatches|16, 32, 64|
|num_steps|64, 128, 256|
|reward_normalization_discount|0.99|
|skip_initial_evaluation|false|
|vf_coef|0.5,0.75,1|



Table 17: CartPole-v1 

Table 18: Pendulum-v1 

|Hyperparameter|eggroll|open_es|
|---|---|---|
|activation|pqn|pqn|
|deterministic_policy|false|true|
|learning_rate|0.1|0.1|
|lr_decay|0.9995|0.9995|
|layer_size|256|256|
|n_layers|3|3|
|n_parallel_evaluations|1|4|
|pop_size|2048|512|
|optimizer|sgd|adamw|
|rank|4|/|
|rank_transform|false|true|
|sigma|0.2|0.5|
|sigma_decay|0.999|0.9995|



|Hyperparameter|eggroll|open_es|
|---|---|---|
|activation|pqn|pqn|
|deterministic_policy|false|true|
|learning_rate|0.01|0.01|
|lr_decay|0.995|0.995|
|layer_size|256|256|
|n_layers|3|3|
|n_parallel_evaluations|1|4|
|pop_size|4096|4096|
|optimizer|adam|adamw|
|rank|4|/|
|rank_transform|false|false|
|sigma|0.05|0.05|
|sigma_decay|0.995|1|



Table 19: brax/ant 

Table 20: brax/humanoid 

|Hyperparameter|eggroll|open_es|
|---|---|---|
|activation|pqn|pqn|
|deterministic_policy|false|false|
|learning_rate|0.01|0.1|
|lr_decay|0.9995|0.995|
|layer_size|256|256|
|n_layers|3|3|
|n_parallel_evaluations|1|8|
|pop_size|2048|512|
|optimizer|adam|adam|
|rank|1|/|
|rank_transform|false|false|
|sigma|0.05|0.05|
|sigma_decay|0.9995|0.9995|



|Hyperparameter|eggroll|open_es|
|---|---|---|
|activation|pqn|pqn|
|deterministic_policy|true|false|
|learning_rate|0.1|0.1|
|lr_decay|1|0.995|
|layer_size|256|256|
|n_layers|3|3|
|n_parallel_evaluations|8|8|
|pop_size|4096|1024|
|optimizer|adam|sgd|
|rank|1|/|
|rank_transform|true|true|
|sigma|0.2|0.2|
|sigma_decay|0.9995|0.995|



73 

Table 21: brax/inverted_double_pendulum 

|Hyperparameter|eggroll|open_es|
|---|---|---|
|activation|pqn|pqn|
|deterministic_policy|true|true|
|learning_rate|0.1|0.1|
|lr_decay|1|0.995|
|layer_size|256|256|
|n_layers|3|3|
|n_parallel_evaluations|1|1|
|pop_size|2048|4096|
|optimizer|adam|adam|
|rank|2|/|
|rank_transform|true|true|
|sigma|0.5|0.05|
|sigma_decay|0.995|1|



Table 23: craftax/Craftax-Symbolic-AutoReset-v1 

|Hyperparameter|eggroll|open_es|
|---|---|---|
|activation|pqn|pqn|
|deterministic_policy|false|false|
|learning_rate|0.01|0.1|
|lr_decay|0.999|0.995|
|layer_size|256|256|
|n_layers|3|3|
|n_parallel_evaluations|1|4|
|pop_size|512|1024|
|optimizer|sgd|adam|
|rank|4|/|
|rank_transform|true|false|
|sigma|0.05|0.5|
|sigma_decay|0.999|1|



Table 25: jumanji/Knapsack-v1 

|Hyperparameter|eggroll|open_es|
|---|---|---|
|activation|pqn|pqn|
|deterministic_policy|false|false|
|learning_rate|0.1|0.01|
|lr_decay|0.999|1|
|layer_size|256|256|
|n_layers|3|3|
|n_parallel_evaluations|4|1|
|pop_size|1024|2048|
|optimizer|sgd|adamw|
|rank|4|/|
|rank_transform|true|true|
|sigma|0.05|0.5|
|sigma_decay|1|0.995|



Table 22: craftax/Craftax-Classic-Symbolic-AutoReset-v1 

|Hyperparameter|eggroll|open_es|
|---|---|---|
|activation|pqn|pqn|
|deterministic_policy|false|false|
|learning_rate|0.01|0.001|
|lr_decay|0.995|0.995|
|layer_size|256|256|
|n_layers|3|3|
|n_parallel_evaluations|4|8|
|pop_size|2048|4096|
|optimizer|sgd|adamw|
|rank|1|/|
|rank_transform|false|false|
|sigma|0.05|0.05|
|sigma_decay|1|0.995|



Table 24: jumanji/Game2048-v1 

|Hyperparameter|eggroll|open_es|
|---|---|---|
|activation|pqn|pqn|
|deterministic_policy|false|true|
|learning_rate|0.1|0.01|
|lr_decay|1|0.999|
|layer_size|256|256|
|n_layers|3|3|
|n_parallel_evaluations|4|4|
|pop_size|1024|1024|
|optimizer|adamw|adamw|
|rank|1|/|
|rank_transform|false|true|
|sigma|0.5|0.05|
|sigma_decay|0.9995|0.9995|



Table 26: jumanji/Snake-v1 

|Hyperparameter|eggroll|open_es|
|---|---|---|
|activation|pqn|pqn|
|deterministic_policy|false|false|
|learning_rate|0.001|0.001|
|lr_decay|0.9995|1|
|layer_size|256|256|
|n_layers|3|3|
|n_parallel_evaluations|8|1|
|pop_size|4096|2048|
|optimizer|adam|sgd|
|rank|1|/|
|rank_transform|true|false|
|sigma|0.05|0.2|
|sigma_decay|0.9995|1|



74 

Table 27: kinetix/l/hard_pinball 

|Hyperparameter|eggroll|open_es|
|---|---|---|
|activation|pqn|pqn|
|deterministic_policy|true|true|
|learning_rate|0.01|0.01|
|lr_decay|0.995|1|
|layer_size|256|256|
|n_layers|3|3|
|n_parallel_evaluations|8|1|
|pop_size|2048|512|
|optimizer|sgd|sgd|
|rank|4|/|
|rank_transform|true|true|
|sigma|0.05|0.5|
|sigma_decay|0.999|0.9995|



Table 29: kinetix/s/h1_thrust_over_ball 

|Hyperparameter|eggroll|open_es|
|---|---|---|
|activation|pqn|pqn|
|deterministic_policy|false|false|
|learning_rate|0.1|0.01|
|lr_decay|0.995|0.995|
|layer_size|256|256|
|n_layers|3|3|
|n_parallel_evaluations|1|1|
|pop_size|512|2048|
|optimizer|adamw|sgd|
|rank|1|/|
|rank_transform|true|true|
|sigma|0.5|0.05|
|sigma_decay|0.9995|1|



Table 31: navix/Navix-Dynamic-Obstacles-6x6-Randomv0 

|Hyperparameter|eggroll|open_es|
|---|---|---|
|activation|pqn|pqn|
|deterministic_policy|false|false|
|learning_rate|0.01|0.01|
|lr_decay|0.999|1|
|layer_size|256|256|
|n_layers|3|3|
|n_parallel_evaluations|4|1|
|pop_size|512|4096|
|optimizer|adam|adam|
|rank|2|/|
|rank_transform|false|false|
|sigma|0.05|0.2|
|sigma_decay|1|0.995|



Table 28: kinetix/m/h17_thrustcontrol_left 

|Hyperparameter|eggroll|open_es|
|---|---|---|
|activation|pqn|pqn|
|deterministic_policy|false|false|
|learning_rate|0.1|0.001|
|lr_decay|0.9995|1|
|layer_size|256|256|
|n_layers|3|3|
|n_parallel_evaluations|4|1|
|pop_size|512|1024|
|optimizer|sgd|adam|
|rank|4|/|
|rank_transform|true|true|
|sigma|0.5|0.5|
|sigma_decay|1|0.999|



Table 30: navix/Navix-DoorKey-8x8-v0 

|Hyperparameter|eggroll|open_es|
|---|---|---|
|activation|pqn|pqn|
|deterministic_policy|false|false|
|learning_rate|0.01|0.01|
|lr_decay|0.9995|1|
|layer_size|256|256|
|n_layers|3|3|
|n_parallel_evaluations|1|8|
|pop_size|1024|2048|
|optimizer|adamw|adam|
|rank|1|/|
|rank_transform|false|true|
|sigma|0.05|0.05|
|sigma_decay|1|1|



Table 32: navix/Navix-FourRooms-v0 

|Hyperparameter|eggroll|open_es|
|---|---|---|
|activation|pqn|pqn|
|deterministic_policy|false|false|
|learning_rate|0.01|0.001|
|lr_decay|0.999|0.9995|
|layer_size|256|256|
|n_layers|3|3|
|n_parallel_evaluations|4|4|
|pop_size|2048|2048|
|optimizer|sgd|adam|
|rank|4|/|
|rank_transform|true|false|
|sigma|0.05|0.05|
|sigma_decay|0.9995|0.9995|



75 

Table 33: PPO Hyperparameters (Set 1) 

|Hyperparameter|CartPole|Pendulum|Ant|Humanoid|IDP|CraftaxClassic|CraftaxSymbolic|Game2048|
|---|---|---|---|---|---|---|---|---|
|activation|pqn|pqn|pqn|pqn|pqn|pqn|pqn|pqn|
|clip_eps|0.2|0.1|0.2|0.3|0.1|0.2|0.2|0.3|
|ent_coef|0.0001|0.001|0|0.0001|0.0001|0.0001|0|0.001|
|gae_lambda|0.9|0.95|0.95|0.9|0.98|0.98|0.9|0.9|
|gamma|0.995|0.999|0.995|0.95|0.99|0.95|0.95|0.99|
|learning_rate|0.0003|0.0003|0.0003|0.0001|0.001|0.001|0.0003|0.0003|
|max_grad_norm|0.5|1|0.5|2|2|2|2|2|
|layer_size|256|256|256|256|256|256|256|256|
|n_layers|3|3|3|3|3|3|3|3|
|normalize_obs|true|true|true|true|true|true|true|true|
|normalize_rew|false|false|false|false|false|false|false|false|
|num_envs|256|256|64|256|64|128|256|64|
|num_epochs|4|16|8|4|4|4|4|8|
|num_minibatches|32|16|32|64|64|32|32|16|
|num_steps|128|256|128|64|128|128|64|64|
|rew_norm_discount|0.99|0.99|0.99|0.99|0.99|0.99|0.99|0.99|
|skip_initial_eval|false|false|false|false|false|false|false|false|
|vf_coef|0.5|1|1|0.75|1|0.5|0.75|0.75|



Table 34: PPO Hyperparameters (Set 2) 

|Hyperparameter|Knapsack|Snake|HardPinball|ThrustLeft|ThrustBall|DoorKey|DynamicObs|FourRooms|
|---|---|---|---|---|---|---|---|---|
|activation|pqn|pqn|pqn|pqn|pqn|pqn|pqn|pqn|
|clip_eps|0.1|0.3|0.1|0.2|0.2|0.1|0.1|0.1|
|ent_coef|0.0001|0.001|0.0001|0.0001|0.0001|0.0001|0.001|0.001|
|gae_lambda|0.9|0.95|0.9|0.9|0.95|0.98|0.98|0.9|
|gamma|0.99|0.999|0.99|0.995|0.999|0.95|0.999|0.99|
|learning_rate|0.0001|0.0001|0.0001|0.0001|0.0001|0.0003|0.001|0.001|
|max_grad_norm|0.5|0.5|1|2|0.5|0.5|1|1|
|layer_size|256|256|256|256|256|256|256|256|
|n_layers|3|3|3|3|3|3|3|3|
|normalize_obs|true|true|true|true|true|true|true|true|
|normalize_rew|false|false|false|false|false|false|false|false|
|num_envs|256|128|256|256|64|64|128|256|
|num_epochs|4|4|16|16|16|16|4|8|
|num_minibatches|64|16|16|32|16|64|16|32|
|num_steps|128|128|64|128|64|256|128|256|
|rew_norm_discount|0.99|0.99|0.99|0.99|0.99|0.99|0.99|0.99|
|skip_initial_eval|false|false|false|false|false|false|false|false|
|vf_coef|0.75|0.75|0.5|0.5|0.5|0.75|0.5|0.75|



76 

