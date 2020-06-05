---
title: response to reviews
bibliography:
	- sc20_paper.bib
---
Dear Editor,

We thank the reviewers for their work and constructive feedback. Below we address their questions and observations. Edits marked in red.

## R1
> If you run your model with a variable number of cores, how is the weak and strong scaling?

We published and cited MPI scaling results in [@witte2019ecl;@luporini2019adp]. While Devito MPI scaling has improved significantly since those publications, it is only a relatively minor consideration within the bigger picture. Applications such as FWI generally only use MPI for computing a shot (single observation). That unit of work sits within an embarrassingly parallel loop processing thousands of shots as part of a gradient-descent algorithm. While computing a single shot the focus on Devito might only require TFLOPS -- iteration of the outer loop would typically be PFLOPS. A description of nested-parallelism in applications such as FWI is outside of the scope of this paper, but we have included several references to FWI papers that describe the underlying algorithm.

> Further, you mention that you validate performance against a wave propagator - how about validation of simulation quality?

Details on solver verification have been added to the paper. In [@devito-api], we analyze the numerical accuracy against analytical solutions and the convergence rate of the finite-differences. We also test through continuous integration the numerical accuracy of all implemented models. For inverse problems, we use standard dot and gradient tests, "we focused on correct and testable implementations for the adjoint wave-equation and the gradient (action of the adjoint Jacobian)" [@louboutin2018segeow; @louboutin2020THmfi].

## R2

> [...] not very suitable for SC

There are few words in this review so we can only guess in what sense the reviewer feels the work is not suitable. High productivity for HPC and performance-portability continue to be hot topics at SC. Many Gordon Bell finalists in recent years have used to some degree DSL's. Devito is one of the few DSL's to have challenged the performance of handwritten HPC codes for real engineering applications at scale. While Devito is fully open source, we have not been able to publish any head-to-head comparisons with commercial codes in this domain owing to commercial sensitivity. However, one can conclude from the fact that Devito is being adopted by companies commercially that comparisons were favourable. The research is highly relevant to SC as it demonstrates the HPC cliche that one can have high-productivity languages or high-performance, but not both, is false.

> Parallelization is pretty straightforward [...]

Whether or not parallelization is straightforward is not the issue. One can make the same comment about any finite difference, stencil code, anything involving linear algebra...
The novelty our work is that it involves developing layers of abstraction to allow finite difference based solvers to be implemented in symbolic mathematical form and automatically generate optimized C complete with nested MPI, OpenMP and SIMD parallelism. This is not "straightforward" - the only comparable software technologies that we are aware of are FEnICS and Firedrake.
We also point out in the paper [Sections I and II-B] that on top of standard domain decomposition, Devito decomposes irregular and unstructured sparse functions, implements distributed NumPy arrays, and applies several optimizations for efficient halo exchange.

> high arithmetic intensity [...] challenging for optimization

High arithmetic intensity does not imply "not as challenging for optimization". Take high-order finite elements for instance -- extremely high arithmetic intensity can be easily obtained, but without tensor-product basis functions and optimizations such as sum-factorization results (GFlops/s) are inflated since a huge fraction of flops are redundant (e.g., common sub-expressions). TTI, a relatively sophisticated propagation model used in industry, is similar. As explained in Section III, without capturing cross-iteration common sub-expressions (high-order cross derivatives), we would obtain very high GFlops/s, but terrible GPoints/s (and therefore poor time to solution). Optimizing away such redundancies, one of the compiler transformations performed by Devito,  is challenging to do by hand since there can be thousands of algebraic terms.

> [...] MPI which is a bit behind times these days.

The authors are curious to know if the reviewer also believes all MPI focused papers are unsuitable for SC. Devito uses a range of parallel programming models, usually nested, depending on the context. For example, SIMD registers, OpenMP for multi-core shared memory, OpenMP 5 offloading and OpenACC for accelerators, and frameworks such as Dask for parallelizing task-graphs. For domain-decomposition parallelism on distributed memory computers, MPI is still the most widely accepted solution.

## R3

> little confusing but [...] style parallelism.

The complexity arises from the fact that there are several layers of parallelism and parallel programming patterns. While we here focus on MPI for domain decomposition, a single application will also use threading, accelerators, SIMD vectorization, irregular computation and distributed numpy arrays.

> some mention of threads, [...]. Only MPI is detailed.

Correct. We explicitly refer to other articles for other aspects.

> [...] FMA?

Operations determining the performance of software are the source-level operations. FMA is a lower-level concept (scalar FMA = two operations). This model is widely adopted.

> Scalability performance

See R1 response.

> much of the performance [...]

We show performance results of three different codes -- TTI, elastic, and isotropic acoustic -- thus exploring different physics and discretizations. As explained in R2, ours is an application-oriented paper built off high-level abstractions running on a cloud-based system. To achieve this, we are using state-of-the-art edge technologies -- an established DSL-based system, Azure cloud, Docker, GitHub-Actions for CI, etc. The use and orchestration of these technologies is thoroughly described in the paper and is of keen interest to SC (strong industry interest for production scale).

> For actual seismic [...] it helps a little.

The industrial problem does indeed involve multiple parts, including data acquisition, processing, and, most importantly, velocity model building. We do not claim to solve all these problems. However, we demonstrate that the combination of advanced technologies such as DSLs and compilers is key to highly-efficient seismic modelling and imaging at industrial scale. We have added references and clarification in the introduction.

> Dropping [...] 3D elastic [...] might demonstrate [...].

The new capabilities of Devito -- tensor language, MPI from high-level abstractions -- for multiple physics and discretizations is as essential as a strong scaling experiment. Strong scaling may, to some extent, be regarded as more of an academic exercise in seismic imaging:

* even with high-frequency FWI, a typical shot-level computation requires no more than a bunch of nodes (often order of units);
* there is an outer-level of parallelism for inversion, essentially a task farm where the individual workers are internally MPI-parallel.


## R4 

> [...] how much of this work is new

We have improved the manuscript to clarify new contributions. These are also reiterated throughout the whole paper.

> [...] vector and tensors [...] unclear [...] involved.

Short answer is significant: https://github.com/devitocodes/devito/pull/873

> for SC [...] scalability and performance.

The single-node performance is discussed in other cited articles. See R1 re. lack of scalability experiments.

> section III describing new work?

Yes. Text improved for clarity.

> parallelism just via MPI?

The inversion (outer-level of parallelism) requires what is essentially task parallelism; each task is internally MPI-parallel (domain decomposition). In this context, task-parallelism means parallelization over sources, accomplished using `batch-shipyard` in a serverless setting [Section II].

> What are the possibilities [...]
  
GPU support is in development and currently (Devito-v4.2) covers some applications. GPU capabilities will be covered in forthcoming work.

> Minor issues [...]

Fixed, thanks.
