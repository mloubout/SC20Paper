Dear Editor,

We thank the reviewers for their work and constructive feedback. Below we address their questions and observations. Edits marked in red.

R1

    If you run your model with a variable number of cores, how is the weak and strong scaling?

We published and cited MPI scaling results (P. A. Witte et al., 2019; F. Luporini et al., 2019). While Devito MPI scaling has improved significantly since those publications, it is only a relatively minor consideration within the bigger picture. Applications such as FWI generally only use MPI for computing a shot (single observation). That unit of work sits within an embarrassingly parallel loop processing thousands of shots as part of a gradient-descent algorithm. While computing a single shot the focus on Devito might only require TFLOPS – iteration of the outer loop would typically be PFLOPS. A description of nested-parallelism in applications such as FWI is outside of the scope of this paper, but we have included several references to FWI papers that describe the underlying algorithm.

    validation of simulation quality?

Details on solver verification have been added to the paper. In (M. Louboutin et al., 2019), we analyze the numerical accuracy against analytical solutions and the convergence rate of the finite-differences. We also test through continuous integration the numerical accuracy of all implemented models. For inverse problems, we use standard dot and gradient tests.

R2

    […] not very suitable for SC

 High productivity for HPC and performance-portability continue to be hot topics at SC. Many recent Gordon Bell finalists have used to some degree DSL’s. Devito is one of the few DSL’s to have challenged the performance of handwritten HPC codes for real engineering applications at scale. While Devito is fully open source, we have not been able to publish any head-to-head comparisons with commercial codes in this domain owing to commercial sensitivity. However, one can conclude from the fact that Devito is being adopted by companies commercially that comparisons were favorable. The research is highly relevant to SC as it demonstrates the HPC cliche that one can have high-productivity languages or high-performance, but not both, is false.

    Parallelization is pretty straightforward […]

One can make the same comment about any finite difference, stencil code, linear algebra… The novelty our work is that it involves developing layers of abstraction to allow finite difference based solvers to be implemented in symbolic mathematical form and automatically generate optimized C complete with nested MPI, OpenMP and SIMD parallelism. This is not “straightforward” - the only comparable software technologies would be FEnICS/Firedrake. We also point out in the paper [Sections I/II-B] that on top of standard domain decomposition, Devito decomposes irregular and unstructured sparse functions, implements distributed NumPy arrays, and applies several optimizations for efficient halo exchange.

    high arithmetic intensity […]

High arithmetic intensity does not imply “not as challenging for optimization”. Take high-order finite elements for instance – extremely high arithmetic intensity can be easily obtained, but without tensor-product basis functions and optimizations such as sum-factorization results (GFlops/s) are inflated since a huge fraction of flops are redundant (e.g., common sub-expressions). TTI, a sophisticated propagation model used in industry, is similar. As explained in Section III, without capturing cross-iteration common sub-expressions (high-order cross derivatives) among thousands of algebraic terms, we would obtain very high GFlops/s, but terrible GPoints/s (and therefore poor time to solution).

    […] MPI which is a bit behind

The authors are curious to know if the reviewer also believes all MPI focused papers are unsuitable for SC. Devito uses a range of parallel programming models, usually nested, depending on the context such as SIMD registers, OpenMP, OpenMP 5 offloading, OpenACC, and task-based parallelism (Dask/julia). For domain-decomposition parallelism on distributed memory computers, MPI is still the most widely accepted solution.

R3

    little confusing but […] style parallelism.

The complexity arises from the several layers of parallelisms. See R2 response.

    Only MPI is detailed.

Correct. We explicitly refer to other articles for other aspects.

    […] FMA?

Operations determining the performance of software are the source-level operations. FMA is a lower-level concept (scalar FMA = two operations). This model is widely adopted.

    Scalability performance

See R1 response.

    much of the performance […]

We show performance results of three different codes – TTI, elastic, and isotropic acoustic – thus exploring different physics and discretizations. As explained in R2, ours is an application-oriented paper built off high-level abstractions running on a cloud-based system. To achieve this, we us state-of-the-art edge technologies – an established DSL-based system, Azure cloud, Docker, GitHub-Actions for CI, etc. The use and orchestration of these technologies is thoroughly described in the paper and is of keen interest to SC (strong industry interest for production scale).

    For actual seismic […] it helps a little.

The industrial problem does indeed involve multiple parts, including data acquisition, processing, and, most importantly, velocity model building. We do not claim to solve all these problems. However, we demonstrate that the combination of advanced technologies such as DSLs and compilers is key to highly-efficient seismic modelling and imaging at industrial scale. We have added references and clarification in the introduction.

    Dropping […] 3D elastic […]

The new capabilities of Devito – tensor language, MPI from high-level abstractions – for multiple physics and discretizations are as essential as a strong scaling experiment. See R1 response.
R4

    […] how much is new

We have improved the manuscript to clarify new contributions.

    […] vector and tensors […]

Short answer is significant: https://github.com/devitocodes/devito/pull/873

    for SC […] scalability and performance.

The single-node performance is discussed in other cited articles. See R1 re. lack of scalability experiments.

    section III describing new work?

Yes. Text improved for clarity.

    parallelism just via MPI?

The inversion (outer-level of parallelism) requires what is essentially task-parallelism; each task is internally MPI-parallel. In this context, task-parallelism means parallelization over sources, accomplished using batch-shipyard in a serverless setting [Section II].

    What are the possibilities […]

GPU support is in development and currently (Devito-v4.2) covers some applications and will be covered in forthcoming work.

    Minor issues.

Fixed.