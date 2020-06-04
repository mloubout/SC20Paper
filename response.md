Dear Editor,

First of all, we would like to thank the reviewers for their work and constructive feedback. Below, we address all of their questions and observations. 

## R1

> you run your model [...] How is that done?

During experimentation, we had two issues:
1. Limited resources on Azure, including the access to an Infiniband-enabled system.
2. Most of the MPI-level optimizations that are today in Devito still needed tuning and consolidation.

We later managed to perform initial scaling experiments [@witte2019ecl;@luporini2019adp], although still without the latest advancements in Devito (e.g., computation/communication overlap). Today, we have preliminary evidence that Devito 4.2 has remarkable parallel efficiency, but we feel publication would be premature.

Concerning the accuracy -- it was out of the scope of this work, but the simulation quality is verified. In [@devito-api], we have analyzed both the numerical accuracy against the analytical solution and the convergence rate of the finite-difference discretization. In Devito, we systematically test, through continuous integration, the numerical accuracy of all of the implemented models.

## R2

> [...] not very suitable for SC

This work was conceived as an application-oriented paper based on high-level abstractions and compilation technology. However, unlike many other works centered on DSLs, ours builds on -- and contributes to -- what is nowadays an established framework for seismic imaging, used in industry and academia. Our work shows the benefits (productivity, HPC, ...) of raising the level of abstraction, so it should be relevant for SC.

> Parallelization is pretty straightforward, [...] straightforward operation.

Our work does _not_ describe some ad-hoc MPI-based domain decomposition, but rather the _generation_ of MPI-parallel code from a mathematical language for solving PDEs. This requires considerable software engineering -- thousands of lines of carefully engineered code, which allow users to run on a distributed-memory system with virtually no changes to their code. Secondly, as explained in the paper [sections I and II-B], it's not just "classic domain decomposition": Devito also decomposes sparse functions, implements distributed numpy arrays, and applies several optimazions for effective halo exchange. This results in a fairly sophisticated system.

> really high arithmetic intensity [...] not as challenging for optimization

High arithmetic intensity does not imply "non-challenging optimization". In fact, it's the opposite, even outside of seismic. Take high-order finite (spectral) elements for instance -- you may get an extremely high arithmetic intensity, but without tensor-product basis functions and optimizations such as sum-factorization, ultimately you'll get a notable yet misleasing GFlops/s performance, since a huge fraction of flops are redundant (e.g., common sub-expressions, hoisting, contractions). The same story happens with our TTI. As we explain in the paper [section III], without capturing the cross-iteration common sub-expressions due to high-order derivatives, we would have obtained remarkable GFlops/s performance, but terrible GPoints/s. Optimizing away such redundancies is cumbersome, and this is one of the reasons there is today so much interest in Devito by companies.

> [...] MPI which is a bit behind times these days.

MPI is still a widely adopted approach for domain decomposition.


## R3

> It was a little confusing but [...] style parallelism.

As we elaborate in R2, it's not just "grid decomposition". Devito deals with distribution of sparse functions, distributed numpy arrays, decomposition over sub-domains and much more.

> some mention of threads, [...]. Only the MPI is detailed.

Correct. We refer to other papers for all these other aspects.

> [...] FMA (Fused Multiple Adds)?

The operations determining the performance of a code are the source-level ones. FMA is a lower-level concept (ie, a scalar FMA would account for two operations). This is not Devito, but rather the model adopted everywhere.

> Scalability performance

See reply to R1: (i) lack of resources and (ii) Devito not ready yet for scalability tests when we performed the experimentation.

> It appears much of the performance [...]

We are showing performance results of three different codes -- TTI, elastic, and isotropic acoustic -- thus exploring different physics and discretizations. Further, as explained to R2, ours is an application-oriented paper built off high-level abstractions running on a cloud-based system. To achieve this, we are using bleeding edge technologies -- an established DSL-based system such as Devito, Azure cloud, Docker, the rapidly-spreading GitHub-Actions for CI, etc. The use and orchestration of these technologies is thouroughly described in the paper, and is, in our opinion, of strong interest for SC.

> For actual seismic [...] it helps a little.

The industrial problem does indeed involve muliple parts, such as data acquisition, processing, and, most importantly, velocity model building. We do not claim here to solve all these problems; however, we demonstrate that the combination of advanced technolgies such as DSLs and compilers is key to highly-efficient sesimic modeling and imaging at industrial scale. We have added the references and clarification in the introduction.

> Dropping [...] 3D elastic [...] might demonstrate the value of Devito better.

We believe that showing the new capabilities of Devito -- the tensor language, distributed computing from high-level abstractions -- for a different physics and discretization is as important to the SC audience as a strong scaling experiment. In fact, strong scaling may, to some extent, be regarded as more of an academic exercise in seismic imaging:

* even with high-frequency FWI, a typical shot-level computation requires no more than a bunch of nodes (often order of units);
* there is an outer-level of parallelism for inversion, essentially a task farm where the individual workers is internally MPI-parallel.


## R4 

> One issue is that I cannot tell how much of this work is new

We have improved the manuscript to clarify what the contributions are. These are also now reiterated throughout the whole paper (edits in red).

> [...] vector and tensors [...] I am unclear of the amount of effort or innovation involved.

The number of additions to the codebase is significant: https://github.com/devitocodes/devito/pull/873

> for SC we would need [...] scalability and performance.

The single-node performance is discussed in other (cited) articles. In the previous responses (R1, R2, R3), we have elaborated why the lack of scalability experiments.

> Is section III describing new work?

Yes. We have improved the text to make this clear.

> Is the parallelism just via MPI?

The inversion (outer-level of parallelism) requires what is essentially task parallelism; each task is internally MPI-parallel (domain decomposition). In this context task-parallelism means parallelization over sources, which we have accomplished using `batch-shipyard` in a serverless setting (see Section II). This allowed optimal usage of the cloud, where one pays based on actual usage of computational resources.

> What are the possibilities [...]
  
GPU support is in development and currently (Devito v4.2) supported for some applications. Its capabilities and related applications will be fully covered in a future paper.

> Minor issues [...]

Fixed, thanks.
