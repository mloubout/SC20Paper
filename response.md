---
title: respons to reviews
bibliography:
	- sc20_paper.bib
---
Dear Editor,

First of all, we would like to thank the reviewers for their work and constructive feedback. Below, we address their questions and observations. All edits are in red in the manuscript.

## R1

> run your model [...] How is that done?

During experimentation, we had two issues:
1 - Limited Azure resources, including access to an Infiniband-enabled system.
2 - Some MPI-level optimizations needed tuning and consolidation.

We managed to perform initial scaling experiments [@witte2019ecl;@luporini2019adp], although without the latest advancements in Devito (e.g., computation/communication overlap). Today, we have preliminary evidence that Devito-v4.2 has remarkable parallel efficiency, but publication would be premature.

Accuracy: details added to the text (in red). In [@devito-api], we analyze the numerical accuracy against the analytical solution and the convergence rate of the finite-differences. We also test through continuous integration the numerical accuracy of all implemented models.

## R2

> [...] not very suitable for SC

This work was conceived as an application-oriented paper based on high-level abstractions and compilation technology. However, unlike many other works centered on DSLs, ours builds on -- and contributes to -- what is nowadays an established framework for seismic imaging, used in industry and academia. Our work shows the benefits (productivity, HPC, ...) of raising the level of abstraction, so it is relevant for SC.

> Parallelization is pretty straightforward [...]

Our work focuse on the generation of MPI-parallel code from a mathematical language for solving PDEs rather than the manual addition of domain decomposition toanexi  sting code. This requires considerable software engineering to allow users to run on a distributed-memory system with virtually no changes to their code. We also point out in the paper [Sections I and II-B], that on top of standard domain decomposition, Devito decomposes unstructured sparse functions, implements distributed numpy arrays, and applies several optimazions for effective halo exchange.

> high arithmetic intensity [...] challenging for optimization

High arithmetic intensity does not imply "non-challenging optimization". High-order finite (spectral) elements for instance -- where an extremely high arithmetic intensity can be obtained, but without tensor-product basis functions and optimizations such as sum-factorization, notable yet misleasing GFlops/s performance is achieved since a huge fraction of flops are redundant (e.g., common sub-expressions). TTI is similar. As explained in Section III, without capturing cross-iteration common sub-expressions (high-order cross derivatives), we would obtaine remarkable GFlops/s, but terrible GPoints/s. Optimizing away such redundancies is cumbersome, and is one of the reasons companies are so much interested in Devito.

> [...] MPI which is a bit behind times these days.

MPI is still curently the most broadly accepted approach for domain decomposition.


## R3

> little confusing but [...] style parallelism.

See R2 -- it's not just "grid decomposition", Devito deals with distribution of sparse functions, numpy arrays, decomposition over sub-domains, etc.

> some mention of threads, [...]. Only MPI is detailed.

Correct. We explicitly refer to other articles for these other aspects.

> [...] FMA?

The operations determining the performance of a code are the source-level ones. FMA is a lower-level concept (ie, a scalar FMA = two operations). This model is adopted everywhere.

> Scalability performance

See R1 -- (i) lack of resources and (ii) Devito not ready yet for scalability tests when the experimentation was performed.

> much of the performance [...]

We show performance results of three different codes -- TTI, elastic, and isotropic acoustic -- thus exploring different physics and discretizations. Further, as explained in R2, ours is an application-oriented paper built off high-level abstractions running on a cloud-based system. To achieve this, we are using bleeding edge technologies -- an established DSL-based system such as Devito, Azure cloud, Docker, the rapidly-spreading GitHub-Actions for CI, etc. The use and orchestration of these technologies is thouroughly described in the paper, and is of strong interest for SC (strong industry interest for production scale).

> For actual seismic [...] it helps a little.

The industrial problem does indeed involve muliple parts, such as data acquisition, processing, and, most importantly, velocity model building. We do not claim to solve all these problems. However, we demonstrate that the combination of advanced technolgies such as DSLs and compilers is key to highly-efficient sesimic modelling and imaging at industrial scale. We have added references and clarification in the introduction.

> Dropping [...] 3D elastic [...] might demonstrate [...].

The new capabilities of Devito -- tensor language, MPI from high-level abstractions -- for multiple physics and discretizations is as important as a strong scaling experiment. Strong scaling may, to some extent, be regarded as more of an academic exercise in seismic imaging:

* even with high-frequency FWI, a typical shot-level computation requires no more than a bunch of nodes (often order of units);
* there is an outer-level of parallelism for inversion, essentially a task farm where the individual workers are internally MPI-parallel.


## R4 

> [...] how much of this work is new

We have improved the manuscript to clarify what the contributions are. These are also now reiterated throughout the whole paper (edits in red).

> [...] vector and tensors [...] unclear [...] involved.

The number of additions to the codebase is significant: https://github.com/devitocodes/devito/pull/873

> for SC [...] sacalability and performance.

The single-node performance is discussed in other cited articles. See R1 about the lack of scalability experiments.

>  section III describing new work?

Yes. We have improved the text to make this clear.

> parallelism just via MPI?

The inversion (outer-level of parallelism) requires what is essentially task parallelism; each task is internally MPI-parallel (domain decomposition). In this context task-parallelism means parallelization over sources, accomplished using `batch-shipyard` in a serverless setting [Section II]. This allowed optimal usage of the cloud, where one pays based on actual usage of computational resources.

> What are the possibilities [...]
  
GPU support is in development and currently (Devito-v4.2) supported for some applications. Its capabilities and related applications will be fully covered in a future paper.

> Minor issues [...]

Fixed, thanks.
