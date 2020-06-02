Dear Editor,

First of all, we would like to thank the reviewers for their work and constructive feedback. Below, we address all of their questions and observations. 

## R1

> If you run your model ... How is that done?

At the time of the experimentation, we ran into two main issues:
1. Limited resources on Azure, including the access to an Infiniband-enabled system.
2. Most of the MPI-level optimizations that are today in Devito still needed tuning and consolidation.

We subsequently managed to work around point one above. Initial scaling experiments have been published in [@witte2019ecl;@luporini2019adp], although this work still doesn't exploit the latest advancements in Devito v4.2 (e.g., computation/communication overlap). Our manuscript centers on the development of HPC production-grade wave propagators from a high-level language, and while performance is one of the cornerstones of our work, the application was the focus. We have preliminary evidence that the latest Devito v4.2 has remarkable parallel efficiency, but we felt that publishing these results today would be premature.

About validation of simulation quality -- we did not include it in this paper as we think it is outside of the scope of the presented results, but the simulation quality is verified in two ways. As described in [@devito-api], the numerical accuracy is compared to the analytical solution for the acoustic wave equation as well as the convergence rate of the finite-difference discretization. The verification of TTI is not as simple as we do not have an analytical solution; we will be using the method of manufactured solution (MMS), but this is still work in progress. However, all of the key parts, such as finite-difference accuracy and comparison with isotropic results for zero-ed anisotropic parameters are implemented and part of our continuous integration system.

## R2

> However, it is not very suitable for SC

As partly explained in our reply to R1, we conceived this work as an application-oriented paper based on high-level abstractions and compilation technology. We are in the middle of an epochal change for computational science, with legacy codes being superseded, gradually, by DSLs; our work is just yet another small step in this direction. However, unlike many other works centered on DSLs, ours builds on -- and contributes to -- what is nowadays an established framework for seismic imaging. We believe that this should be of strong interest for the SC community, especially since -- we believe -- the methodology and philosophy underpinning Devito is reusable in computational science.

> Parallelization is pretty straightforward, uniform decomposition of relatively straightforward operation.

We remark that our work does _not_ describe some ad-hoc MPI-based domain decomposition; what Devito can do is _generation_ of MPI parallel code from a mathematical language for solving partial differential equations. This requires considerable software engineering (especially for an open source project such as Devito that was born in academia): we're talking about thousands of lines of carefully engineered code, plus tests and examples, to enable users to just type `mpirun -n X python ...` and get their code to run on a distributed-memory system with virtually no changes to their code. Secondly, as we explained in the paper [section ], it's not just "classic domain decomposition": Devito orchestrates sparse functions, implements distributed numpy arrays, and performs several compiler-level optimazions for effective halo exchange. Based on our experience, this is a fairly sophisticated system, quite far from being "straightforward".

> The computational kernels have really high arithmetic intensity that is great for the domain, but not as challenging for optimization

High arithmetic intensity does not imply "non challenging optimization". In fact, it's the exact opposite, and there are several examples in computational science that we could use as evidence, even outside of seismic. Take high-order finite (spectral) elements for instance -- you may get an extremely good arithmetic intensity, but without tensor-product basis functions and suitable optimizations such as sum-factorization you'll likely end up with a notable, yet totally misleasing, GFlops/s performance, as a huge fraction of flops will stem from redundant computation (e.g., common sub-expressions, lack of code hoisting, distributivity). The same story happens with our TTI. As we explain in the paper [section III] without capturing the cross-iteration common sub-expressions (due to high order derivatives), we would have obtained remarkable GFlops/s performance, but terrible GPoints/s. And capturing such redundancies is indeed quite challenging. In fact, this is one of the reasons several companies are today so much interested in Devito: machines (compilers) can be better than humans at detecting redundancies, and definitely way faster than them.

> The parallelism applied so far is straight up MPI which is a bit behind times these days.

In 2020 MPI is still de facto a widely adopted approach for domain decomposition.


## R3

> It was a little confusing but the MPI parallelism appears to essentially be a domain decomposition style parallelism.

As we elaborate in R2, it's not just about decomposing a grid. Devito deals with distribution of sparse functions, python-level distributed arrays, decomposition over sub-domains and much more.

> There was some mention of threads, in particular there seems to be several compilation passes that abstracts out the parallelism which can include SIMD, shared-memory and MPI. Only the MPI is detailed.

Correct. We refer to other papers for all these other aspects.

> One question that arises, in the performance is how does Devito account for FMA (Fused Multiple Adds)?

The operations determining the performance of a code are the source level ones, while FMA is a lower level concept (ie, a scalar FMA would account for two operations). This is not Devito, but rather the model adopted everywhere, otherwise two performance values from two different architectures, one with FMAs the other without, would not even be comparable.

> Scalability performance would also be interesting.

See reply to R1: (i) lack of resources and (ii) Devito not ready yet for scalability tests at the time we ran the experiments.

> It appears much of the performance result are provided in the references so it is unclear as to what the purpose of this paper is for an SC audience. 

We are showing performance results of three different codes -- TTI, elastic, and isotropic acoustic -- thus exploring different physics and discretizations. These results are, respectively, in sections II, III and I. As explained  to R2, ours is an application-oriented paper built off high-level abstractions running on a cloud-based system. To achieve this, we are using several bleeding edge technologies -- an established DSL-based system such as Devito (used in companies and academia), the Azure cloud, Docker (for ease of installation, use and reproducibility), the rapidly-spreading GitHub Actions for continuos integration, and much more. The use and orchestration of these technologies is thouroughly described in the paper, and this should be, in our opinion, of strong interest for the SC audience.

> For actual seismic surveys, the imaging (interpretation) part is only 10% of the survey. [...] while important doesn't address the real "industrial" problems, it helps a little.

The industrial problem does indeed involve muliple parts, such as data acquisition, processing, and, most importantly, velocity model building. We do not claim here to solve all these problems; however, we demonstrate that the combination of advanced technolgies such as high-level DSLs and compilers is key to highly efficient sesimic modeling and imaging at industrial scale. The complexity of model building requires advanced optimization methods. The straightforward access to computationally and easy-to-develop wave-equation solvers enable research and development for seismic inversion. Devito is therfore used for the development of inital background model building [@mojica2019tab] or cycle skipping mitigating methods [@rizzuti2020EAGEtwri; @louboutin2020SEGtwri] and compressive least square migration [@witte2018cls] based on a linear algebra framework for seismic inversion built on top of devito [@witte2018alf]. We have added a clarification in the introduction.

> Dropping the discussion on the 3D elastic wave and [...] might demonstrate the value of Devito better.

We believe that showing the new capabilities of Devito -- the tensor language, distributed omputing via high-level abstractions -- for a different physics and discretization is as important to the SC audience as a strong scaling experiment. In fact, strong scaling may, to some extent, be regarded as more of an academic exercise in seismic imaging:

* even with high-frequency FWI, a typical shot-level computation requires no more than a bunch of nodes (often order of units);
* there is an outer-level of parallelism for inversion, essentially a task farm where the individual workers is internally MPI-parallel.


## R4 

> One issue is that I cannot tell how much of this work is new

We have improved the manuscript to clarify what the contributions are. These are also now reiterated throughout the whole paper (edits in red).

> [...] vector and tensors [...] I am unclear of the amount of effort or innovation involved.

The number of additions to the codebase is notable: https://github.com/devitocodes/devito/pull/873
And for Devito and his users, this was an invaluable addition.

> but I think for SC we would need more information on its scalability and performance.

The single-node performance is discussed in other (cited) articles. In the previous responses (R1, R2, R3), we have elaborated why the lack of scalability experiments.

> Is section III describing new work? It is unclear to me if this is an overview or new work for this paper…

Yes. We have improved the text to make it more explicit and detailed.

> Is the parallelism just via MPI?

The inversion (outer-level of parallelism) requires what is essentially task parallelism; each task is internally MPI-parallel (domain decomposition). In this context task-parallelism means paralleization over sources, which we have accomplished using `batch-shipyard` in a serverless setting (see Section II end). We remark that this allows optimal usage of the typical cloud infrastructure, where one pays based on actual usage of computational resources.

> What are the possibilities for this code in terms of heterogeneous clusters? Will it be able to take advantage of GPUs?
  
GPU support is in development and currently (Devito v4.2) supported for some applications. Its capabilities and related applications will be fully covered in a future dedicated paper.

> Minor issues [...]

Fixed, thanks.
