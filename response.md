Dear Editor,

First of all, we would like to thank the reviewers for their work and constructive feedback. Below, we address all of their questions and observations. 

## Review of pap429s2 by Reviewer 1
Detailed Comments for Authors
    This is an interesting paper, well-written, logically ordered and shows interesting results. I only have small questions having to do with scaling results and validation.
Comments for Revision
    If you run your model with a variable number of cores, how is the weak and strong scaling? Can you comment on that? Further, you mention that you validate performance against a wave propagator - how about validation of simulation quality? How is that done?
Scored Review Questions ACCEPT (4)

### Reply

At the time of the experimentation, we ran into two main issues:
1. Limited resources on Azure, including the access to an Infiniband-enabled system.
2. Most of the MPI-level optimizations that are today in Devito were still in a tuning and consolidation phase.

We subsequently managed to work around point one above; we have published some scaling experiments in [..., page ..., figure ...], although they still don't exploit the latest advancements in Devito v4.2 (e.g., computation/communication overlap). What we want to remark is that our manuscript centers on the development of HPC production-grade wave propagators from a high-level language, and while performance is one of the cornerstones of our work, it was not the objective of this paper to show how powerful Devito is in an extreme scaling setting. As -- we are sure -- you have understood, this is completely sensible for a venue such as SC. We could have added scaling plots showing what, with the latest Devito v4.2, appears to be a remarkable parallel efficiency, but we feel that this would have been premature, and our philosophy is not to publish any results that has not been adequately validated and reproduced over multiple architectures. This attitude is at the hearth of the success of Devito, and it's not our intention to undermine it for a mere publication.

About validation of simulation quality -- we did not include it in this paper as we think it is outside of the scope of the presented results but the simulation quality is verified in two ways. As described in [@devito-api] The numerical accuracy is compared to the analytical solution for the acoustic wave equation as well as the convergence rate of the finite-difference discretization. The verification of the TTI equation is not as simple as we  do not have an analytical solution but the implementation of the method of manufactured solution is in progress. All parts such as finite-difference accuracy, comparison with isotropicr esult for zeros anisotropic parameters are part of the continuous integration.

## Review of pap429s2 by Reviewer 2
Detailed Comments for Authors
    This paper presents a methodology to use abstractions and compiler technology effectively in the field of exploration seismology. It is a very nicely written paper that puts down all the motivation and technical details quite well. However, it is not very suitable for SC. Parallelization is pretty straightforward, uniform decomposition of relatively straightforward operation. The computational kernels have really high arithmetic intensity that is great for the domain, but not as challenging for optimization, and therefore not very interesting to this community. The parallelism applied so far is straight up MPI which is a bit behind times these days.
Comments for Revision
    I don't think any revision will change the nature of the work, which, as I mentioned in comments section is nicely done but not best suited for SC.
Scored Review Questions REJECT (1)

### Reply

With all due respect, we strongly disagree with all the critiques moved to the manuscript. Let's go through them one at a time.

* However, it is not very suitable for SC

As partly explained in our reply to R1, this was conceived as an application oriented paper exploiting high-level abstractions and compilation technology. Computational science is evolving rapidly, with production-level DSL-based systems replacing unmaintainable legacy codes, and our work is just yet another small brick in this direction. Unlike the majority of DSL-based tools that often get published, our work builds on and contributes to an established framework for seismic imaging, so we believe that this should be of strong interest for the SC community.

*  Parallelization is pretty straightforward, uniform decomposition of relatively straightforward operation.

First, we observe that this is not just manual implementation of some MPI-based domain decomposition for some pre-existing code. This is rather _generation_ of MPI parallel code from a purely mathematical description of solvers for partial differential equations. The software engineering required is considerable, especially for an open source project such as Devito that was born in academia. We're talking about thousands of lines of carefully engineered code, plus tests and examples, which allow users to just run `mpirun -n X python ...` to run over MPI with virtually no changes to code. Also, as we explained in the paper, it's not grid decomposition only -- sparse functions, distributed numpy arrays, compiler-level optimazion for effective halo exchange -- these are just a few examples of features that are extremely far from being regarded as "pretty straightforward".

* The computational kernels have really high arithmetic intensity that is great for the domain, but not as challenging for optimization

This is simply an error. A high arithmetic intensity does not imply simple optimization. In fact, it's entirely the opposite, and this returns over and over again in computational science. For example, with high-order finite (spectral) elements -- you may get an extremely good OI, but if you're not using tensor-product basis functions and suitable techniques such as sum-factorization, yes, you'll perform loads of flops and obtain a beautiful GFlops/s performance, but unfortunately a huge fraction of those flops will be totally useless, as redundant. The same story happens with our TTI. As we explain in the paper, without having captured all those cross-iteration common sub-expressions,  we would have obtained remarkable GFlops/s performance, but terrible GPoints/s. And capturing those redundancies is extremely difficult -- in fact, this is one of the reasons several companies are today evaluating Devito: machines (compilers) can be better than humans at detecting redundancies, and definitely way faster than them. In the paper we spend a few paragraphs describing how this is achieved. We think this has potentially great interest for the SC community.

* The parallelism applied so far is straight up MPI which is a bit behind times these days.

As far as we know, in 2020 MPI is still the most widely used approach for domain decomposition.


## Review of pap429s2 by Reviewer 3
Detailed Comments for Authors
    The authors in the paper describe their Devito, open-source Python project targeted in the area of seismic exploration. The idea of this particular piece of work is the ease of use through Devito to address, in a Domain Specific Language, creating code for seismic exploration problems which due to being computational intensive often require the use of distributed parallel systems. This is built upon MPI but appears to be hidden from the end user. It was a little confusing but the MPI parallelism appears to essentially be a domain decomposition style parallelism. There was some mention of threads, in particular there seems to be several compilation passes that abstracts out the parallelism which can include SIMD, shared-memory and MPI. Only the MPI is detailed. Through Devito, the authors demonstrate its utility on two problems; (1) Reverse Time Migration (RTM) for a pseudo-acoustic wave problem and (2) an elastic wave problem. With the RTM, they consider titled-transverse isotropic (TTI) wave equation formulation. In RTM, the forward problem is calculated and then the adjoint problem is solve backwards in time to get the imaging. RTM tends to require significant memory and/or fast access to storage of the forward problem during solves for the adjoint problem. Devito is used to set these problems up in VM (Virtual Machines) to be run in a Cloud (Azure) on 32 nodes. If this reviewer understands correctly, Devito is set up to calculate the FLOPs count which is how the authors get the performance. One question that arises, in the performance is how does Devito account for FMA (Fused Multiple Adds)? A comparison was made with an open source hand-code propagator "fdelmode" for the elastic wave equation which seems to be on a 2D problem. The result of this comparison are "essentially identical". A table comparison might be helpful. Scalability performance would also be interesting.
    It appears much of the performance result are provided in the references so it is unclear as to what the purpose of this paper is for an SC audience. For actual seismic surveys, the imaging (interpretation) part is only 10% of the survey. Of course, one does us the RTM in developing the velocity model which adds to complexity in a way. Ease of use coupled with excellent scalable performance while important doesn't address the real "industrial" problems, it helps a little.
Comments for Revision
    Perhaps restricting the discussion of both ease of use of Devito, with a comparison including scalability on one problem, say the 3D TTI problem against a known method and doing this in a cloud setting up to a significant number of nodes would make this a more appropriate paper for the SC audience. This might be too tall an order here. Dropping the discussion on the 3D elastic wave and including a table of comparison with the fdelmode for the 2D elastic wave problem or even just going through scalability comparison on the TTI problem against another code might demonstrate the value of Devito better.
Scored Review Questions WEAK REJECT (2)

### Reply

* One question that arises, in the performance is how does Devito account for FMA (Fused Multiple Adds)?

The operations determining the performance of a code are the source level ones, while FMA is a lower level concept (ie, a scalar FMA would account for two operations). This is not Devito, but rather then model adopted everywhere, otherwise two performance values from two different architectures, one with FMAs the other without, would not even be comparable.

* Scalability performance would also be interesting.

[TODO] Reuse part of replies above

*     It appears much of the performance result are provided in the references so it is unclear as to what the purpose of this paper is for an SC audience. For actual seismic surveys, the imaging (interpretation) part is only 10% of the survey. Of course, one does us the RTM in developing the velocity model which adds to complexity in a way. Ease of use coupled with excellent scalable performance while important doesn't address the real "industrial" problems, it helps a little.

The industrial problem does indeed involve muliple parts, such as data acquisition then processing, and mst importatly velocity model building. We do not claim here to solve all the problems, but demonstrate that the combination of a high-elvel DSL and a highly efficient compiler provides the necessary tool for sesimic modeling and imaging at (industrial) scale. The complexity of model building requires advance optimization methods, and the easy access to computationally and easy to develop wave-equation solvers enable research and development for seismic inversion. Devito is therfore used for the development of inital background model building [@mojica2019tab] or cycle skipping mitigating methods [@rizzuti2020EAGEtwri; @louboutin2020SEGtwri] and compressive least square migration [@witte2018cls] based on a linear algebra framework for seismic inversion built on top of devito [@witte2018alf]
We added a clarification in the introduction.


## Review of pap429s2 by Reviewer 4 
Detailed Comments for Authors
    This manuscript describes optimizations for Devito, a symbolic DSL designed for geophysics, to extend its use for industrial scale modeling of seismic inversion problems.
    Strengths:
        added support for vector and tensor objects for staggered-grid finite-differences
        demonstrated solving large-scale inverse problems (requiring large memory and computation)
        demonstrated its use on Cloud-based HPC
        paper is well-written
    Weaknesses:
        One issue is that I cannot tell how much of this work is new
        contributions. The paper reads a lot like an overview of previous work in many parts. The parts that I am sure are new work (e.g., adding support for vector and tensors), I am unclear of the amount of effort or innovation involved.
        I like this paper and the work (the ease-of-use of Devito is impressive), but I think for SC we would need more information on its scalability and performance.
Comments for Revision
        Please address the weaknesses above
        Is section III describing new work? It is unclear to me if this is an overview or new work for this paper…
        Is the parallelism just via MPI? What are the possibilities for this code in terms of heterogeneous clusters? Will it be able to take advantage of GPUs?
    Minor issues:
    (1) section 3, 1st paragraph: "must to be" => "must be"
    (2) section 1, 3rd paragraph: make an itemized list?
Scored Review Questions WEAK REJECT (2)

### Reply

* One issue is that I cannot tell how much of this work is new

...

*   but I think for SC we would need more information on its scalability and performance.

See previous responses

*    Is section III describing new work? It is unclear to me if this is an overview or new work for this paper…

Yes. We have refined the txt to make it more explicit and detailed.

*    Is the parallelism just via MPI?

The domain decomposition is via MPI, however, the overall task parallelism is not. Seismic inversion is a fullys eparable problem on terms of surce experiment, and the paralleization over these sources is done using `batch-shipyard` in a serverless settting. THis allow optimal usage of Cloud ressource rather than firing up a cluster aht will be IDLE most of time but still payed for bytt he hour in the cloud.

*   What are the possibilities for this code in terms of heterogeneous clusters? Will it be able to take advantage of GPUs?
  
GPU support is in development and currently supported for some applications. Its capabilities and related applications will be fully covered in a future dedicated paper.