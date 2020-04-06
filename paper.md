---
title: Devito for large scale elastic modelling and anisotropic inversion.
author: |
    Mathias Louboutin^1^, Fabio Luporini^2^, Philipp Witte^1^ , Rhodri Nelson^2^, George Bisbas^2^, Jan Thorbecke^3^, Felix J. Herrmann^1^ and Gerard Gorman^1^ \
    ^1^School of Computational Science and Engineering, Georgia Institute of Technology \
    ^2^ Imperial College London \
    ^3^ TU-Delft
bibliography:
    - sc20_paper.bib
---


## Abstract

[Devito] is an open-source Python project based on domain-specific languages and compiler technology originally devised for rapid implementation of high-performance applications in exploration seismology, where popular inversion methods such as Full-Waveform Inversion and Reverse-Time Migration are used to create images of the earth's subsurface. Throughout several years of development, and thanks to the feedback of companies and community, both the language and compiler have deeply evolved. Today, sophisticated boundary conditions, tensor contractions, sparse operations (e.g., interpolations), as well as fundamental features such as staggered grids and sub-domains, are supported and may be used altogether in the same code. There is virtually no limit to the complexity of a Devito operator. To support all of this flexibility and to ensure competitive performance, the compiler relies on a generic framework for data dependency analysis to schedule loops and detect computational properties, such as parallelism. In this article we harness all this potential to show that Devito is ready for large-scale production-grade seismic inversion problems. We have used  Devito to generate MPI-parallel wave propagators -- as well as their corresponding adjoints -- for the pseudo-acoustic anisotropic wave-equation in a transverse tilted isotropic media and for the elastic wave-equation and ran them on a synthetic industry scale datasets in an HPC cloud system, reaching the performance of 28TFlop/s.

## Introduction

Devito is a symbolic domain specific language (DSL) originally deigned to accelerate research and development in exploration geophysics and more specifically wave-equation based seismic inverse problems. The high-level interface, previously described in detail in [@devito-api], is built on top of `SymPy` [@sympy] and is inspired by the underlying mathematics and similar high-level DSLs such as FEniCS [@fenics] or Firedrake [@firedrake]. This interface allows user to formulate wave-equations, and more generally time-dependent partial differential equations(PDEs) in a simple and mathematically coherent way. The symbolic API then automatically generates finite-difference stencils from these mathematical expressions. One of the main advantages of [Devito] over other finite-difference DSLs is that generic expressions such as spares operations (i.e. point source injection or localized measurements) are fully supported and expressable in a high-level fashion as well. The second part of [Devito] is its compiler (c.f [@devito-compiler]), which generates highly optimized C code from the symbolic expressions and compiles it at runtime for the hardware at hand.

Earlier work introduced Devito for fairly simple problems, such as the acoustic isotropic wave-equation, and limited its applications to simple two-dimensional cases, while the generated code was benchmarked on large-scale three-dimensional problems for single-source experiments. In the work we present here, we demonstrate the scalability of Devito to real-world problems such as industry-scale anisotropic subsurface imaging and elastic modeling. These proof of concepts highlight two main contributions: First, we show that the symbolic interface and the compiler translates to large-scale adjoint-based inverse problem that require both massive compute, as thousands of PDEs need to be solved, as well as large amounts of memory, since the adjoint state method requires the forward state to be saved in memory. Second of all, we demonstrate with an elastic modelling example that [Devito] now fully supports and automatizes vectorial and second order tensorial staggered-grid finite-difference with the same high-level interface previously presented for a scalar field and a cartesian grid.

This paper is organized as follows: First, we provide a brief overview of [Devito] and its symbolic API and present the distributed memory implementation that allows large-scale modeling and inversion with domain decomposition. We then provide a brief comparison with a state of the art hand-coded wave propagator to validate the performance previously benchmarked with the roofline model ([@patterson, @devito-compiler, @devito-api, @louboutin2016ppf]). Next, we describe the two applications the demonstrate the scalability and staggered capabilities of [Devito]: We conduct a three-dimensional imaging example in the cloud-based on the tilted transverse isotropic wave equation (TTI, [@zhang-tti, @duveneck, @louboutin2018segeow]), as well as an elastic modeling example that highlights the vectorial and tensorial capabilities.


## Devito

We first provide an overview of Devito [@devito-api, @devito-compiler] and describe the capabilities that enable real-world applications as presented in the subsequent sections. Devito is a finite-difference domain-specific language (DSL) built on top of `Sympy` [@sympy] and provides a high-level symbolic interface for the definition of partial differential equations. Devito automatically generates optimized finite-difference stencil associated with the PDE and supports both cartesian and staggered grids. The symbolic Devito stencils are then passed to the Devito compiler, which generates and compiles C-code that is optimized for the architecture at hand using its just-in-time (JIT) compiler. In previous work, we have focused on the DSL and the compiler to highlight the potential application and use cases of Devito, while in this paper, we present a series of extensions and applications to large-scale problem sizes as encountered in exploration geophysics, including elastic modelling using distributed-memory parallelism [@...], and multi-experiment seismic imaging in anisotropic media [@virieux, @thomsen, @zhang2011, @duveneck, @louboutin2018segeow]. We briefly describe the symbolic API and compiler and give a brief overview of the computational performance of the generated code.

### Symbolic API

The core of Devito's symbolic API relies on three basic object classes for representing grids and state variables of partial differential equations. These classes are

- `Grid` objects represent the discretized model.
- `(Time)Function` objects represent spatially (and time-) varying variables defined on a `Grid` object.
- `Sparse(Time)Function` objects represent (time-varying) point objects on the specified grid.

A `Grid` represents a discretized finite n-dimensional space and is created as follows:

```python
from devito import Grid
grid = Grid(shape=(nx, ny, nz), extent=(ext_x, ext_y, ext_z), origin=(o_x, o_y, o_z))
```

where `(nx, ny, nz)` are the number of grid points in each direction, `(ext_x, ext_y, ext_z)` is the physical extent of the domain in physical units (i.e `m`) and `(o_x, o_y, o_z)` is the origin of the domain in the same physical units. The `grid` contains all the information related to the discretization such as the grid spacing, and automatically initializes the `Dimension` that define the domain `x, y, z`. With this grid, the symbolic objects can be created for the discretization of a PDE. First, we can define a spatially varying model parameter `m` and a time-space varying field `u`

```python
from devito import Function, TimeFunction
m = Function(name="m", grid=grid, space_order=so)
u = TimeFunction(name="u", grid=grid, space_order=so, time_order=to)
```

where `so` is the spatial discretization order and `to` the time discretization order that is used for the generation of the finite-difference stencil. Second, we can define point-wise objects such as point sources `src` located at a (limited number) of physical coordinates `s_coords` and receiver (measurement) objects `rec` with sensors located at the physical locations `r_coords`.

```python
from devito import Function, TimeFunction
src = SparseFunction(name="src", grid=grid, npoint=1, coordinates=s_coords)
rec = SparseTimeFunction(name="rec", grid=grid, npoint=1, nt=nt, coordinates=r_coords)
```

From these object, we can define, as an example, the acoustic wave-equation in five lines as:
{>> Add equation <<}

{>> Fabio: I have a feeling we should show some generated code here (do not forget the audience) and reiterate that the bulk of the computation is stencil-like <<}

```python
from devito import solve, Eq, Operator
eq = m * u.dt2 - u.laplace
stencil = [Eq(u.forward, solve(eq, u.forward))]
src_eq = s1.inject(u.forward, expr=s1 * dt**2 / m)
rec_eq = rec.interpolate(u)
wave_solve = Operator(stencil _ src_eq + rec_eq)
```

The compiler then evaluates the finite-difference expressions and generate the C code associated with it threw passes such as common subexpression elimination, factorization, cross-iteration redundancies elimination or time-invariant extraction. These compiler details are described in [@devito-compiler] while the symbolic API is fully presented in [@devito-api]. We know give a brief description of the distributed memory parallelism implemented and available in Devito that has been used for the two applications we present in this paper.

### Overview of distributed-memory parallelism

We here provide a succinct description of distributed-memory parallelism in Devito; the interested reader should refer to [@mpi-notebook] for thorough explanations and practical examples.

Devito implements distributed-memory parallelism on top of MPI. The design is such that users can almost entirely abstract away from it. Given *any* Devito code, just running it as

```python
DEVITO_MPI=1 mpirun -n X python ...
```

will trigger the compiler to generate C with routines for halo exchanges. The routines are scheduled at a suitable depth in the various loop nests thanks to data dependency analysis. The following optimizations are automatically applied:

* redundant halo exchanges are detected and dropped;
* computation/communication overlap, with prodding of the asynchronous progress engine by a designated thread through repeated calls to `MPI_Test`;
* a halo exchange is placed as far as possible from where it is needed to maximize computation/communication overlap;
* data packing and unpacking is threaded.

The domain decomposition occurs in Python upon creation of a `Grid` object. Exploiting the MPI Cartesian topology abstraction, Devito logically splits a grid based on the number of available MPI processes (users are given an "escape hatch" to override Devito's default decomposition strategy). `Function` and `TimeFunction` objects inherit the `Grid` decomposition. For `SparseFunction` objects the approach is different. Since a `SparseFunction` represents a sparse set of points, Devito looks at the physical coordinates of each point and, based on the `Grid` decomposition, schedules the logical ownership to an MPI rank. If a sparse point lies along the boundary of two or more MPI ranks, then it is duplicated to be accessible by all neighboring processes. Eventually, a duplicated point may be redundantly computed by multiple processes, but any redundant increments will be discarded.

When accessing or manipulating data in a Devito code, users have the illusion to be working with classic NumPy arrays, while underneath they actually are distributed. All manner of NumPy indexing schemes (basic, slicing, etc.) are supported. In the implementation, proper global-to-local and local-to-global index conversion routines are used to propagate a read/write access to the impacted subset of ranks. For example, consider the array

```python
A = [[ 1,  2,  3,  4],
     [ 5,  6,  7,  8],
     [ 9, 10, 11, 12],
     [13, 14, 15, 16]])
```

which is distributed across 4 ranks such that `rank 0` contains the elements reading `1, 2, 5, 6`, `rank 1` the elements `3, 4, 7, 8`,  `rank 2` the elements `9, 10, 13, 14` and `rank 3` the elements `11, 12, 15, 16`. The slicing operation `A[::-1, ::-1]` will then return

```python
    [[ 16, 15, 14, 13],
     [ 12, 11, 10,  9],
     [  8,  7,  6,  5],
     [  4,  3,  2,  1]])
```

such that now `rank 0` contains the elements `16, 15, 12, 11` and so forth.

Finally, we remark that while providing abstractions for distributed data manipulation, Devito does not support natively any mechanisms for parallel I/O. {>> Add positive note<<}

## Industry-scale 3D seismic imaging in anisotropic media

On of the main applications of seismic finite-difference modeling in exploration geophysics is reverse-time migration (RTM), a wave equation-based seismic imaging technique. Real-world seismic imaging presents a number of challenges that make applying this method to industry-scale problem sizes difficult. First of all, RTM requires an accurate representation of the physics through sophisticated wave equations such as the tilted-transverse isotropic (TTI) wave equation, for which both forward and adjoint implementations have to be provided. Second of all, wave equations have to be solved for a large number of independent experiments, where each individual PDE solve in itself is expensive in terms of FLOPs and memory usage and domain decomposition of wavefield checkpointing has to be applied. In the following section, we highlight Devito's capabilities to address these challenges, making it possible to use Devito for realistic seismic imaging on an industry scale. First, we address how forward and adjoint TTI equations can be implemented with Devito, and subsequently we carry out a 3D seismic imaging case study on Azure using a synthetic data set.

### Anisotropic wave equations

{{>> Remove PDE and show code for rotation<<}

{>> FAbio: say somewhere it's sigle precision (even in elastic)<<}

(section needs to be shortened)

In our seismic imaging case study, we use an anisotropic representation of the physics called tilted transverse isotropic modeling [@thomsen1986]. This representation for wave motion is one of the most widely used in exploration geophysics since it captures the leading order kinematics and dynamics of acoustic wave motion in highly heterogeneous elastic media where the medium properties vary more rapidly in the direction perpendicular to sedimentary strata [@alkhalifah2000; @baysal1983; @bubetti2012; @bubetti2014; @bubesatti2016; @chu2011; @duveneck; @fletcher; @fowlertti2010; @louboutin2018segeow; @whitmore1983; @witte2016segpve; @xu2014; @zhang2005; @zhang2011; @zhan2013]. The TTI wave equation is an acoustic, low dimensional (4 parameters, 2 wavefields) simplification of the 21 parameter and 12 wavefields tensorial equations of motions [@hooke]. This simplified representation is parametrized by the Thomsen parameters ``\epsilon(x), \delta(x)`` that relate to the global (many wavelength propagation) difference in propagation speed in the vertical and horizontal directions, and the tilt and azimuth angles ``\theta(x), \phi(x)`` that define the rotation of the vertical and horizontal axis around the cartesian directions. However, unlike the scalar isotropic acoustic wave-equation itself, the TTI wave equation is extremely computationally costly to solve and it is also not self-adjoint. The TTI wave-equation reads as follows:

{>> fabio: "self-adjoint." is too much for the SC audience -- can we expand on this with very intuitive words ?<<} 

```math {#TTIfwd}
&m(x) \frac{d^2 p(x,t)}{dt^2} - (1+2\epsilon(x))H_{\bar{x}\bar{y}}p(x,t) - \sqrt{1+2\delta(x)} \ H_{\bar{z}} r(x,t) = q,  \\
&m(x) \frac{d^2 r(x,t)}{dt^2} - \sqrt{1+2\delta(x)} \ H_{\bar{x}\bar{y}} p(x,t) - H_{\bar{z}} r(x,t) = q,
```

where ``p(x,t)`` and ``r(x,t)`` are the two component of the anisotropic wavefield and ``H_{\bar{z}}`` and ``H_{\bar{x}\bar{y}} = G_{\bar{x}\bar{x}} + G_{\bar{y}\bar{y}}`` are the rotated second order differential operators that depend on the tilt, azimuth  (``\theta(x), \phi(x)``) and the conventional (isotropic) cartesian spatial derivatives ``\frac{d}{dx}, \frac{d}{dy}`` and ``\frac{d}{dz}``.

After discretization, the TTI wave-equation can be rewritten in a linear algebra form:

```math {#TTIfwd-la}
 \mathbf{m}  \begin{bmatrix}\frac{\mathrm{d}^2 \mathbf{p}}{\mathrm{d} t^2} \\\frac{\mathrm{d}^2 \mathbf{r}}{\mathrm{d} t^2} \end{bmatrix} &=  \begin{bmatrix} (1 + 2\mathbf{\epsilon})H_{\bar{x}\bar{y}} & \sqrt{1 + 2 \mathbf{\delta}}\ H_{\bar{z}} \\ \sqrt{1 + 2 \mathbf{\delta}} \ H_{\bar{x}\bar{y}} & H_{\bar{z}} \end{bmatrix} \begin{bmatrix} \mathbf{p} \\ \mathbf{r} \end{bmatrix} + \mathbf{P}_s^\top \mathbf{q}
```

where the bold font represents discretized version of the wavefield and physical parameters. With this expression, we can rewrite the solution of the anisotropic wave equation as the solution of a linear system ``\mathbf{u}(\mathbf{m}) = \mathbf{A}(\mathbf{m})^{-1} \mathbf{P}_s^\top`` where ``\mathbf{u}(\mathbf{m})`` is a two component vector ``(\mathbf{p}(\mathbf{m})^\top, \mathbf{r}(\mathbf{m})^\top)^\top``. The matrix ``\mathbf{P}_s^\top`` injects the source in both wavefield components. This matricial formulation is extremely useful for the mathematical derivation of inversion operation such as gradients.

As discussed in @zhang2011 and @duveneck, we choose a finite-difference discretization of the three differential operators ``H_{\bar{z}}, G_{\bar{x}\bar{x}}, G_{\bar{y}\bar{y}}`` that is self-adjoint to ensure numerical stability. For example, we define ``G_{\bar{x}\bar{x}}`` as a function of the discretized tilt ``\mathbf{\theta}`` and azymuth ``\mathbf{\phi}`` as:

```math {#rot}
  G_{\bar{x}\bar{x}} &= D_{\bar{x}}^T D_{\bar{x}} \\
  D_{\bar{x}} &= \cos(\mathbf{\theta})\cos(\mathbf{\phi})\frac{\mathrm{d}}{\mathrm{d}x} + \cos(\mathbf{\theta})\sin(\mathbf{\phi})\frac{\mathrm{d}}{\mathrm{d}y} - \sin(\mathbf{\theta})\frac{\mathrm{d}}{\mathrm{d}z}.
```

{>> Fabio: need to describe BCs here <<}

Because of the very high number of floating-point operations (FLOP) needed per grid point for the weighted rotated Laplacian, this anisotropic wave-equation is extremely challenging to implement. As we show in Figure #ttiflops, and previously analysed in [@louboutin2016ppf], the computational cost with high-order finite-difference is in the order of thousands of FLOPs per grid point without optimizations.

#### Table: {#ttiflops}
|                 | w/o optimizations | w/  optimizations |
|:----------------|:----------------- |:----------------- |
| space_order=4   | 501               | 95                |
| space_order=8   | 539               | 102               |
| space_order=12  | 1613              | 160               |
| space_order=16  | 5489              | 276               |

: Per-grid-point flops of the finite-difference stencil for the TTI wave-equation with different spatial discretization orders.

The version without flop-reducing optimizations is a direct translation of the discretized operators into stencil expressions. The version with optimizations employs transformations such as common sub-expressions elimination, factorization, and cross-iteration redundancy elimination -- the latter being key in removing the redundancies induced by mixed derivatives. Implementing all of these techniques manually is inherently difficult and laborous. Further, to obtain the desired performance improvements it is necessary to orchestrate them with aggressive loop fusion (for data locality), tiling (for data locality and tensor temporaries), and potentially ad-hoc vectorization strategies (if rotating registers are used as in [@cire1]). While an explanation of the optimization strategy employed by Devito is beyond the scope of this paper (see [@devito-compiler] for details), what should be appreciated here is that all this complexity is hidden away from the users.

With such complex physics, mathematics, and engineering, it becomes evident that the implementation of a solver for this wave-equation is exceptionally time-consuming and can lead to thousands of lines of code even for a single type of discretization. The verification of the result is no less complicated, since any small error is effectively untrackable and any change to the finite-difference scheme or to the time-stepper is difficult to achieve without substantial re-coding. Another complication stems from the fact that practitioners of seismic inversion are often geoscientists and not computer scientists/programmers. Unfortunately, this background often either results in poorly written low performant codes or it leads to complications when research codes are handed off to computer scientists who know how to write fast codes but who often miss the necessary geophysical domain knowledge. Neither situation is conducive to addressing the complexities that come with implementing codes based on the latest geophysical insights in geophysics and high-performance computing. With Devito on the other hand, both the forward and adjoint equations can be implemented in a few lines of Python code:

```
# Example of how to implement TTI w/ Devito
...

# Forward
...

# Adjoint
...

```

Simulation of wave motion is only one aspect of solving problems in seismology. During wave-equation based imaging, we also need to compute sensitivities (gradient) with respect to the quantities of interest. This imposes additional constraints on the design and implementations of our simulation codes as outlined in [@virieux]. Among several factors, such as fast setup time etc., we focused on correct and testable implementations for the adjoint wave equation and the gradient (action of the adjoint Jacobian) [@louboutin2018segeow, @louboutin2020THmfi].

### 3D Imaging example on Azure

{>> Remove lift and shift and just ref<<}

One of the main challenges in modern HPC is to modernize legacy codes for the cloud, which are usually hand-tuned or designed for on-premise clusters with a known and fixed architecture and setup. Porting these codes and algorithms to the cloud can be straightforward using a lift-and-shift strategy that essentially boils down to renting a cluster in the cloud. However, this strategy is not cost-efficient. Pricing in the cloud is typically based on a pay-as-you-go model, which charges for requested computational resources, regardless of whether or not instances and cores are actively used or sit idle. This pricing model is disadvantageous for the lift-and-shift strategy and oftentimes incurs higher costs than required by the actual computations, especially for communication-heavy but task-based algorithms that only need partial resources depending of the stage of the computation. On the other hand, serverless software design provides flexible and cost efficient usage of cloud resources including for large scale inverse problem such as seismic inversion. With Devito, we had access to a portable yet computationally efficient framework for wave-equation based seismic exploration that allowed us to quickly develop a new strategy to execute seismic inversion algorithms in the cloud. This new serverless and event-driven approach led to significant early results [@witte2019TPDedas, @witte2019SEGedw] that caught the attention of both seismic inverse problems practitioners and cloud providers. This led to a proof of concept project on an industry size problem in collaboration with Microsoft Azure. The main objectives of this project were:

{>>I don't know whether the last paragraph is needed. Suddenly, we're talking about lift and shift for cloud computing, which seems out of context here.<<}

- Demonstrate the scalability, robustness and cost effectiveness of a serverless implementation of seismic imaging in the cloud. In this case, we imaged a synthetic three dimensional anisotropic subsurface model that mimics a realistic industry size problem with a realistic representation of the physics (TTI). The physical size of the problem is `10kmx10kmx2.8km` discretized on a `12.5m` grid.

- Demonstrate the flexibility and portability of Devito. The seismic image (RTM as defined in Chapter 3) was computed with Devito and highlights the code-generation and high performance capability of Devito on an at-scale real world problem. This result shows that in addition to conventional benchmark metrics such as soft and hard scaling and the roofline model, Devito provides state of the art performance on practical applications as well.

***Computational performance***

We briefly describe the computational setup and the performance achieved for this anisotropic imaging problem. Due to time constraints, and because the resources we were given access to for this Proof of concept with Microsoft Azure were limited, we did not have access to HPC virtual machines (BM) nor Infiniband enabled ones. The nodes we ran this experiment on are `Standard_E64_v3` and `Standard_E64s_v3` that while not HPC VM are memory optimized allowing to save the wavefield in memory for imaging (TTI adjoint state gradient [@virieux, @louboutin2018segeow]).
These VMs are Intel&reg; Broadwell E5-2673 v4 2.3GH that are dual socket, 32 physical cores (and hyperthreading enabled) and 432Gb of memory CPUs. The overall inversion involved computing the image for 1500 source positions, i.e. solving 1500 forward and 1500 adjoint TTI wave-equation. A single image required 600Gb of memory and we used two VM per source with MPI with one rank per socket (4 MPI ranks per source) and imaged 100 sources in parallel due to resources limitations (in theory we could have ran the 1500 in parallel with the necessary quotas). The performance achieved was as follow:

- 140 GFlop/s per VM
- 280 GFlop/s per source
- 28 TFlops/s for all 100 running sources. This would have led to 0.4PFlop/s with the quotas to run all sources at once instead of 100 at a time.
- 110min runtime per source (forward + adjoint + image computation)

***How was the performance measured***

The execution time is computed through Python-level timers prefixed by an MPI barrier. This includes the overhead due to processing the arguments supplied to `Operator` (see snippet ... TODO). Such overhead is however negligible.

The floating-point operations are counted once all of the symbolic flop-reducing transformations have been performed during compilation. Devito uses an in-house estimate of cost, rather than `SymPy`'s estimate, to take care of some low-level intricacies. For example, Devito's estimate ignores the cost of integer arithmetic used for offset indexing into multi-dimensional arrays. To calculate the total number of FLOPs performed, Devito multiplies the floating-point operations calculated at compile time by the size of the iteration space, and it does that at the granularity of individual expressions. Thanks to aggressive code motion, the amount of innermost-loop-invariant sub-expressions in a Devito Operator is typically negligible, so the Devito estimate doesn't basically suffer from this issue, or at least not in a tangible way to the best of our knowledge. The produced GFlops/s has also been checked against that reported by Intel Advisor on several single-node experiments, and the results were extremely close, which gives confidence about the soundness of the Devito estimate.


***Imaging result***

The subsurface velocity model that was used in this study is an artificial anisotropic model that is designed and built combining two broadly known and used open-source SEG/EAGE acoustic velocity models. The anisotropy parameters are derived from smoothed version of the velocity while the tilt angles were derived from a combination of the smooth velocity models and vertical and horizontal edge detection. The final seismic image of the subsurface model is plotted in Figure #OverTTI and highlights the fact that 3D seismic imaging based on a serverless approach and automatic code generation is feasible and provides good results on a realistic model.


#### Figure: {#OverTTI}
![](./Figures/OverTTI1.png){width=50%}
![](./Figures/OverTTI2.png){width=50%}\
![](./Figures/OverTTI3.png){width=50%}
![](./Figures/OverTTI4.png){width=50%}\
: 3D TTI imaging on a custom made model.

@witte2019TPDedas fully describes the serverless implementation of seismic inverse problems, including iterative algorithms for least square minimization problems (LSRTM). The 3D anisotropic imaging results were presented as part of a keynote presentation at the EAGE HPC workshop in October 2019 [@herrmann2019EAGEHPCaii]. This work perfectly illustrates the flexibility and portability of Devito, as we were able to easily port a code only tested and developed on local hardware to the cloud, with only requiring minor adjustments. This portability included the possibility to run MPI-based code for domain decomposition in the cloud, after developing it on a desktop computer. The code for reproducibility can be found at [AzureTTI] that contains, the propagators and gradient computation, Dockerfiles and azure [batch-shipyard] setup for running the RTM.

While this subsurface image is obtained with anisotropic propagators that mimic the real physics, in order to model both the kinematics and the amplitudes correctly, elastic propagator are required. Theses propagators are for example extremely important for global seismology as the Shear waves (component ignored in TTI) are the most hazardous ones. We know show that wit the new vectorial capabilities of Devito, we can model elastic waves while conserving a high-level symbolic interface.


## Elastic modelling

The elastic isotropic wave-equation, parametrized by the Lamé parameters ``\lambda, \mu`` and the density ``\rho`` reads:

```math{#elas1}
&\frac{1}{\rho}\frac{dv}{dt} = \nabla . \tau \\
&\frac{d \tau}{dt} = \lambda tr(\nabla v) \mathbf{I}  + \mu (\nabla v + (\nabla v)^T)
```

where ``v`` is a vector valued function with one component per cartesian direction:

```math{#partvel}
v =  \begin{bmatrix} v_x(t, x, y, z) \\ v_y(t, x, y)) \end{bmatrix}
```

and the stress ``\tau`` is a symmetric second-order tensor-valued function:

```math{#stress}
    \tau = \begin{bmatrix}\tau_{xx}(t, x, y) & \tau_{xy}(t, x, y)\\\tau_{xy}t, x, y) & \tau_{yy}(t, x, y)\end{bmatrix}.
```

The discretization of such a set of coupled PDEs requires five equations in two dimensions (two equations for the particle velocity and three for stress) and nine equations in three dimensions (three particle velocities and six stress equations). However the mathematical definition only requires two coupled vector/tensor-valued equations for any number of dimensions. We extend the previously scalar-only capabilities of Devito to vector and second-order tensors and allow a straightforward and mathematical definition of high-dimensional PDEs such as the elastic wave equation in Eq #elas1\.


### Vectorial and tensorial API

Once again, based on `sympy`, we augmented the symbolic interface to vectorial and tensorial object to allow for a straightforward definition of equations such as the elastic wave-equation, as well as computational fluid dynamics equations. The extended API defines two new types, `VectorFunction` (and `VectorTimeFunction`) for vectorial objects such as the particle velocity, and `TensorFunction` (and `TensorTimeFunction`) for second-order tensor objects (matrices) such as the stress. These new objects are constructed the exact same way as the scalar `Function` objects and automatically implement staggered grid and staggered finite-differences with the possibility of half-node averaging. This new extended API now allows users to define the elastic wave-equation in four lines as follows:

### Elastic modelling

```python
v = VectorTimeFunction(name='v', grid=model.grid, space_order=so, time_order=1)
tau = TensorTimeFunction(name='t', grid=model.grid, space_order=so, time_order=1)

u_v = Eq(v.forward, model.damp * (v + s/rho*div(tau)))
u_t = Eq(tau.forward,  model.damp *  (tau + s * (l * diag(div(v.forward)) + mu * (grad(v.forward) + grad(v.forward).T))))
```

The `sympy` expressions created by these commands can be displayed via the `sympy` pretty printer (`sympy.pprint`) as shown in Figure #PrettyElas\. This representation reflects perfectly the mathematics while still providing computational portability and efficiency through the Devito compiler.

#### Figure: {#PrettyElas}
![](./Figures/vel_symb.png){width=100%}
: Update stencil for the particle velocity. The stencil for updating the stress component is left out for readability, as the equation does not fit onto a single page. However, it can be found in the Devito tutorial on elastic modelling.

Each component of a vectorial or tensorial object is accessible via conventional vector and matrix indices (i.e. `v[0], t[0,1],....`).

### 2D example

We show the elastic particle velocity and stress for a well known 2D synthetic model, the elastic Marmousi-ii[@versteeg927, @marmouelas] model. The wavefields are shown on Figure #ElasWf and its corresponding elastic shot records are displayed in Figure #ElasShot\.

#### Figure: {#ElasWf}
![](./Figures/marmou_snap.png){width=100%}
: Particle velocities and stress at time ``t=3\text{s}`` for a source at 10m depth and `x=5\text{km}` in the marmousi-ii model.

#### Figure: {#ElasShot}
![](./Figures/pressure_marmou.png){width=30%}
![](./Figures/vz_marmou.png){width=30%}
![](./Figures/vx_marmou.png){width=30%}
: Seismic shot record for 5sec of modelling. `a` is the pressure (trace of stress tensor) at the surface (5m depth), `b` is the vertical particle velocity and `c` is the horizontal particle velocity at the ocean bottom (450m depth).

### 3D proof of concept in the cloud

{>> This is the actual SEAM run but can't say it don't have license<<}

Finally, we modelled three dimensional elastic data in the Cloud to demonstrate the scalability of Devito to cluster-size problems in the Cloud. The model we chose mimics the reference model in geophysics known as the SEAM model [@...] that is a three dimensional extreme scale synthetic representation of the subsurface. The physical dimension of the model are `45kmx35kmx15km` then discretized with a grid spacing of `20mx20mx10m` that led to a computational grid of `2250x1750x1500` grid points (5.9 billion grid points). One of the main challenges of elastic modelling is the extreme memory cost due to the number of wavefield. For a three dimensional propagator, a minimum of 21 fields need to be stored:

- Three particle velocities with two time steps (`v.forward` and `v`)
- Six stress with two time steps (`tau.forward` and `tau`)
- Three model parameters `lamda`, `mu` and `rho`

These 21 fields, with the grid we just describe, leads to a minimum of 461Gb of memory for modelling only. For this experiment, we obtained access to small HPC VM on azure called `Standard_H16r` that are 16 cores Intel Xeon E5 2667 v3 with no hyperthreading and used 32 nodes for a single source experiment (we solved a single wave equation). We used a 12th order discretization that leads to 2.8TFlop/time-step to be computed for this model and propagated the elastic wave for 16 seconds (23000 time steps). The modelling finished in 16 hours which converts to 1.1TFlop/s. While these number may seem low, the elastic kernel is extremely memory bound, while the TTI kernel is nearly compute bound (see rooflines in [@louboutin2016ppf, @devito-api, @devito-compiler]) making it more computationally efficient, in particular in combination with MPI. Future work with involve working on the InfiniBand enabled and true HPC VM on azure to achieve Cloud performance on par with state of the art Cluster performance.

## Performance comparison with other codes

We also compared the runtime of Devito with a reference open source hand-coded propagator in collaboration with its author. This propagator, described in [@thorbecke] is a state of the art elastic kernel (Eq #elas1). For a fair comparison, we ensure, in collaboration with the author of [fdelmodc] that the physical and computational settings were identical. The setting were the following:

- 2001 by 1001 physical grid points.
- 200 grid points of dampening layer (absorbing layer [@cerjan]) on all four sides (total of 2401x1401 computational grid points).
- 10001 time steps.
- Single point source, 2001 receivers.
- Same compiler (gcc/icc) to compile [fdelmodc] and run Devito.
- Intel(R) Xeon(R) CPU E3-1270 v6 @ 3.8GHz.
- Single socket, four physical cores, four physical threads, thread pinning to cores and hyperthreading off.

The runtime obtained for this problem for both propagators were identical with less than one percent of difference. This similar runtime were obtained both with the Intel compiler and the GNU compiler and we ran the experiment with a fourth order and a sixth order discretization. The kernels were executed five time each to ensure consistency between the results and we consistently obtained similar runtimes. This comparison illustrates the performance achieved with Devito is at least on par with hand-coded propagators.

## Conclusions

Can solve large scale and non-trivial physics problem both on conventional clusters and in the Cloud
Good performance
High level interface that allows simple and mathematically based expression of complicated physics.


[fdelmodc]:https://github.com/JanThorbecke/OpenSource.git
[AzureTTI]:https://github.com/slimgroup/Azure2019
[batch-shipyard]:https://batch-shipyard.readthedocs.io
[JUDI]:https://github.com/slimgroup/JUDI.jl
[Devito]:https://github.com/devitocodes/devito

## References
