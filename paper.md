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

[Devito] is an open-source Python project based on domain-specific {==languages==} and compiler technology. Driven by the requirements of rapid HPC applications development in exploration seismology, the language and compiler have evolved significantly since inception. Sophisticated boundary conditions, tensor contractions, sparse operations and features such as staggered grids and sub-domains are all supported; operators of essentially arbitrary complexity can be generated. To accommodate this flexibility whilst ensuring performance, data dependency analysis is utilized to schedule loops and detect computational-properties such as parallelism. In this article, the generation and simulation of MPI-parallel propagators (along with their adjoints) for the pseudo-acoustic wave-equation in tilted transverse isotropic media and the elastic wave-equation are presented. Simulations are carried out on industry scale synthetic models in a HPC cloud system and reach a performance of 28TFlop/s, hence demonstrating Devito's suitability for production-grade seismic inversion problems.

## Introduction

Devito is a symbolic domain specific language (DSL) originally deigned to accelerate research and development in exploration geophysics and more specifically wave-equation based seismic inverse problems. The high-level interface, previously described in detail in [@devito-api], is built on top of `SymPy` [@sympy] and is inspired by the underlying mathematics and similar high-level DSLs such as FEniCS [@fenics] or Firedrake [@firedrake]. This interface allows user to formulate wave-equations, and more generally time-dependent partial differential equations(PDEs) in a simple and mathematically coherent way. The symbolic API then automatically generates finite-difference stencils from these mathematical expressions. One of the main advantages of [Devito] over other finite-difference DSLs is that generic expressions such as spares operations (i.e. point source injection or localized measurements) are fully supported and expressable in a high-level fashion as well. The second part of [Devito] is its compiler (c.f [@devito-compiler]), which generates highly optimized C code from the symbolic expressions and compiles it at runtime for the hardware at hand.

Earlier work introduced Devito for fairly simple problems, such as the acoustic isotropic wave-equation, and limited its applications to simple two-dimensional cases, while the generated code was benchmarked on large-scale three-dimensional problems for single-source experiments. In the work we present here, we demonstrate the scalability of Devito to real-world problems such as industry-scale anisotropic subsurface imaging and elastic modeling. These proof of concepts highlight two main contributions: First, we show that the symbolic interface and the compiler translates to large-scale adjoint-based inverse problem that require both massive compute, as thousands of PDEs need to be solved, as well as large amounts of memory, since the adjoint state method requires the forward state to be saved in memory. Second of all, we demonstrate with an elastic modelling example that [Devito] now fully supports and automatizes vectorial and second order tensorial staggered-grid finite-difference with the same high-level interface previously presented for a scalar field and a cartesian grid.

This paper is organized as follows: First, we provide a brief overview of [Devito] and its symbolic API and present the distributed memory implementation that allows large-scale modeling and inversion with domain decomposition. We then provide a brief comparison with a state of the art hand-coded wave propagator to validate the performance previously benchmarked with the roofline model ([@patterson, @devito-compiler, @devito-api, @louboutin2016ppf]). Next, we describe the two applications the demonstrate the scalability and staggered capabilities of [Devito]: We conduct a three-dimensional imaging example in the cloud-based on the tilted transverse isotropic wave equation (TTI, [@zhang-tti, @duveneck, @louboutin2018segeow]), as well as an elastic modeling example that highlights the vectorial and tensorial capabilities.


## Overview of Devito

Devito [@devito-api] provides a functional language built on top of `SymPy` [@sympy] to symbolically express PDEs at a mathematical level and implements automatic discretization with the finite difference method. The language is by design flexible enough to express other types of non finite-diffrence operators, such as interpolation and tensor contractions, that are inherent to measurement-based inverse problems. Several additional features are supported, among which staggered grids, sub-domains, and stencils with custom coefficients. The last major buildign block of a good PDE solver are the boundary conditions, that for finite difference methods are notoriously various and often complicated. The system is however sufficiently flexible to express them through composition of core mechanisms. For example, free surface and perfectly-matched layers (PMLs) boundary conditions can be expressed as equations -- just like any other PDE equations -- over a suitable sub-domain.

It is up to the Devito compiler to translate the symbolic specification into C code. The lowering of the input language down to C consists of several compilation passes, some of which introducing performance optimizations that are the key to fast code. Next to classic stencil optimizations (e.g., cache blocking, alignment, SIMD and OpenMP parallelism), Devito applies a series of flop-reducing transformations as well as aggressive loop fusion. This is to some extent elaborated in Section {TODO: PERF SECTIOn}, but for a complete treatment the interested reader should refer to [@devito-compiler].


### Symbolic language and compilation

In this section we illustrate the Devito language showing how to implement the acoustic wave-equation in isotropic media

```math {#acou}
\begin{cases}
 m \frac{d^2 u(t, x)}{dt^2} - Δ u(t, x) = \delta(xs) q(t) \\
 u(0, .) = \frac{d u(t, x)}{dt}(0, .) = 0 \\
 rec(t, .) = u(t, xr).
 \end{cases}
```

The core of the Devito symbolic API consists of three classes:

- `Grid`, a representation of the discretized model.
- `(Time)Function`, a representation of spatially (and time-) varying variables defined on a `Grid` object.
- `Sparse(Time)Function` a represention of (time-varying) point objects on a `Grid` object, generally unaligned with respect to the grid points, hence called "sparse".

A `Grid` represents a discretized finite n-dimensional space and is created as follows

```python
from devito import Grid
grid = Grid(shape=(nx, ny, nz), extent=(ext_x, ext_y, ext_z), origin=(o_x, o_y, o_z))
```

where `(nx, ny, nz)` are the number of grid points in each direction, `(ext_x, ext_y, ext_z)` is the physical extent of the domain in physical units (i.e `m`) and `(o_x, o_y, o_z)` is the origin of the domain in the same physical units. The object `grid` contains all the information related to the discretization such as the grid spacing. We use `grid` to create the symbolic objects that will be used to express the wave equation. First, we define a spatially varying model parameter `m` and a time-space varying field `u`

```python
from devito import Function, TimeFunction
m = Function(name="m", grid=grid, space_order=so)
u = TimeFunction(name="u", grid=grid, space_order=so, time_order=to)
```

where `so` is the spatial discretization order and `to` the time discretization order that is used for the generation of the finite-difference stencil. Second, we define point-wise objects such as point sources `src`, located at physical coordinates `xs`, and receiver (measurement) objects `rec`, with sensors located at the physical locations `xr`

```python
from devito import Function, TimeFunction
src = SparseFunction(name="src", grid=grid, npoint=1, coordinates=xs)
rec = SparseTimeFunction(name="rec", grid=grid, npoint=1, nt=nt, coordinates=xr)
```

The source term is handled separately from the PDE as a pointwise operation called `injection`, while the measurement is handled with an interpolation. By default, [Devito] initializes all `Function` data to 0, and thus automatically satisfies the zero Dirichlet condition at `t=0`. The isotropic acoustic wave-equation can then be implemented in [Devito] as below

```python
from devito import solve, Eq, Operator
eq = m * u.dt2 - u.laplace
stencil = [Eq(u.forward, solve(eq, u.forward))]
src_eqns = s1.inject(u.forward, expr=s1 * dt**2 / m)
rec_eqns = rec.interpolate(u)
```

To trigger compilation one needs to pass the constructed equations to an `Operator`. 

```python
from devito import Operator
op = Operator(stencil + src_eqns + rec_eqns)
```

The first compilation passes process the equations individually. The equations are lowered to an enriched representation, while the finite-difference constructs (e.g., derivatives) are translated into actual arithmetic operations. Subsequently, data dependency analysis is used to compute a performance-optimized topological ordering of the input equations (e.g., to maximize the likelihood of loop fusion) and to group them into so called "clusters". Basically, a cluster will eventually be a loop nest in the generated code, and consecutive clusters may share some outer loops. The ordered sequence of clusters undergoes several optimization passes, including cache blocking and flop-reducing transformations. It is then further lowered into an abstract syntax tree, and it is on such representation that parallelisms is introduced (SIMD, shared-memory, MPI). Finally, all remaining low-level aspects of code generation are handled, among which the most relevant one is the data management (e.g., definition of variables, transfers between host and device).

The output of the Devito compiler for the running example used in this section is available at [CodeSample] in `acou-so8.c`.

[CodeSample]:https://github.com/mloubout/SC20Paper/tree/master/gencode

### Distributed-memory parallelism

We here provide a succinct description of distributed-memory parallelism in Devito; the interested reader should refer to [@mpi-notebook] for thorough explanations and practical examples.

Devito implements distributed-memory parallelism on top of MPI. The design is such that users can almost entirely abstract away from it. Given any Devito code, just running it as

```python
DEVITO_MPI=1 mpirun -n X python ...
```

triggers the generation of code with routines for halo exchanges. The routines are scheduled at a suitable depth in the various loop nests thanks to data dependency analysis. The following optimizations are automatically applied:

- redundant halo exchanges are detected and dropped;
- computation/communication overlap, with prodding of the asynchronous progress engine by a designated thread through repeated calls to `MPI_Test`;
- a halo exchange is placed as far as possible from where it is needed to maximize computation/communication overlap;
- data packing and unpacking is threaded.

The domain decomposition occurs in Python upon creation of a `Grid` object. Exploiting the MPI Cartesian topology abstraction, Devito logically splits a grid based on the number of available MPI processes (users are given an "escape hatch" to override Devito's default decomposition strategy). `Function` and `TimeFunction` objects inherit the `Grid` decomposition. For `SparseFunction` objects the approach is different. Since a `SparseFunction` represents a sparse set of points, Devito looks at the physical coordinates of each point and, based on the `Grid` decomposition, schedules the logical ownership to an MPI rank. If a sparse point lies along the boundary of two or more MPI ranks, then it is duplicated in each of these ranks to be accessible by all neighboring processes. Eventually, a duplicated point may be redundantly computed by multiple processes, but any redundant increments will be discarded.

When accessing or manipulating data in a Devito code, users have the illusion to be working with classic NumPy arrays, while underneath they are actually distributed. All manner of NumPy indexing schemes (basic, slicing, etc.) are supported. In the implementation, proper global-to-local and local-to-global index conversion routines are used to propagate a read/write access to the impacted subset of ranks. For example, consider the array

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

Finally, we remark that while providing abstractions for distributed data manipulation, Devito does not support natively any mechanisms for parallel I/O. The distributed NumPy arrays, however, provide a generic and flexible infrastructure for the implementation of parallel I/O for any sort of file format (e.g., see [@witte2018alf]). 

## Industry-scale 3D seismic imaging in anisotropic media

One of the main applications of seismic finite-difference modeling in exploration geophysics is reverse-time migration (RTM), a wave equation based seismic imaging technique. Real-world seismic imaging presents a number of challenges that make applying this method to industry-scale problem sizes difficult. First of all, RTM requires an accurate representation of the physics through sophisticated wave equations such as the tilted-transverse isotropic (TTI) wave equation, for which both forward and adjoint implementations have to be provided. Secondly, wave equations must be solved for a large number of independent experiments, where each individual PDE solve in itself is expensive in terms of FLOPs and memory usage. For certain workloads, domain decomposition and checkpointing techniques have to be applied. In the following sections we describe an industry-scale seismic imaging problem presenting all of the aforementioned challenges, its implementation with Devito, and the results of an experimentation carried out on the Azure cloud using a synthetic data set.

### Anisotropic wave equation

In our seismic imaging case study, we use an anisotropic representation of the physics called tilted transverse isotropic modeling [@thomsen1986]. This representation for wave motion is one of the most widely used in exploration geophysics since it captures the leading order kinematics and dynamics of acoustic wave motion in highly heterogeneous elastic media where the medium properties vary more rapidly in the direction perpendicular to sedimentary strata [@alkhalifah2000; @baysal1983; @bubetti2012; @bubetti2014; @bubesatti2016; @chu2011; @duveneck; @fletcher; @fowlertti2010; @louboutin2018segeow; @whitmore1983; @witte2016segpve; @xu2014; @zhang2005; @zhang2011; @zhan2013]. The TTI wave equation is an acoustic, low dimensional (4 parameters, 2 wavefields) simplification of the 21 parameter and 12 wavefields tensorial equations of motions [@hooke]. This simplified representation is parametrized by the Thomsen parameters ``\epsilon(x), \delta(x)`` that relate to the global (many wavelength propagation) difference in propagation speed in the vertical and horizontal directions, and the tilt and azimuth angles ``\theta(x), \phi(x)`` that define the rotation of the vertical and horizontal axis around the cartesian directions. However, unlike the scalar isotropic acoustic wave-equation itself, the TTI wave equation is extremely computationally costly to solve and it is also not self-adjoint as shown in [@louboutin2018segeow].

The main complexity of the TTI wave equation is due to rotation of the symmetry axis of the physics that leads to rotated second-order finite-difference stencils. In order to ensure numerical stability, these rotated finite-difference operators are designed to be self-adjoint (c.f. [@zhang2011, @duveneck]). For example, we define the rotated second order derivative with respect to ``x`` as:

```math {#rot}
  G_{\bar{x}\bar{x}} &= D_{\bar{x}}^T D_{\bar{x}} \\
  D_{\bar{x}} &= \cos(\mathbf{\theta})\cos(\mathbf{\phi})\frac{\mathrm{d}}{\mathrm{d}x} + \cos(\mathbf{\theta})\sin(\mathbf{\phi})\frac{\mathrm{d}}{\mathrm{d}y} - \sin(\mathbf{\theta})\frac{\mathrm{d}}{\mathrm{d}z}.
```

where ``\theta`` is called the tilt angle (rotation around ``y``) and ``\phi`` is the azymuth angle (rotation around ``x``). We enable the simple expression of these complex stencil in [Devito] as the finite-difference shortcuts such as `.dx` are enabled not only for the basic types (i.e. `Function`) but for generic expressions. As a consequence, the rotated derivative defined in #rot write in [Devito] in two lines as:

```python
dx_u = cos(theta) * cos(phi) * u.dx + cos(theta) * sin(phi) * u.dy - sin(theta) * u.dz
dxx_u = (cos(theta) * cos(phi) * dx_u).dx.T + (cos(theta) * sin(phi) * dx_u).dy.T - (sin(theta) * dx_u).dz.T
```

Note that while the adjoint of the finite-difference stencil is enabled via the standard Python `.T` shortcut, the expression needs to be reordered by hand as the tilt and azymuth angle are spatially dependent and require to be inside the second pass of first-order derivative. We can see from these simple two lines that the rotated stencil involves all second-order derivatives (`.dx.dx`, `.dy.dy` and `.dz.dz`) and all second-order cross-derivatives (`dx.dy`, `.dx.dz` and `.dy.dz`) that leads to a denser stencil support and higher computational complexity (c.f. [@louboutin2016ppf]).

Because of the very high number of floating-point operations (FLOP) needed per grid point for the weighted rotated Laplacian, this anisotropic wave equation is extremely challenging to implement. As we show in Figure #ttiflops, and previously analysed in [@louboutin2016ppf], the computational cost with high-order finite-difference is in the order of thousands of FLOPs per grid point without optimizations.

#### Table: {#ttiflops}
| spatial order   | w/o optimizations | w/  optimizations |
|:----------------|:----------------- |:----------------- |
| 4               | 501               | 95                |
| 8               | 539               | 102               |
| 12              | 1613              | 160               |
| 16              | 5489              | 276               |

: Per-grid-point flops of the finite-difference stencil for the TTI wave-equation with different spatial discretization orders.

The version without flop-reducing optimizations is a direct translation of the discretized operators into stencil expressions. The version with optimizations employs transformations such as common sub-expressions elimination, factorization, and cross-iteration redundancy elimination -- the latter being key in removing the redundancies induced by mixed derivatives. Implementing all of these techniques manually is inherently difficult and laborious. Further, to obtain the desired performance improvements it is necessary to orchestrate them with aggressive loop fusion (for data locality), tiling (for data locality and tensor temporaries), and potentially ad-hoc vectorization strategies (if rotating registers are used as in [@cire1]). While an explanation of the optimization strategy employed by Devito is beyond the scope of this paper (see [@devito-compiler] for details), what should be appreciated here is that all this complexity is hidden away from the users.

With such complex physics, mathematics, and engineering, it becomes evident that the implementation of a solver for this wave-equation is exceptionally time-consuming and can lead to thousands of lines of code even for a single type of discretization. The verification of the result is no less complicated, since any small error is effectively untrackable and any change to the finite-difference scheme or to the time-stepper is difficult to achieve without substantial re-coding. Another complication stems from the fact that practitioners of seismic inversion are often geoscientists and not computer scientists/programmers. Unfortunately, this background often either results in poorly written low performant codes or it leads to complications when research codes are handed off to computer scientists who know how to write fast codes but who often miss the necessary geophysical domain knowledge. Neither situation is conducive to addressing the complexities that come with implementing codes based on the latest geophysical insights in geophysics and high-performance computing. With Devito on the other hand, both the forward and adjoint equations can be implemented in a few lines of Python code as illustrated with the rotated operator in #rotxpy\.

Simulation of wave motion is only one aspect of solving problems in seismology. During wave-equation based imaging, we also need to compute sensitivities (gradient) with respect to the quantities of interest. This imposes additional constraints on the design and implementations of our simulation codes as outlined in [@virieux]. Among several factors, such as fast setup time, we focused on correct and testable implementations for the adjoint wave equation and the gradient (action of the adjoint Jacobian) [@louboutin2018segeow, @louboutin2020THmfi].

### 3D Imaging example on Azure

We now demonstrate the scalability of [Devito] to real-world applications with the imaging of an industry-scale three-dimensional TTI subsurface model. This imaging was carried out in the cloud on Azure and takes advantage of recent work to port conventional cluster code to the Cloud using a serverless approach. The serverless implementation is described in detail in [@witte2019TPDedas, @witte2019SEGedw] and describes the steps to run computationnaly and financially efficient HPC workloads in the cloud. This imaging project, in collaboration with Azure, demonstrates the scalability and robustness of [Devito] to large scale wave equation based inverse problems and its cost-effectiveness in combination with a serverless implementation of seismic imaging in the cloud. In this example, we imaged a synthetic three-dimensional anisotropic subsurface model that mimics a realistic industry size problem with a realistic representation of the physics (TTI). The physical size of the problem is `10kmx10kmx2.8km` discretized on a `12.5m` grid with 40 points of absorbing layer on each sides that leads to `881x881x371` computational grid points (300 Million grid points). The final image is the sum of 1500 single-source images and we computed 100 single-source images in parallel with the available resources (200 nodes available, 2 nodes per source).

***Computational performance***

We briefly describe the computational setup and the performance achieved for this anisotropic imaging problem. Due to time constraints, and because the resources we were given access to for this proof of concept with Azure were limited, we did not have access to HPC virtual machines (VM) nor Infiniband enabled ones. The nodes we ran this experiment on are `Standard_E64_v3` and `Standard_E64s_v3` that while not HPC VM are memory optimized allowing to save the wavefield in memory for imaging (TTI adjoint state gradient [@virieux, @louboutin2018segeow]). These VMs are Intel&reg; Broadwell E5-2673 v4 2.3GH that are dual socket, 32 physical cores (and hyperthreading enabled) and 432Gb of DRAM. The overall inversion involved computing the image for 1500 source positions, i.e. solving 1500 forward and 1500 adjoint TTI wave equations. A single image required 600Gb of memory in single precision and we used two VMs per source with MPI with one rank per socket (4 MPI ranks per source) and imaged 100 sources in parallel due to resources limitations. The performance achieved, in single precision, was as follow:

- 140 GFLOP/s per VM;
- 280 GFLOP/s per source;
- 28 TFLOP/s for all 100 running sources;
- 110min runtime per source (forward + adjoint + image computation).

We also observe that if we had had access to more resources, we could have attempted imaging of all of the 1500 sources in parallel, which theoretically leads to a performance of 0.4PFLOP/s.

***How was the performance measured***

The execution time is computed through Python-level timers prefixed by an MPI barrier. The floating-point operations are counted once all of the symbolic flop-reducing transformations have been performed during compilation. Devito uses an in-house estimate of cost, rather than `SymPy`'s estimate, to take care of some low-level intricacies. For example, Devito's estimate ignores the cost of integer arithmetic used for offset indexing into multi-dimensional arrays. To calculate the total number of FLOPs performed, Devito multiplies the floating-point operations calculated at compile time by the size of the iteration space, and it does that at the granularity of individual expressions. Thanks to aggressive code motion, the amount of innermost-loop-invariant sub-expressions in a Devito Operator is typically negligible, so the Devito estimate doesn't basically suffer from this issue, or at least not in a tangible way to the best of our knowledge. The Devito-reported GFLOP/s were also checked against those produced by Intel Advisor on several single-node experiments: the differences -- typically Devito underestimating the achieved performance -- were always at most in the order of units, and therefore negligible.

***Imaging result***

The subsurface velocity model that was used in this study is an artificial anisotropic model that is designed and built combining two broadly known and used open-source SEG/EAGE acoustic velocity models. The anisotropy parameters are derived from a smoothed version of the velocity while the tilt angles were derived from a combination of the smooth velocity models and vertical and horizontal edge detection. The final seismic image of the subsurface model is plotted in Figure #OverTTI and highlights the fact that 3D seismic imaging based on a serverless approach and automatic code generation is feasible and provides good results on a realistic model.


#### Figure: {#OverTTI}
![](./Figures/OverTTI1.png){width=50%}
![](./Figures/OverTTI2.png){width=50%}\
![](./Figures/OverTTI3.png){width=50%}
![](./Figures/OverTTI4.png){width=50%}\
: 3D TTI imaging on a custom made model.

@witte2019TPDedas fully describes the serverless implementation of seismic inverse problems, including iterative algorithms for least-square minimization problems (LSRTM). The 3D anisotropic imaging results were presented as part of a keynote presentation at the EAGE HPC workshop in October 2019 [@herrmann2019EAGEHPCaii]. This work perfectly illustrates the flexibility and portability of Devito, as we were able to easily port a code only tested and developed on local hardware to the cloud, with only minor adjustments. This portability included the possibility to run MPI-based code for domain decomposition in the cloud, after developing it on a desktop computer. Our experiments are reproducibile using the instructions in a public repository [AzureTTI], which contains, among the other things, the Dockerfiles and Azure [batch-shipyard] setup.

## Elastic modelling

While the subsurface image in the previous Section is obtained with anisotropic propagators capable of mimicing the real physics, in order to model both the kinematics and the amplitudes correctly elastic propagator are required. These propagators are for example extremely important for global seismology, as the Shear waves (component ignored in TTI) are the most hazardous ones. In this section we exploit the tensor algebra language introduced in Devito v4.0 to express elastic waves in a compact and elegant notation.

The elastic isotropic wave equation, parametrized by the Lamé parameters ``\lambda, \mu`` and the density ``\rho`` reads:

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

The discretization of such a set of coupled PDEs requires five equations in two dimensions (two equations for the particle velocity and three for stress) and nine equations in three dimensions (three particle velocities and six stress equations). However, the mathematical definition only requires two coupled vector/tensor-valued equations for any number of dimensions.

### Tensor algebra language

We have augmented the Devito language with tensorial objects to enable a straightforward -- and at the same time mathematically rigorous -- definition of high-dimensional PDEs, such as the elastic wave equation in Eq#elas1\. This effort was inspired by [@ufl], a functional language for finite element methods. 

The extended Devito language introduces two new types, `VectorFunction` (and `VectorTimeFunction`) for vectorial objects such as the particle velocity, and `TensorFunction` (and `TensorTimeFunction`) for second-order tensor objects (matrices) such as the stress. These new objects are constructed the exact same way as the scalar `Function` objects. They also automatically implement staggered grid and staggered finite-differences with the possibility of half-node averaging. Each component of a tensorial object -- a (scalar) Devito `Function` -- is accessible via conventional vector notation (i.e. `v[0], t[0,1],....`).

With this extended language, the elastic wave equation can be defined in only four lines as follows:

```python
v = VectorTimeFunction(name='v', grid=model.grid, space_order=so, time_order=1)
tau = TensorTimeFunction(name='t', grid=model.grid, space_order=so, time_order=1)

u_v = Eq(v.forward, model.damp * (v + s/rho*div(tau)))
u_t = Eq(tau.forward,  model.damp *  (tau + s * (l * diag(div(v.forward)) + mu * (grad(v.forward) + grad(v.forward).T))))
```

The `SymPy` expressions created by these commands can be displayed with `sympy.pprint` as shown in Figure #PrettyElas\. This representation reflects perfectly the mathematics while still providing computational portability and efficiency through the Devito compiler.

#### Figure: {#PrettyElas}
![](./Figures/vel_symb.png){width=100%}
: Update stencil for the particle velocity. The stencil for updating the stress component is left out for readability, as the equation does not fit onto a single page. However, it can be found in the Devito tutorial on elastic modelling on github.

### 2D example

We show the elastic particle velocity and stress for a broadly recognized 2D synthetic model, the elastic Marmousi-ii[@versteeg927, @marmouelas] model. The wavefields are shown on Figure #ElasWf and its corresponding elastic shot records are displayed in Figure #ElasShot\.

#### Figure: {#ElasWf}
![](./Figures/marmou_snap.png){width=100%}
: Particle velocities and stress at time ``t=3\text{s}`` for a source at 10m depth and `x=5\text{km}` in the marmousi-ii model.

#### Figure: {#ElasShot}
![](./Figures/pressure_marmou.png){width=30%}
![](./Figures/vz_marmou.png){width=30%}
![](./Figures/vx_marmou.png){width=30%}
: Seismic shot record for 5sec of modelling. `a` is the pressure (trace of stress tensor) at the surface (5m depth), `b` is the vertical particle velocity and `c` is the horizontal particle velocity at the ocean bottom (450m depth).

### 3D proof of concept

{>> This is the actual SEAM run but can't say it don't have license<<}

Finally, we modelled three dimensional elastic data in the Cloud to demonstrate the scalability of Devito to cluster-size problems in the Cloud. The model we chose mimics the reference model in geophysics known as the SEAM model [@fehler2011seam] that is a three dimensional extreme-scale synthetic representation of the subsurface. The physical dimension of the model are `45kmx35kmx15km` then discretized with a grid spacing of `20mx20mx10m` that led to a computational grid of `2250x1750x1500` grid points (5.9 billion grid points). One of the main challenges of elastic modelling is the extreme memory cost due to the number of wavefield. For a three dimensional propagator, a minimum of 21 fields need to be stored:

- Three particle velocities with two time steps (`v.forward` and `v`)
- Six stress with two time steps (`tau.forward` and `tau`)
- Three model parameters `lamda`, `mu` and `rho`

These 21 fields, with the grid we just describe, leads to a minimum of 461Gb of memory for modelling only. For this experiment, we obtained access to small HPC VM on azure called `Standard_H16r` that are 16 cores Intel Xeon E5 2667 v3 with no hyperthreading and used 32 nodes for a single source experiment (we solved a single wave equation). We used a 12th order discretization that leads to 2.8TFlop/time-step to be computed for this model and propagated the elastic wave for 16 seconds (23000 time steps). The modelling ran in 16 hours which converts to 1.1TFlop/s. While these number may seem low, the elastic kernel is extremely memory bound, while the TTI kernel is nearly compute bound (see rooflines in [@louboutin2016ppf, @devito-api, @devito-compiler]) making it more computationally efficient, in particular in combination with MPI. Future work with involve working on the InfiniBand enabled and true HPC VM on azure to achieve Cloud performance on par with state of the art Cluster performance.

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

The transition from academic size problems, such as the two-dimensional acoustic wave-equation, to real-world application can be challenging in particular when attempted as an afterthought. In this work, we showed that thanks to design principles that aimed at this type of application from the beginning, [Devtio] provides the scalability necessary. Firstavall we showed that we provide a high-level interface not only for simple scalar equations but for coupled PDEs and allow the expression of non-trivial differential operators in a simple and concise way. Second and most importantly, we demonstrated that the compiler enables large-scale with state-f-the art computational performance and programming paradigm. The single-node performance is on par with hand-coded codes while providing the flexibility  from the symbolic interface, and multi-node parallelism is integrated in the compiler and interface in a conservative way. Finally, we demonstrated that our abstraction provides to performance portability that enables both on-premise HPC and Cloud HPC.


[fdelmodc]:https://github.com/JanThorbecke/OpenSource.git
[AzureTTI]:https://github.com/slimgroup/Azure2019
[batch-shipyard]:https://batch-shipyard.readthedocs.io
[JUDI]:https://github.com/slimgroup/JUDI.jl
[Devito]:https://github.com/devitocodes/devito

## References
