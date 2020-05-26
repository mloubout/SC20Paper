# Summary

- Emphasize cost of scaling analysis in the cloud + add ref to AWS
- Add CX2 scaling with explicit caption and description
- Be explicit about available softwares and and difficulty of comparison. Mention contractors working on proper comparison.
- Put refs and text in intro and discussion on SLIM inversion results/paper for the cycle skippingand application side (JUDI, TWRI)
- Add ref and text to GMD for accuracy analysis (and link to notebook)

# Reviewer 1

Comments for Revision
    If you run your model with a variable number of cores, how is the weak and strong scaling? Can you comment on that? Further, you mention that you validate performance against a wave propagator - how about validation of simulation quality? How is that done?

# Reviewer 2

Comments for Revision
    I don't think any revision will change the nature of the work, which, as I mentioned in comments section is nicely done but not best suited for SC.

# Review of pap429s2 by Reviewer 3

Comments for Revision
    Perhaps restricting the discussion of both ease of use of Devito, with a comparison including scalability on one problem, say the 3D TTI problem against a known method and doing this in a cloud setting up to a significant number of nodes would make this a more appropriate paper for the SC audience. This might be too tall an order here. Dropping the discussion on the 3D elastic wave and including a table of comparison with the fdelmode for the 2D elastic wave problem or even just going through scalability comparison on the TTI problem against another code might demonstrate the value of Devito better.

# Reviewer 4

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
