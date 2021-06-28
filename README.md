# GraphFlux

Graph Neural Nets can be a tad finicky to work with today given the *incredibly* massive set of flavors they come in (inductive problems? transductive problems? temporal problems?). This repo, GraphFlux.jl represents early-stage experimentation in what a graph neural network library can be. It's very v0.0.1 (read: it's the bare minimum that actually *works*!).

The goal is to design a set of *tearable* abstractions: abstractions that are, at the high level good at helping you quickly build well established network architectures *and* just as good at getting out of the way when you start going off the path well travelled. 

I won't promise this to be the most stable GNN library right now (check out [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) for that, which is truly super neat). Within Julia, for an established API I'd recommend [GeometricFlux.jl](https://github.com/FluxML/GeometricFlux.jl) by FluxML.

GraphFlux.jl is first and foremost a research library for exploring applications of GNNs across a broad range of scientific fields. It has no stable API as of right now (read: everything can, and probably will change), because my goal right now is [not to get too attached](https://en.wikipedia.org/wiki/Sunk_cost) to a "clean and beautiful" technical implementation. GraphFlux.jl's purpose is singular: being an extremely adaptable tool to demonstrate that GNNs can provide powerful insight for real-world *problems*. Everything else is a delightful coincidence!