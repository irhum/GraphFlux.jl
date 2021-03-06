# GraphFlux

Graph Neural Nets can be a tad finicky to work with today given the *incredibly* massive set of flavors they come in (inductive problems? transductive problems? temporal problems?). This repo, GraphFlux.jl represents early-stage experimentation in what a graph neural network library can be. It's built from the ground up in Julia (complete with CUDA kernels, also written in Julia itself). It's very v0.1 (read: it's the bare minimum that actually **works**!).

Because how dynamic GNNs are, the goal here is to design a set of *tearable* abstractions: abstractions that are, at the high level good at helping you quickly build well established network architectures *and* just as good at getting out of the way when you start going off the path well travelled. 

I won't promise this to be the most stable GNN library right now (check out [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) for that, which is truly super neat). Within Julia, for an established API I'd recommend [GeometricFlux.jl](https://github.com/FluxML/GeometricFlux.jl) by FluxML.

### So *what is this?*
GraphFlux.jl is 
* First and foremost a research library
* That hopes to build on the best practices of other GNN libraries
* To explore applications of GNNs across a broad range of scientific fields 
* Across a broad range of supported hardware (already supports CUDA GPUs, and TPU support is on the horizon once `libtpu` becomes available in Julia)

It has no stable API as of right now (read: everything can and probably will change), because my goal right now is [not to get too attached](https://en.wikipedia.org/wiki/Sunk_cost) to a "clean and beautiful" technical implementation. Perhaps some of its ideas will get merged into an existing library. Perhaps it's destined to grow into its own library.

Right now, GraphFlux.jl has exactly one singular, focused purpose: being an extremely adaptable tool to demonstrate that GNNs can provide powerful insight for real-world *problems*. Everything else is a delightful coincidence!

### A problem-first approach
To maximize learning, I'm currently focused on replicating key experiments from academic literature on GNNs (check `examples` and [here](https://github.com/irhum/GraphFlux.jl/blob/main/examples/info.md) for more on that). This also means that new features are built *as new problems require them*. For instance, there's no fused, memory-efficient message-aggregate operation yet (even though the Julia CUDA kernel would be fairly straightforward to write) because I've not run into a problem that requires it (yet). Once I do, it'll be written.

This "running head first into a problem" may be a tad strange, but it also ensures that this library, as it matures[^1] will become one whose API is best suited to solve a wide range of research problems (instead of trying to force problems through an overengineered API). 

[^1]: (*If* GraphFlux.jl matures. I'm not above killing my projects if I find they're not as promising as initially anticipated. Right now, I definitely hope this survives!)

### Update: why this got killed
* Training Neural ODEs is, quite simply, very slow, and did not result in a corresponding increase in performance on graph classification tasks to be worth the tradeoff.
* GeometricFlux.jl is already covering most of the regular use cases, so a duplication effort seems like a not great idea.
