# GraphFlux

Early-stage experimentation in what a graph neural network library can be. Very much a WIP. 

Eventual questions:
* How do you handle multiple input graphs per data point? (e.g. paratope-epitope predictions)
* How do you design abstractions that "dissolve away gracefully" (i.e. when the high level ones don't fit the research need, they *get out of the way* for lower-level access)
* How much flexibility can you have in input types while still maintaining performance on modern parallel hardware?