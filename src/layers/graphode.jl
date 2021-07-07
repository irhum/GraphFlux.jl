# modified from https://github.com/SciML/DiffEqFlux.jl/blob/ccc3dcd0ccb7c18a176fc30dc9cb964acf98d50b/src/neural_de.jl#L38
# solvers only know how to work with arrays, so modifies to let solver 
# know we're interested in error tolerances on the node feature vectors

struct GraphNeuralODE{M,P,RE,T,A,K} <: DiffEqFlux.NeuralDELayer
    model::M
    p::P
    re::RE
    tspan::T
    args::A
    kwargs::K

    function GraphNeuralODE(model,tspan,args...;p = nothing,kwargs...)
        _p,re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(args),typeof(kwargs)}(
            model,p,re,tspan,args,kwargs)
    end
end

function (n::GraphNeuralODE)(g::AbstractGraphTuple,p=n.p,reltol=1e-3,abstol=1e-3,save_everystep=false,save_start=false)
# function (n::GraphNeuralODE)(g::AbstractGraphTuple,p=n.p,save_everystep=false,save_start=false)
    function dudt_(u,p,t)
		g = (@set g.nodes = u)
		(n.re(p)(g)).nodes
	end

    ff = ODEFunction{false}(dudt_,tgrad=x->zero(x))
    prob = ODEProblem{false}(ff,g.nodes,n.tspan,p)
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    @set g.nodes = solve(prob,n.args...;sense=sense,reltol=reltol,abstol=abstol,save_everystep=save_everystep,save_start=save_start,n.kwargs...).u[end]
    # @set g.nodes = solve(prob,n.args...;sense=sense,save_everystep=save_everystep,save_start=save_start,n.kwargs...).u[end]
end