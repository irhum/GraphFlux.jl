using NPZ, Flux, Setfield, DifferentialEquations
using Statistics

using Revise
using GraphFlux

cora = npzread("download/cora/cora.npz")

g = GraphTuple(nodes=cora["x"] ./ sum(cora["x"], dims=1), 
			senders=cora["edgelist"][:, 1], 
			receivers=cora["edgelist"][:, 2]) |> gpu

y = Flux.onehotbatch(cora["y"], 1:7) |> Array |> gpu
trainidx = cora["trainidx"] |> gpu
validx = cora["validx"] |> gpu

modelf(dim) = Chain(
			NodeLayer(Dropout(0.5)),
			NodeLayer(Dense(1433, dim, relu)),
			NodeLayer(Dropout(0.1)),
			# NodeLayer(x -> vcat(x, zero(x))),
			GraphNeuralODE(GCN(Dense(zeros(dim, dim), zeros(dim)), identity), (0.0f0, 1.0f0), Tsit5()),
			# GraphNeuralODE(GCN(Dense(zeros(dim*2, dim*2), zeros(dim*2)), identity), (0.0f0, 1.0f0), Tsit5()),
			# GraphNeuralODE(GCN(dim*2, dim*2), (0.0f0, 1.0f0), Tsit5()),
			# g -> relu.(g.nodes[1:dim, :]),
			g -> relu.(g.nodes),
			Dense(dim, 7))


function training_run()
	m = modelf(64) |> gpu
	ps = Flux.params(m)
	
	bestloss = Inf
	bestmodel = nothing
				
	loss(g, y, idxs) = Flux.Losses.logitcrossentropy(m(g)[:, idxs], y[:, idxs])
	
	opt = Flux.Optimise.Optimiser(Flux.Optimise.WeightDecay(0.001), Flux.Optimise.ADAMW(0.01))

	Flux.@epochs 100 begin
		gs = gradient(ps) do
			losstrn = loss(g, y, trainidx)
			return losstrn
		end

		lossval = loss(g, y, validx)
		
		if lossval < bestloss
			@show bestloss = lossval

			acc = sum(Flux.onecold(m(g))[validx] .== Flux.onecold(y)[validx]) / length(y[validx])
			@show acc

			bestmodel = deepcopy(m)
		end
		Flux.update!(opt, ps, gs)
	end
	
	return bestmodel
end

bmod = training_run()
Flux.Losses.logitcrossentropy(bmod(g)[:, validx], y[:, validx])

sum(length, Flux.params(bmod))

# bmod[4].re(bmod[4].p).l.W
using BSON
bparams = Flux.params(bmod |> cpu)
BSON.@save "download/coraode.bson" bparams


