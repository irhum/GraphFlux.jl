using BSON 
using Flux 
using ProgressMeter
using Setfield
using ROC
using Random

using Revise
using GraphFlux

BSON.@load "hiv.bson" features graphdata idxs

struct OGBGCNBlock
    gcn 
    batchnorm
    dropout
    last
end

function OGBGCNBlock(embdim::Integer; droprate::AbstractFloat=0.5, last::Bool=false)
    OGBGCNBlock(GCNₑ(embdim, embdim),
                BatchNorm(embdim),
                Dropout(droprate), 
                last)
end

function (block::OGBGCNBlock)(g)
    g = block.gcn(g)
    nodeh = block.batchnorm(GraphFlux.nodes(g))
    if block.last nodeh = relu.(nodeh) end
    nodeh = block.dropout(nodeh)
    @set g.nodes = nodeh
end

Flux.@functor OGBGCNBlock

model = Chain(OGBAtomEncoder(300),
            OGBGCNBlock(300),
            OGBGCNBlock(300),
            OGBGCNBlock(300),
            OGBGCNBlock(300),
            OGBGCNBlock(300; last=true), 
            graphmeanpool,
            Dense(300, 1), 
            x -> x[1, :]) |> gpu 

ps = Flux.params(model)

loss(x, y) = Flux.Losses.logitbinarycrossentropy(model(x), y)
opt = Flux.Optimise.ADAM()

EPOCHS = 50
batchsize = 1024
# Training v0.0.1, let's go!
Flux.@epochs EPOCHS begin
    @showprogress for batchidxs in Iterators.partition(Random.shuffle(idxs["train"]), batchsize)
        X = GraphFlux.getmolbatch(features, graphdata, batchidxs) |> gpu
        y = features["targets"][batchidxs] |> gpu
        
        data = [(X, y)]
        Flux.train!(loss, ps, data, opt)
    end
    
    valpreds = vcat([model(GraphFlux.getmolbatch(features, graphdata, idsub) |> gpu) for idsub in Iterators.partition(idxs["valid"], 2048)]...)
    println(AUC(roc(sigmoid.(valpreds) |> cpu, features["targets"][idxs["valid"]])))
end


testpreds = vcat([model(GraphFlux.getmolbatch(features, graphdata, idsub) |> gpu) for idsub in Iterators.partition(idxs["test"], 2048)]...)
println(AUC(roc(sigmoid.(testpreds) |> cpu, features["targets"][idxs["test"]])))

# model = model |> cpu
# BSON.@save "molhiv.bson" model

# [length(x) for x in ps] |> sum