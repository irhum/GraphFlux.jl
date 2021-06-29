### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 5061fe80-bbdf-443c-ab81-ac1a6e2d5160
using BSON, Flux, ProgressMeter, Setfield, Random, ROC, NPZ

# ╔═╡ f375b5a5-cb41-40c1-be03-e413e744c6be
using GraphFlux

# ╔═╡ 93e83138-d84a-11eb-1c94-c54c588f853d
# begin
# 	using Pkg
	
# 	Pkg.add(["BSON", "Flux", "ProgressMeter", "Setfield", "Random", "NPZ"])
# 	Pkg.add(url="https://github.com/diegozea/ROC.jl")
# end

# ╔═╡ e20d7b3d-6559-477d-a258-7fcfe690837b
md"""If you need to download the data the first time, uncomment the hidden cell below"""

# ╔═╡ f2e46818-2bce-4a70-9908-933b638e42d0
# begin
# 	run(`curl http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip --create-dirs -o "../download/hiv.zip"`)
# 	run(`unzip ../download/hiv.zip -d ../download`)

# 	Pkg.add(["CSV", "CodecZlib", "Mmap", "DataFrames"])
# 	using CSV, CodecZlib, Mmap, DataFrames
	
# 	function csv_to_arr(fname)
# 		memmap = Mmap.mmap(fname)
# 		csv = CSV.File(transcode(GzipDecompressor, memmap), header=false) 
# 		csv |> DataFrame |> Array
# 	end
	
# 	prefix = "../download/hiv/raw/"
	
# 	edgelist = csv_to_arr(prefix * "edge.csv.gz") .+ 1
	
# 	nodes = csv_to_arr(prefix * "node-feat.csv.gz")' .+ 1
# 	edges = csv_to_arr(prefix * "edge-feat.csv.gz")' .+ 1
	
# 	trainidx = csv_to_arr("../download/hiv/split/scaffold/train.csv.gz")[:, 1] .+ 1
# 	validx = csv_to_arr("../download/hiv/split/scaffold/valid.csv.gz")[:, 1] .+ 1
# 	testidx = csv_to_arr("../download/hiv/split/scaffold/test.csv.gz")[:, 1] .+ 1
	
# 	targets = csv_to_arr(prefix * "graph-label.csv.gz")[:, 1]
	
# 	numnodes = csv_to_arr(prefix * "num-node-list.csv.gz")[:, 1]
# 	numedges = csv_to_arr(prefix * "num-edge-list.csv.gz")[:, 1]
	
# 	graphdata = Dict("numnodes" => numnodes, "numedges" => numedges, "edgelist" => edgelist)
	
# 	features = Dict("nodes" => nodes, "edges" => edges, "targets" => targets)
# 	idxs = Dict("train" => trainidx, "valid" => validx, "test" => testidx)
	
# 	BSON.@save "../download/hiv.bson" graphdata features idxs
# end

# ╔═╡ 5adf7e4f-0eee-482c-a6d5-fb85b0de7bc3
BSON.@load "../download/hiv.bson" features graphdata idxs

# ╔═╡ 238a3c8d-70cd-4077-8804-68a5449da136
md"""We define the model here, leveraging heavily from `features/molecules/ogbhiv.jl`"""

# ╔═╡ 07cf68c9-6e8d-4316-92aa-df879b690039
begin
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
end

# ╔═╡ dda37dd5-b122-4666-acf2-a7750863fd2e
# model constructor
model() = Chain(GraphFlux.OGBAtomEncoder(300),
	            OGBGCNBlock(300),
	            OGBGCNBlock(300),
	            OGBGCNBlock(300),
	            OGBGCNBlock(300),
	            OGBGCNBlock(300; last=true), 
	            graphmeanpool,
	            Dense(300, 1), 
	            x -> x[1, :])


# ╔═╡ 5dd07e15-f9a5-41f4-ae79-b5eee2ba0087
md"""Total number of parameters match that used in the original paper :)"""

# ╔═╡ 3a9e78d2-5514-48b9-a1cf-3370fe92ebae
# number of parameters in model, total
[length(x) for x in Flux.params(model())] |> sum

# ╔═╡ 72bace50-0338-4f95-8c43-f74bf45d236e
md"""We define a training run here. There's probably a cleaner way of doing this, but this was fast."""

# ╔═╡ 947b7429-efb5-4cc5-a5d8-e72ba5c7fb97
begin
	NUM_TRIALS = 5
	EPOCHS = 100
	TRAIN_BATCHSIZE = 1024
end

# ╔═╡ a26bcee2-5c06-452a-bb79-9382890b5404
md"""We define temporary evaluation functions (e.g. `getroc`) in the hidden cell here, final evaluations are done inside Python with OGB Evaluators"""

# ╔═╡ ca67a788-5c58-42bf-8352-8ad7f3984b12
begin
	function getpreds(m, subset="valid")
		vcat([m(GraphFlux.getmolbatch(features, graphdata, idsub) |> gpu) for idsub in Iterators.partition(idxs[subset], 2048)]...)
	end
	
	function getroc(m, subset="valid")
		preds = getpreds(m, subset)
		AUC(roc(sigmoid.(preds) |> cpu, features["targets"][idxs[subset]]))
	end
end

# ╔═╡ 8cb4646b-bce3-4d21-a50c-d34cdcb6b64a
function training_run(model_num)
	# create a new model
	m = model() |> gpu
	ps = Flux.params(m)
	
	# Metrics need to be computed on model with best val roc as in here: 
	# https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/main_pyg.py
	bestvalroc = 0
	bestmodel = nothing

	# new loss fn and optimizers
	loss(x, y) = Flux.Losses.logitbinarycrossentropy(m(x), y)
	opt = Flux.Optimise.ADAM()

	allbatchidxs = Iterators.partition(Random.shuffle(idxs["train"]), TRAIN_BATCHSIZE)

	for epoch in 1:EPOCHS
		@showprogress for batchidxs in allbatchidxs
			X = GraphFlux.getmolbatch(features, graphdata, batchidxs) |> gpu
			y = features["targets"][batchidxs] |> gpu

			data = [(X, y)]
			Flux.train!(loss, ps, data, opt)
		end

		# visible in terminal
		rocauc = getroc(m, "valid") 
		@show (epoch, rocauc)
		
		if rocauc > bestvalroc
			@show bestvalroc = rocauc
			bestmodel = deepcopy(m)
		end
	end

	finishedmodel = bestmodel |> cpu
	BSON.@save "../download/modelhiv$(model_num).bson" finishedmodel
	testpreds = sigmoid.(getpreds(bestmodel, "test")) |> cpu
	valpreds = sigmoid.(getpreds(bestmodel, "valid")) |> cpu
	
	return valpreds, testpreds
end

# ╔═╡ c4871692-9a1f-40c6-b6f2-5d069a753918
begin
	allvalpreds = []
	alltestpreds = []
	
	for i in 1:NUM_TRIALS
		@show "training run $(i)"
		valpreds, testpreds = training_run(i)
		push!(alltestpreds, testpreds)
		push!(allvalpreds, valpreds)
	end
end

# ╔═╡ 3b9abe03-f709-443e-b463-1fe839e5a65b
npzwrite("../download/hivpreds.npz", 
	Dict("testpreds" => hcat(alltestpreds...), 
		"testtargets" => features["targets"][idxs["test"]],
		"valpreds" => hcat(allvalpreds...), 
		"valtargets" => features["targets"][idxs["valid"]]))

# ╔═╡ Cell order:
# ╠═93e83138-d84a-11eb-1c94-c54c588f853d
# ╠═5061fe80-bbdf-443c-ab81-ac1a6e2d5160
# ╟─e20d7b3d-6559-477d-a258-7fcfe690837b
# ╟─f2e46818-2bce-4a70-9908-933b638e42d0
# ╠═f375b5a5-cb41-40c1-be03-e413e744c6be
# ╠═5adf7e4f-0eee-482c-a6d5-fb85b0de7bc3
# ╟─238a3c8d-70cd-4077-8804-68a5449da136
# ╠═07cf68c9-6e8d-4316-92aa-df879b690039
# ╠═dda37dd5-b122-4666-acf2-a7750863fd2e
# ╟─5dd07e15-f9a5-41f4-ae79-b5eee2ba0087
# ╠═3a9e78d2-5514-48b9-a1cf-3370fe92ebae
# ╟─72bace50-0338-4f95-8c43-f74bf45d236e
# ╠═947b7429-efb5-4cc5-a5d8-e72ba5c7fb97
# ╠═8cb4646b-bce3-4d21-a50c-d34cdcb6b64a
# ╟─a26bcee2-5c06-452a-bb79-9382890b5404
# ╟─ca67a788-5c58-42bf-8352-8ad7f3984b12
# ╠═c4871692-9a1f-40c6-b6f2-5d069a753918
# ╠═3b9abe03-f709-443e-b463-1fe839e5a65b
