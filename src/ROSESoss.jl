module ROSESoss
#Turn off precompilations because of GG bug https://github.com/cscherrer/Soss.jl/issues/267
__precompile__(false)

using Reexport
@reexport using Soss
@reexport using ROSE
using PyCall
using MCMCChains
using StatsBase: sample, median
using NestedSamplers
using Requires
using StructArrays
import Distributions as Dists
using MeasureTheory
using Random


# This is a hack for MeasureTheory since it wants the type to output
Base.rand(rng::AbstractRNG, ::Type{Float64}, d::Dists.ContinuousDistribution) = rand(rng, d)


const rad2μas = 180.0/π*3600*1e6

#These hold the pointers to the PyObjects that will store dynesty and ehtim
#and are loaded at initialization
const dynesty = PyNULL()
const ehtim  = PyNULL()


export extract_amps, extract_vis, extract_cphase,
       ObsChar, scandata,
       create_joint,
       dynesty_sampler, nested_sampler,
       threaded_optimize,
       chi2, plot_mean, plot_samples, plot_vis_comp, plot_amp_comp,
       plot_cp_comp, plot_res_density,
       SossModels, hform, transform,
       ehtim


include("ehtim.jl")
include("utility.jl")
include("inference.jl")
include("models.jl")
include("hypercube.jl")
using .SossModels


function __init__()
    copy!(dynesty, pyimport("dynesty"))
    copy!(ehtim, pyimport("ehtim"))
    #    @require BlackBoxOptim="a134a8b2-14d6-55f6-9291-3336d3ab0209"
    @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
        @require StatsPlots="f3b207a7-027a-5e70-b257-86293d7955fd" include("plots.jl")
    end
    @require UltraNest="6822f173-b0be-4018-9ee2-28bf56348d09" include("ultranest.jl")
    @require BlackBoxOptim="a134a8b2-14d6-55f6-9291-3336d3ab0209" include("bboptim.jl")
end


end #end ROSESoss
