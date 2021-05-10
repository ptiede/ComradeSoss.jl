module ROSESoss

using Soss
using ROSE
using PyCall
using MCMCChains
using StatsBase
using BlackBoxOptim
using NestedSamplers
using Requires
using StructArrays
import Distributions as Dists
using MeasureTheory


const rad2μas = 180.0/π*3600*1e6

const dynesty = PyNULL()
const ehtim  = PyNULL()


export extract_amps, extract_vis, extract_cphase,
       ObsChar, scandata,
       mringVM2wf, smringVM2wf,
       create_joint,
       dynesty_sampler, nested_sampler, threaded_optimize,
       chi2, plot_mean, plot_samples, plot_vis_comp, plot_amp_comp,
       plot_cp_comp, plot_res_density


include("models.jl")
include("read_ehtim.jl")
include("utility.jl")
include("inference.jl")


function __init__()
    copy!(dynesty, pyimport("dynesty"))
    copy!(ehtim, pyimport("ehtim"))
    #    @require BlackBoxOptim="a134a8b2-14d6-55f6-9291-3336d3ab0209"
    @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
        @require StatsPlots="f3b207a7-027a-5e70-b257-86293d7955fd" include("plots.jl")
    end
end


end #end ROSESoss
