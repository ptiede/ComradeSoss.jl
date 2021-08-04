using Pkg
Pkg.activate(@__DIR__)
using ROSE
using ForwardDiff
import Distributions as Dists
using Soss
using Plots
using MCMCTempering
using AdvancedHMC
using StatsPlots
using StatsBase
using TransformVariables
using DiffResults
include("tempering.jl")
using .Tempering

test = @model u, v, err begin
    σ1 ~ Dists.truncated(Dists.Normal(10.0,10.0), 0.0, Inf)
    σ2 ~ Dists.truncated(Dists.Normal(10.0,10.0), 0.0, Inf)
    x1 ~ Dists.Normal(0.0, 10.0)
    y1 ~ Dists.Normal(0.0, 10.0)
    x2 ~ Dists.Normal(0.0, 10.0)
    y2 ~ Dists.Normal(0.0, 10.0)
    f1 ~ Dists.truncated(Dists.Normal(1.0, 1.0), 0.0, Inf)
    f2 ~ Dists.truncated(Dists.Normal(1.0, 1.0), 0.0, Inf)
    #img = #shifted(stretched(Gaussian(), σ1,σ1),x1,y1) + 
    img = renormed(shifted(stretched(Disk(), σ2,σ2),x2,y2), f2) + renormed(shifted(stretched(Gaussian(), σ1, σ1), x1, y1), f1)
    vis = visibility.(Ref(img), u, v)
    visr ~ Dists.MvNormal(real.(vis), err)
    visi ~ Dists.MvNormal(imag.(vis), err)
end

nuv = 500
u = [randn(nuv)*0.02..., 0.0001]
v = [randn(nuv)*0.02..., 0.0]
err = fill(0.001, nuv+1)

m = test(u=u, v=v, err=err)
truth = (σ2=15.0, σ1=2.0 ,  y2=5.0 , y1=-10.0 , x2=10.0, x1 = -10.0, f2=0.5, f1=1.0)
simvis = Soss.predict(m,truth)
visr = simvis[:visr]
visi = simvis[:visi]

scatter(hypot.(u,v), hypot.(visr, visi))
cm = m|(visr=visr, visi=visi)

chain, stats = dynesty_sampler(cm)
echain = sample(chain, Weights(vec(chain[:weights])), 5000)
density(echain[:σ2])

t = xform(cm)
function make_ℓπ(t, cm)
    function (x)
        θ, logjac = TransformVariables.transform_and_logjac(t, x)
        return logdensity(cm, θ) + logjac
    end
end

function make_∂ℓπ(ℓπ, dimension; chunksize=2)
    cfg = ForwardDiff.GradientConfig(ℓπ, zeros(dimension), ForwardDiff.Chunk{chunksize}())
    res = DiffResults.GradientResult(zeros(dimension))
    function (x)
        ForwardDiff.gradient!(res, ℓπ, x, cfg)
        return (DiffResults.value(res), DiffResults.gradient(res))
    end
end

ℓπ = make_ℓπ(t, cm)
∂ℓπ1 = make_∂ℓπ(ℓπ, t.dimension; chunksize=4)
function dp(p)
    m = stretched(Disk(), p[1], p[2])
    return abs2(visibility(m, 0.00, 0.0))
end

function make_dp(visr, visi, u, v, err)
    function (p)
        m = stretched(Disk(), p[1], p[2])
        vis = visibility.(Ref(m), u, v)
        return -0.5*sum(abs2, (visr.-real(vis))./err) -0.5*sum(abs2, (visi.-imag(vis))./err)
    end
end

using LogDensityProblems
using DynamicHMC
using Random
using Optim

P = LogDensityProblems.TransformedLogDensity(t, θ->logdensity(cm, θ))
∇P = LogDensityProblems.ADgradient(:ForwardDiff, P)
reporter = DynamicHMC.LogProgressReport(step_interval=10)
results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇P, 2000; initialization=(q=10 .* randn(t.dimension),), reporter = reporter)
hchain = transform.(Ref(t), first(results))


density(getproperty.(hchain, :f2))