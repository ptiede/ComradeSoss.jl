
"""
    LogJoint
Holds the information for the logjoint. This is really just a convinence object
to unify some algorithmic choices, and make it easier to add new samplers.
"""
struct LogJoint{D,M,T}
    data::D
    model::M
    transform::T
end

trans(lj::LogJoint{D,M,T}) where {D,M,T} = lj.transform::T
dim(lj) = trans(lj).dimension
gdata(lj::LogJoint{D,M,T}) where {D,M,T} = lj.data::D

function LogJoint(data, model)
    return LogJoint(data, model, xform(model|data))
end


function (ℓ::LogJoint)(x)
    (θ, logjac) = Soss.transform_and_logjac(trans(ℓ), x)
    data = gdata(ℓ)
    args = merge(θ, data)
    return logdensity(ℓ.model, args)::eltype(x) + logjac
end




"""
    create_ampcpjoint
Takes in a model, ampobs, cpobs file and creates a joint distribution.
Optionally, can fit for the gains. If true the gains are set to unity.
"""
function create_joint(model,
                      ampobs::ROSE.EHTObservation{F,A},
                      cpobs::ROSE.EHTObservation{F,P};
                      fitgains=false
                      ) where {F, A<:ROSE.EHTVisibilityAmplitudeDatum,P<:ROSE.EHTClosurePhaseDatum}
    uamp = ROSE.getdata(ampobs, :u)
    vamp = ROSE.getdata(ampobs, :v)
    bl = ROSE.getdata(ampobs, :baselines)
    s1 = first.(bl)
    s2 = last.(bl)
    erramp = ROSE.getdata(ampobs, :error)
    amps = ROSE.getdata(ampobs, :amp)

    u1cp = ROSE.getdata(cpobs, :u1)
    v1cp = ROSE.getdata(cpobs, :v1)
    u2cp = ROSE.getdata(cpobs, :u2)
    v2cp = ROSE.getdata(cpobs, :v2)
    u3cp = ROSE.getdata(cpobs, :u3)
    v3cp = ROSE.getdata(cpobs, :v3)
    errcp = ROSE.getdata(cpobs, :error)
    cps = ROSE.getdata(cpobs, :phase)

    joint = model(uamp=uamp,
                  vamp=vamp,
                  s1=s1,
                  s2=s2,
                  erramp=erramp,
                  u1cp = u1cp,
                  v1cp = v1cp,
                  u2cp = u2cp,
                  v2cp = v2cp,
                  u3cp = u3cp,
                  v3cp = v3cp,
                  errcp = errcp
                )
    if fitgains
        conditioned = (amp = amps, cphase = cps,)
    else
        conditioned = (amp = amps, cphase = cps,
                       AP=1.0, AZ=1.0, JC=1.0, SM=1.0,
                       AA=1.0, LM=1.0, SP=1.0)
    end
    return LogJoint(conditioned, joint)
end



function prior_transform(lj)
    priors = Soss.prior(lj.model.model,:g, keys(lj.data)...)
    pr = priors(lj.model.argvals)
    pnames = keys(priors.dists)
    pdists = [eval(priors.dists[n]) for n in pnames]
    return x->quantile.(pdists, x), pnames, pr
end


function resample_equal(chain)
    return sample(chain, Weights(vec(chain["weights"])), length(chain))
end


"""
    nested_sampler(logj; nlive=400, kwargs...)
Takes a problem defined by the logj likelihood object and runs the
NestedSampler.jl nested sampler algorithm on it. This constructs
the approporate likliehood and prior transform for you if you stick to one
of the predefined models.

Returns a chain, state, names
"""
function nested_sampler(lj;nlive=400, kwargs...)
    prt,pnames,pr = prior_transform(lj)

    function lklhd(x::Vector{T}) where {T}
        θ = NamedTuple{pnames, NTuple{length(x),T}}(x)
        var = merge(θ, lj.data)
        ℓ =  logdensity(lj.model, var)::T - logdensity(pr, θ)::T
        return isnan(ℓ) ? -1e10 : ℓ
    end

    sampler = Nested(lj.transform.dimension, nlive)
    model = NestedModel(lklhd, prt)
    chain, state = sample(model, sampler; dlogz=0.2, param_names=String.(collect(pnames)))
    return chain, state, pnames
end


"""
    dynesty_sampler(logj; nlive=400, kwargs...)
Takes a problem defined by the logj likelihood object and runs the
dynesty on it using the default options and static sampler.

Returns a chain, state
"""
function dynesty_sampler(lj; kwargs...)
    prt, pnames,pr = prior_transform(lj)


    function lklhd(x::Vector{T}) where {T}
        θ = NamedTuple{pnames, NTuple{length(x),T}}(x)
        var = merge(θ, lj.data)
        ℓ =  logdensity(lj.model, var)::T - logdensity(pr, θ)::T
        return isnan(ℓ) ? -1e10 : ℓ
    end

    sampler =  dynesty.NestedSampler(lklhd, prt, lj.transform.dimension; kwargs...)
    sampler.run_nested()
    res = sampler.results
    samples, weights = res["samples"], exp.(res["logwt"] .- res["logz"][end])
    logz = res["logz"][end]
    logzerr = res["logzerr"][end]

    vals = hcat(samples, weights)
    chain = Chains(vals, [pnames..., :weights], Dict(:internals => ["weights"]),evidence=logz)
    return chain, (logzerr=logzerr, ), pnames
end


"""
    threaded_optimize
Runs a optimizer `nopt` times that are split across how many threads are currently running.
Note this optimizes the unbounded version of the lj

Returns the best parameters and divergences sorted from best to worst fit.
"""
function threaded_optimize(nopt, lj, maxevals)
    results = [zeros(lj.transform.dimension) for _ in 1:nopt]
    divs = zeros(nopt)
    srange = [(-5.0,5.0) for i in 1:lj.transform.dimension]
    for i in 1:nopt
        res = bboptimize(x->-lj(x), SearchRange=srange, MaxFuncEvals=maxevals, TraceMode=:compact)
        results[i] = best_candidate(res)
        divs[i] = -best_fitness(res)
    end
    I = sortperm(divs, rev=true)
    return logj.transform.(results[I]), divs[I]
end
