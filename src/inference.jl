
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


function create_joint(model,
                      ampobs::ROSE.EHTObservation{F,A};
                      fitgains=false
                      ) where {F, A<:ROSE.EHTVisibilityAmplitudeDatum}
    u = ROSE.getdata(ampobs, :u)
    v = ROSE.getdata(ampobs, :v)
    bl = ROSE.getdata(ampobs, :baselines)
    s1 = first.(bl)
    s2 = last.(bl)
    err = ROSE.getdata(ampobs, :error)
    amps = ROSE.getdata(ampobs, :amp)


    joint = model(u=u,
                  v=v,
                  s1=s1,
                  s2=s2,
                  err=err,
                )
    if fitgains
        conditioned = (amp = amps,)
    else
        conditioned = (amp = amps,
                       AP=1.0, AZ=1.0, JC=1.0, SM=1.0,
                       AA=1.0, LM=1.0, SP=1.0, PV=1.0)
    end
    return LogJoint(conditioned, joint)
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
                       AA=1.0, LM=1.0, SP=1.0, PV=1.0)
    end
    return LogJoint(conditioned, joint)
end


"""
    create_joint
Takes in a model, visobs file and creates a joint distribution.
Optionally, can fit for the gains. If true the gains are set to unity.
"""
function create_joint(model,
                      visobs::ROSE.EHTObservation{F,A};
                      fitgains=false
                      ) where {F, A<:ROSE.EHTVisibilityDatum}

    u = ROSE.getdata(visobs, :u)
    v = ROSE.getdata(visobs, :v)
    bl = ROSE.getdata(visobs, :baselines)
    s1 = first.(bl)
    s2 = last.(bl)
    err = ROSE.getdata(visobs, :error)
    visr = ROSE.getdata(visobs, :visr)
    visi = ROSE.getdata(visobs, :visi)

    joint = model(u=u,
                  v=v,
                  s1=s1,
                  s2=s2,
                  err=err
                )
    if fitgains
        conditioned = (visr = visr, visi = visi,)
    else
        conditioned = (visr = visr, visi = visi,
                       aAP=1.0, aAZ=1.0, aJC=1.0, aSM=1.0,
                       aAA=1.0, aLM=1.0, aSP=1.0, aPV=1.0,
                       pAP=0.0, pAZ=0.0, pJC=0.0, pSM=0.0,
                       pAA=0.0, pLM=0.0, pSP=0.0, pPV = 0.0
                       )
    end

    return LogJoint(conditioned, joint)
end





function prior_transform(lj)
    priors = Soss.prior(lj.model.model, keys(lj.data)...)
    pr = priors(merge(lj.model.argvals, lj.model.obs))
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
        var = merge(θ, gdata(lj))
        ℓ::T =  logdensity(lj.model, var)::T - logdensity(pr, var)::T
        return convert(T,isnan(ℓ) ? -1e10 : ℓ)
    end

    sampler = Nested(lj.transform.dimension, nlive)
    model = NestedModel(lklhd, prt)
    chain, state = sample(model, sampler; dlogz=0.2, param_names=String.(collect(pnames)))
    return chain, state, pnames
end


"""
    nested_sampler(logj; nlive=400, kwargs...)
Takes a problem defined by the logj likelihood object and runs the
NestedSampler.jl nested sampler algorithm on it. This constructs
the approporate likliehood and prior transform for you if you stick to one
of the predefined models.

Returns a chain, state, names
"""
function nested_sampler(lj::Soss.ConditionalModel;nlive=400, kwargs...)
    prt,pnames,pr = prior_transform(lj)
    println(prt, pnames, pr)
    function lklhd(x::Vector{T}) where {T}
        θ = NamedTuple{pnames, NTuple{length(x),T}}(x)
        var = merge(θ, observations(lj))
        println(var)
        ℓ::T =  logdensity(lj, var)::T - logdensity(pr, var)::T
        return convert(T,isnan(ℓ) ? -1e10 : ℓ)
    end

    sampler = Nested(length(pnames), nlive)
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
function dynesty_sampler(lj::Soss.ConditionalModel; progress=true, kwargs...)
    prt, pnames,pr = prior_transform(lj)


    function lklhd(x::Vector{T}) where {T}
        θ = NamedTuple{pnames, NTuple{length(x),T}}(x)
        ℓ =  logdensity(lj, θ)::T - logdensity(pr, θ)::T
        return convert(T, isnan(ℓ) ? -1e10 : ℓ)
    end

    sampler =  dynesty.NestedSampler(lklhd, prt, length(pnames); kwargs...)
    sampler.run_nested(print_progress=progress)
    res = sampler.results
    samples, weights = res["samples"], exp.(res["logwt"] .- res["logz"][end])
    logz = res["logz"][end]
    logzerr = res["logzerr"][end]

    vals = hcat(samples, weights)
    chain = TupleVector(NamedTuple{merge(pnames, (:weights,))}(c for c in eachcol(vals)))
    return chain, (logz=logz, logzerr=logzerr, ), pnames
end

function prior_transform(lj::Soss.ConditionalModel)
    priors = Soss.prior(lj.model, observed(lj)...)
    println(priors)
    pr = priors(merge(argvals(lj), observations(lj)))
    pnames = keys(priors.dists)
    pdists = [eval(priors.dists[n]) for n in pnames]
    return x->quantile.(pdists, x), pnames, pr
end






"""
    dynesty_sampler(logj; nlive=400, kwargs...)
Takes a problem defined by the logj likelihood object and runs the
dynesty on it using the default options and static sampler.

Returns a chain, state
"""
function dynesty_sampler(lj; progress=true, kwargs...)
    prt, pnames,pr = prior_transform(lj)


    function lklhd(x::Vector{T}) where {T}
        θ = NamedTuple{pnames, NTuple{length(x),T}}(x)
        var = merge(θ, gdata(lj))
        ℓ =  logdensity(lj.model, var)::T - logdensity(pr, var)::T
        return convert(T, isnan(ℓ) ? -1e10 : ℓ)
    end

    sampler =  dynesty.NestedSampler(lklhd, prt, lj.transform.dimension; kwargs...)
    sampler.run_nested(print_progress=progress)
    res = sampler.results
    samples, weights = res["samples"], exp.(res["logwt"] .- res["logz"][end])
    logz = res["logz"][end]
    logzerr = res["logzerr"][end]

    vals = hcat(samples, weights)
    chain = Chains(vals, [pnames..., :weights], Dict(:internals => ["weights"]),evidence=logz)
    return chain, (logzerr=logzerr, ), pnames
end
