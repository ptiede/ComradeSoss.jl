function create_joint(model,
                      ampobs::ROSE.EHTObservation{F,A};
                      fitgains=false
                      ) where {F, A<:ROSE.EHTVisibilityAmplitudeDatum}
    uamp = ROSE.getdata(ampobs, :u)
    vamp = ROSE.getdata(ampobs, :v)
    bl = ROSE.getdata(ampobs, :baselines)
    s1 = first.(bl)
    s2 = last.(bl)
    stations = Tuple(unique(vcat(s1,s2)))
    gpriors = values(select(amppriors, stations))
    erramp = ROSE.getdata(ampobs, :error)
    amps = ROSE.getdata(ampobs, :amp)


    joint = va(
                image=model,
                gamps=gamps(stations=stations, spriors=gpriors,),
                uamp=uamp,
                vamp=vamp,
                s1=s1,
                s2=s2,
                erramp=erramp
                )
    conditioned = (amp = amps,)
    return joint | conditioned
end



"""
    create_ampcpjoint
Takes in a model, ampobs, cpobs file and creates a joint distribution.
Optionally, can fit for the gains. If true the gains are set to unity.
"""
function create_joint(model,
                      ampobs::ROSE.EHTObservation{F,A},
                      cpobs::ROSE.EHTObservation{F,P};
                      amppriors=(AA=0.1,AP=0.1,AZ=0.1,LM=0.2,JC=0.1,PV=0.1,SM=0.1)
                      ) where {F, A<:ROSE.EHTVisibilityAmplitudeDatum,P<:ROSE.EHTClosurePhaseDatum}
    uamp = ROSE.getdata(ampobs, :u)
    vamp = ROSE.getdata(ampobs, :v)
    bl = ROSE.getdata(ampobs, :baselines)
    s1 = first.(bl)
    s2 = last.(bl)
    stations = Tuple(unique(vcat(s1,s2)))
    gpriors = values(select(amppriors, stations))
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

    joint = vacp(
                  image=model,
                  gamps=gamps(stations=stations, spriors=gpriors,),
                  uamp=uamp,
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
    conditioned = (amp = amps, cphase = cps,)
    return joint | conditioned
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

    joint = vis(image = model, u=u,
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

    return joint | conditioned
end


function create_joint(model,
                     dlca::ROSE.EHTObservation{F,A},
                     dcp::ROSE.EHTObservation{F,P};
                     kwargs...) where {F,
                                       A<:ROSE.EHTLogClosureAmplitudeDatum,
                                       P<:ROSE.EHTClosurePhaseDatum}

    u1a = getdata(dlca, :u1)
    v1a = getdata(dlca, :v1)
    u2a = getdata(dlca, :u2)
    v2a = getdata(dlca, :v2)
    u3a = getdata(dlca, :u3)
    v3a = getdata(dlca, :v3)
    u4a = getdata(dlca, :u4)
    v4a = getdata(dlca, :v4)
    lcamp = getdata(dlca, :amp)
    errcamp = getdata(dlca, :error)

    u1cp = getdata(dcp, :u1)
    v1cp = getdata(dcp, :v1)
    u2cp = getdata(dcp, :u2)
    v2cp = getdata(dcp, :v2)
    u3cp = getdata(dcp, :u3)
    v3cp = getdata(dcp, :v3)
    cp = getdata(dcp, :phase)
    errcp = getdata(dcp, :error)

    m = lcacp(
              image=model,
              u1a=u1a,v1a=v1a,
              u2a=u2a,v2a=v2a,
              u3a=u3a,v3a=v3a,
              u4a=u4a,v4a=v4a,
              errcamp=errcamp,
              u1cp=u1cp,v1cp=v1cp,
              u2cp=u2cp,v2cp=v2cp,
              u3cp=u3cp,v3cp=v3cp,
              errcp=errcp
        )

    conditioned = (lcamp = lcamp, cphase = cp,)
    return m | conditioned
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
    tc = hform(lj)
    pr = Soss.prior(lj.model, Soss.observed(lj)...)
    function lklhd(x::Vector{T}) where {T}
        θ = transform(tc, x)
        ℓ::T =  logdensity(lj, θ)::T - logdensity(pr, θ)::T
        return convert(T,isnan(ℓ) ? -1e10 : ℓ)
    end

    sampler = Nested(dimension(tc), nlive)
    model = NestedModel(lklhd, x->transform(tc, x))
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
    tc = ascube(lj)
    pr = Soss.prior(lj.model, Soss.observed(lj)...)
    cpr = pr(argvals(lj))
    base = transform(tc, fill(0.5, dimension(tc)))
    med, unflatten = ParameterHandling.flatten(base)
    function lklhd(x::Vector{T}) where {T}
        θ = unflatten(x)
        ℓ::T =  logdensity(lj, θ)::T - logdensity(cpr, θ)::T
        return convert(T,isnan(ℓ) ? -1e10 : ℓ)
    end

    prt = x->first(ParameterHandling.flatten(transform(tc, x)))

    sampler =  dynesty.NestedSampler(lklhd, prt, length(med); kwargs...)
    sampler.run_nested(print_progress=progress)
    res = sampler.results
    samples, weights = res["samples"], exp.(res["logwt"] .- res["logz"][end])
    logz = res["logz"][end]
    logzerr = res["logzerr"][end]
    res = hcat(samples, weights)


    return _create_tv(unflatten, samples, weights), (logz=logz, logzerr=logzerr)
end


function _create_tv(unflatten, samples, weights)
    ts = TupleVector([unflatten(Vector(x)) for x in eachrow(samples)])
    tv = TupleVector(merge(getfield(ts, :data), (weights=weights,)))
    return tv
end
