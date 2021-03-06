
using HypercubeTransform

function _split_conditional(lj)
    tc = Base.invokelatest(HypercubeTransform.ascube, lj)
    pr = Soss.prune(lj.model, Soss.observed(lj)...)
    cpr = pr(argvals(lj))
    base = transform(tc, fill(0.5, dimension(tc)))
    med, unflatten = ParameterHandling.flatten(base)
    function lklhd(x::Vector{T}) where {T}
        θ = unflatten(x)
        ℓ::T =  logdensity(lj, θ)::T - logdensity(cpr, θ)::T
        return convert(T,isnan(ℓ) ? -1e10 : ℓ)
    end

    prt = x->first(ParameterHandling.flatten(transform(tc, x)))
    return lklhd, prt, tc, unflatten
end



function create_joint(model,
                      ampobs::Comrade.EHTObservation{F,A};
                      amppriors=(AA=0.1,AP=0.1,AZ=0.1,LM=0.2,JC=0.1,PV=0.1,SM=0.1, SP=0.1)
                      ) where {F, A<:Comrade.EHTVisibilityAmplitudeDatum}
    uamp = Comrade.getdata(ampobs, :u)
    vamp = Comrade.getdata(ampobs, :v)
    bl = Comrade.getdata(ampobs, :baseline)
    s1 = first.(bl)
    s2 = last.(bl)
    st = Tuple(unique(vcat(s1,s2)))
    gpriors = select(amppriors, st)
    spriors = values(gpriors)
    erramp = Comrade.getdata(ampobs, :error)
    amps = Comrade.getdata(ampobs, :amp)


    joint = va(
                image=model,
                gamps=gamps(spriors=spriors,stations=st),
                uamp=μas2rad.(uamp),
                vamp=μas2rad.(vamp),
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
                      ampobs::Comrade.EHTObservation{F,A},
                      cpobs::Comrade.EHTObservation{F,P};
                      amppriors=(AA=0.1,AP=0.1,AZ=0.1,LM=0.2,JC=0.1,PV=0.1,SM=0.1, SP=0.1)
                      ) where {F, A<:Comrade.EHTVisibilityAmplitudeDatum,P<:Comrade.EHTClosurePhaseDatum}
    uamp = Comrade.getdata(ampobs, :u)
    vamp = Comrade.getdata(ampobs, :v)
    bl = Comrade.getdata(ampobs, :baseline)
    s1 = first.(bl)
    s2 = last.(bl)
    st = Tuple(unique(vcat(s1,s2)))
    gpriors = select(amppriors, st)
    spriors = values(gpriors)
    stations = keys(gpriors)
    erramp = Comrade.getdata(ampobs, :error)
    amps = Comrade.getdata(ampobs, :amp)

    u1cp = Comrade.getdata(cpobs, :u1)
    v1cp = Comrade.getdata(cpobs, :v1)
    u2cp = Comrade.getdata(cpobs, :u2)
    v2cp = Comrade.getdata(cpobs, :v2)
    u3cp = Comrade.getdata(cpobs, :u3)
    v3cp = Comrade.getdata(cpobs, :v3)
    errcp = Comrade.getdata(cpobs, :error)
    cps = Comrade.getdata(cpobs, :phase)

    joint = vacp(
                  image=model,
                  gamps=gamps(stations=stations, spriors=spriors,),
                  uamp= μas2rad(uamp),
                  vamp= μas2rad(vamp),
                  s1=s1,
                  s2=s2,
                  erramp=erramp,
                  u1cp = μas2rad(u1cp),
                  v1cp = μas2rad(v1cp),
                  u2cp = μas2rad(u2cp),
                  v2cp = μas2rad(v2cp),
                  u3cp = μas2rad(u3cp),
                  v3cp = μas2rad(v3cp),
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
                      visobs::Comrade.EHTObservation{F,A},
                      amppriors,
                      phasepriors
                      ) where {F, A<:Comrade.EHTVisibilityDatum}

    u = Comrade.getdata(visobs, :u)
    v = Comrade.getdata(visobs, :v)
    bl = Comrade.getdata(visobs, :baseline)
    s1 = first.(bl)
    s2 = last.(bl)
    st = Tuple(unique(vcat(s1,s2)))
    gpriors = select(amppriors, st)
    ppriors = select(phasepriors, st)
    sapriors = values(gpriors)
    sppriors = values(ppriors)
    err = Comrade.getdata(visobs, :error)
    visr = Comrade.getdata(visobs, :visr)
    visi = Comrade.getdata(visobs, :visi)

    joint = vis(image = model,
                gamps=gamps(spriors=sapriors, stations=st),
                gphases=gphases(spriors=sppriors, stations=st),
                u=μas2rad(u),
                v=μas2rad(v),
                s1=s1,
                s2=s2,
                err=err
                )
    conditioned = (visr = visr, visi = visi,)

    return joint | conditioned
end


"""
    create_joint
Takes in a model, visobs file and creates a joint distribution.
Optionally, can fit for the gains. If true the gains are set to unity.
"""
function create_joint_wnoise(model,
                      visobs::Comrade.EHTObservation{F,A},
                      amppriors,
                      phasepriors,
                      gainamps,
                      gainphases
                      ) where {F, A<:Comrade.EHTVisibilityDatum}

    u = Comrade.getdata(visobs, :u)
    v = Comrade.getdata(visobs, :v)
    bl = Comrade.getdata(visobs, :baseline)
    s1 = first.(bl)
    s2 = last.(bl)
    st = Tuple(unique(vcat(s1,s2)))
    gpriors = select(amppriors, st)
    ppriors = select(phasepriors, st)
    sapriors = values(gpriors)
    sppriors = values(ppriors)
    err = Comrade.getdata(visobs, :error)
    visr = Comrade.getdata(visobs, :visr)
    visi = Comrade.getdata(visobs, :visi)

    joint = viswnoise(image = model,
                gamps=gamps(spriors=sapriors, stations=st),
                gphases=gphases(spriors=sppriors, stations=st),
                u=μas2rad(u),
                v=μas2rad(v),
                s1=s1,
                s2=s2,
                err=err
                )
    if gainamps && gainphases
        conditioned = (visr = visr, visi = visi,)
    elseif gainamps
        conditioned = (visr = visr, visi = visi,
                        gp = (σ = zeros(length(st)-1),),
                      )
    elseif gainphases
        conditioned = (visr = visr, visi = visi,
                        ga = (σ = ones(length(st)),),
                      )
    end

    return joint | conditioned
end




function create_joint(model,
                     dlca::Comrade.EHTObservation{F,A},
                     dcp::Comrade.EHTObservation{F,P};
                     kwargs...) where {F,
                                       A<:Comrade.EHTLogClosureAmplitudeDatum,
                                       P<:Comrade.EHTClosurePhaseDatum}

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
              u1a=μas2rad(u1a),v1a=μas2rad(v1a),
              u2a=μas2rad(u2a),v2a=μas2rad(v2a),
              u3a=μas2rad(u3a),v3a=μas2rad(v3a),
              u4a=μas2rad(u4a),v4a=μas2rad(v4a),
              errcamp=errcamp,
              u1cp= μas2rad(u1cp),v1cp= μas2rad(v1cp),
              u2cp= μas2rad(u2cp),v2cp= μas2rad(v2cp),
              u3cp= μas2rad(u3cp),v3cp= μas2rad(v3cp),
              errcp=errcp
        )

    conditioned = (lcamp = lcamp, cphase = cp,)
    return m | conditioned
end



abstract type AbstractOptimizer end

"""
    optimize(opt::AbstractOptimizer, cm::Soss.ConditionalModel)
Optimize a Soss conditional model to find the MAP. Usually this
requires bringing in other packages to load. For example to use
BlackBoxOptim you would do

```julia
using ComradeSoss, BlackBoxOptim

opt = BBO()
cm = ... #create log joint
x, logMAP = optimize(opt, cm)
```
"""
function optimize end

abstract type AbstractSampler end
abstract type AbstractMCMC end
abstract type AbstractNested <: AbstractSampler end

Base.@kwdef struct DynestyStatic <: AbstractNested
    nlive::Int = 500
    bound::String = "multi"
    sample::String = "auto"
    walks::Int = 25
    slices::Int = 5
    max_move::Int = 100
end

Base.@kwdef struct DynestyDynamic <: AbstractNested
    bound::String="multi"
    sample::String="auto"
    walks::Int = 25
    slices::Int = 5
    max_move::Int = 100
end

function StatsBase.sample(dy::DynestyDynamic, lj::Soss.ConditionalModel; progress=true, kwargs...)
    lklhd, prt, tc, unflatten = _split_conditional(lj)

    sampler =  dynesty.dynamicsampler.DynamicSampler(lklhd,
                                     prt,
                                     dimension(tc);
                                     bound=dy.bound,
                                     sample=dy.sample,
                                     walks = dy.walks,
                                     slices = dy.slices,
                                     max_move=dy.max_move
                                    )
    sampler.run_nested(;kwargs...)
    res = sampler.results
    samples, weights = res["samples"], exp.(res["logwt"] .- res["logz"][end])
    logz = res["logz"][end]
    logzerr = res["logzerr"][end]
    logl = res["logl"][end]
    stats = (logz=logz, logzerr=logzerr, logl=logl)
    return _create_tv(unflatten, samples, weights), stats
end


"""
    dynesty_sampler(logj; nlive=400, kwargs...)
Takes a problem defined by the logj likelihood object and runs the
dynesty on it using the default options and static sampler.

Returns a chain, state
"""
function StatsBase.sample(dy::DynestyStatic, lj::Soss.ConditionalModel; progress=true, kwargs...)
    lklhd, prt, tc, unflatten = _split_conditional(lj)

    sampler =  dynesty.NestedSampler(lklhd,
                                     prt,
                                     dimension(tc);
                                     nlive=dy.nlive,
                                     bound=dy.bound,
                                     sample=dy.sample,
                                     walks = dy.walks,
                                     slices = dy.slices,
                                     max_move=dy.max_move
                                    )
    sampler.run_nested(;kwargs...)
    res = sampler.results
    samples, weights = res["samples"], exp.(res["logwt"] .- res["logz"][end])
    logz = res["logz"][end]
    logzerr = res["logzerr"][end]
    logl = res["logl"][end]
    stats = (logz=logz, logzerr=logzerr, logl=logl)
    return _create_tv(unflatten, samples, weights), stats
end


function _create_tv(unflatten, samples, weights)
    ts = TupleVector([unflatten(Vector(x)) for x in eachrow(samples)])
    tv = TupleVector(merge(getfield(ts, :data), (weights=weights,)))
    return tv
end
