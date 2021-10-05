function create_joint(model,
                      ampobs::ROSE.EHTObservation{F,A};
                      amppriors=(AA=0.1,AP=0.1,AZ=0.1,LM=0.2,JC=0.1,PV=0.1,SM=0.1, SP=0.1)
                      ) where {F, A<:ROSE.EHTVisibilityAmplitudeDatum}
    uamp = ROSE.getdata(ampobs, :u)
    vamp = ROSE.getdata(ampobs, :v)
    bl = ROSE.getdata(ampobs, :baselines)
    s1 = first.(bl)
    s2 = last.(bl)
    st = Tuple(unique(vcat(s1,s2)))
    gpriors = select(amppriors, st)
    spriors = values(gpriors)
    erramp = ROSE.getdata(ampobs, :error)
    amps = ROSE.getdata(ampobs, :amp)


    joint = va(
                image=model,
                gamps=gamps(spriors=spriors,stations=stations),
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
                      amppriors=(AA=0.1,AP=0.1,AZ=0.1,LM=0.2,JC=0.1,PV=0.1,SM=0.1, SP=0.1)
                      ) where {F, A<:ROSE.EHTVisibilityAmplitudeDatum,P<:ROSE.EHTClosurePhaseDatum}
    uamp = ROSE.getdata(ampobs, :u)
    vamp = ROSE.getdata(ampobs, :v)
    bl = ROSE.getdata(ampobs, :baselines)
    s1 = first.(bl)
    s2 = last.(bl)
    st = Tuple(unique(vcat(s1,s2)))
    gpriors = select(amppriors, st)
    spriors = values(gpriors)
    stations = keys(gpriors)
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
                  gamps=gamps(stations=stations, spriors=spriors,),
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
                      amppriors=(AA=0.1,AP=0.1,AZ=0.1,LM=0.2,JC=0.1,PV=0.1,SM=0.1, SP=0.1)
                      ) where {F, A<:ROSE.EHTVisibilityDatum}

    u = ROSE.getdata(visobs, :u)
    v = ROSE.getdata(visobs, :v)
    bl = ROSE.getdata(visobs, :baselines)
    s1 = first.(bl)
    s2 = last.(bl)
    st = Tuple(unique(vcat(s1,s2)))
    gpriors = select(amppriors, st)
    spriors = values(gpriors)
    err = ROSE.getdata(visobs, :error)
    visr = ROSE.getdata(visobs, :visr)
    visi = ROSE.getdata(visobs, :visi)

    joint = vis(image = model,
                gamps=gamps(spriors=spriors, stations=stations),
                gphases=gphases(stations=stations),
                u=u,
                v=v,
                s1=s1,
                s2=s2,
                err=err
                )
    conditioned = (visr = visr, visi = visi,)

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



@inline function _split_conditional(lj)
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
    return lklhd, prt, tc, unflatten
end


abstract type AbstractOptimizer end

"""
    optimize(opt::AbstractOptimizer, cm::Soss.ConditionalModel)
Optimize a Soss conditional model to find the MAP. Usually this
requires bringing in other packages to load. For example to use
BlackBoxOptim you would do

```julia
using ROSESoss, BlackBoxOptim

opt = BBO()
cm = ... #create log joint
x, logMAP = optimize(opt, cm)
```
"""
function optimize end

abstract type AbstractSampler end
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

function sample(dy::DynestyDynamic, lj::Soss.ConditionalModel; progress=true, kwargs...)
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
function sample(dy::DynestyStatic, lj::Soss.ConditionalModel; progress=true, kwargs...)
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
