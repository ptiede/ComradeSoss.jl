using .NestedSamplers

export NestedStatic

Base.@kwdef struct NestedStatic{B} <: AbstractNested
    nlive::Int = 500
    bounds::B=NestedSamplers.Bounds.MultiEllipsoid
    proposal::Symbol=:auto
end

"""
    sample(ns::NestedStatic, logj; nlive=400, kwargs...)
Takes a problem defined by the logj likelihood object and runs the
NestedSampler.jl nested sampler algorithm on it. This constructs
the approporate likliehood and prior transform for you if you stick to one
of the predefined models.

Returns a chain, state, names
"""
function StatsBase.sample(ns::NestedStatic, lj::Soss.ConditionalModel; kwargs...)
    lklhd, prt, tc, unflatten = _split_conditional(lj)

    sampler = Nested(dimension(tc),
                     ns.nlive,
                     bounds=ns.bounds,
                     proposal=ns.proposal)
    model = NestedModel(lklhd, prt)
    chain, state = sample(model, sampler; chain_type=Array, kwargs...)
    logz = state[:logz]
    logzerr = state[:logzerr]
    logl = state[:logl]
    stats = (logz=logz, logzerr=logzerr, logl=logl)
    return _create_tv(unflatten, @view(chain[:,1:end-1]), @view(chain[:,end]) ), stats
end
