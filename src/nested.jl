using .NestedSamplers

export nested_sampler

"""
    nested_sampler(logj; nlive=400, kwargs...)
Takes a problem defined by the logj likelihood object and runs the
NestedSampler.jl nested sampler algorithm on it. This constructs
the approporate likliehood and prior transform for you if you stick to one
of the predefined models.

Returns a chain, state, names
"""
function nested_sampler(lj::Soss.ConditionalModel;nlive=400,dlogz=0.001*nlive, kwargs...)
    lklhd, prt, tc, unflatten = _split_conditional(lj)

    sampler = Nested(dimension(tc), nlive)
    model = NestedModel(lklhd, prt)
    chain, state = sample(model, sampler; dlogz=dlogz, chain_type=Array)
    logz = state[:logz]
    logzerr = state[:logzerr]
    logl = state[:logl]
    stats = (logz=logz, logzerr=logzerr, logl=logl)
    return _create_tv(unflatten, @view(chain[:,1:end-1]), @view(chain[:,end]) ), stats
end
