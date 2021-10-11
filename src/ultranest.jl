using .UltraNest

export UltraReact

struct UltraReact <: ROSESoss.AbstractNested end

"""
    dynesty_sampler(logj; nlive=400, kwargs...)
Takes a problem defined by the logj likelihood object and runs the
dynesty on it using the default options and static sampler.

Returns a chain, state
"""
function ROSESoss.sample(::UltraReact, lj::Soss.ConditionalModel; kwargs...)
    lklhd, prt, tc, unflatten = ROSESoss._split_conditional(lj)

    lklhdvec(X) = lklhd.(eachrow(X))
    vpnames = String["p$i" for i in 1:dimension(tc)]
    prtransvec(X) = reduce(vcat, (x -> prt(x)').(eachrow(X)))

    sampler = ultranest.ReactiveNestedSampler(
                                        vpnames,
                                        lklhdvec,
                                        transform=prtransvec,
                                        vectorized=true
                                        )
    res = sampler.run(;kwargs...)


    #res = sampler.results
    samples, weights = res["weighted_samples"]["points"], res["weighted_samples"]["weights"]
    logz = res["logz"][end]
    logzerr = res["logzerr"][end]
    logl = res["logl"][end]

    stats = (logz=logz, logzerr=logzerr, logl=logl)
    return ROSESoss._create_tv(unflatten, samples, weights), stats
end
