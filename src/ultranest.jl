using .UltraNest

export ultranest_sampler

"""
    dynesty_sampler(logj; nlive=400, kwargs...)
Takes a problem defined by the logj likelihood object and runs the
dynesty on it using the default options and static sampler.

Returns a chain, state
"""
function ultranest_sampler(lj; kwargs...)
    prt, pnames,pr = prior_transform(lj)
    prtransvec(X) = reduce(vcat, (x -> prt(x)').(eachrow(X)))


    function lklhd(x::Vector{T}) where {T}
        θ = NamedTuple{pnames, NTuple{length(x),T}}(x)
        var = merge(θ, gdata(lj))
        ℓ =  logdensity(lj.model, var)::T - logdensity(pr, var)::T
        return convert(T, isnan(ℓ) ? -1e10 : ℓ)
    end
    lklhdvec(X) = lklhd.(eachrow(X))
    vpnames = [String.(pnames)...]
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

    vals = hcat(samples, weights)
    chain = Chains(vals, [pnames..., :weights], Dict(:internals => ["weights"]),evidence=logz)
    return chain, (logzerr=logzerr, ), pnames
end
