using .BlackBoxOptim: bboptimize, best_candidate, best_fitness

export BBO
Base.@kwdef struct BBO <: ROSESoss.AbstractOptimizer
    tracemode::Symbol = :compact
    maxevals::Int = 10_000
end

function optimize(opt::BBO, lj::Soss.ConditionalModel)
    tc = ascube(lj)
    lower = transform(tc, zeros(dimension(tc)))
    upper = transform(tc, 0.999*ones(dimension(tc)))

    upflat, _ = ParameterHandling.flatten(upper)
    lowflat, unflatten = ParameterHandling.flatten(lower)

    bounds = [(lowflat[i], upflat[i]) for i in eachindex(upflat)]
    println(bounds)
    f(x) = -logdensity(lj, unflatten(x))
    res = bboptimize(f, SearchRange=bounds, MaxFuncEvals=opt.maxevals, TraceMode=opt.tracemode)
    return unflatten(best_candidate(res)), -best_fitness(res)
end
