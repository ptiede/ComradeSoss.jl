import .Metaheuristics
const MH = Metaheuristics

Base.@kwdef struct MetaH{T<:MH.Algorithm} <: ROSESoss.AbstractOptimizer
    alg::T=ECA(options=MH.Options(f_calls_limit=10^5))
end

function optimize(opt::MetaH, lj::Soss.ConditionalModel)
    tc = ascube(lj)
    lower = transform(tc, zeros(dimension(tc)))
    upper = transform(tc, 0.999*ones(dimension(tc)))
    upflat, _ = ParameterHandling.flatten(upper)
    lowflat, unflatten = ParameterHandling.flatten(lower)

    bounds = [lowflat upflat]'
    f(x) = -logdensity(lj, unflatten(x))
    res = MH.optimize(f, bounds, opt.alg)
    show(res)

    return unflatten(MH.minimizer(res)), -MH.minimum(res)
end
