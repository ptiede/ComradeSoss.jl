import NLopt
import TransformVariables
const TV = TransformVariables
import ReverseDiff

export NLoptim

Base.@kwdef struct NLoptim{N,G,T} <: ROSESoss.AbstractOptimizer
    nlopt::N
    grado::G
    transform::T
    dimension::Int
end

function NLoptimRev(lj::Soss.ConditionalModel; alg=:LD_LBFGS, xtol=1e-8, maxeval::Int=50_000)
    #Constuct our objective function and gradients
    tc = xform(lj)
    dim = tc.dimension

    function f(x)
        p, logjac = TV.transform_and_logjac(tc, x)
        return logdensity(lj, p) + logjac
    end

    grado = ReverseDiff.GradientTape(f, rand(dim))
    ctape = ReverseDiff.compile(grado)
    res = DiffResults.GradientResult(zeros(dim))
    function gradf!(x, grad)
        ReverseDiff.gradient!(res, ctape, x)
        grad .= DiffResults.gradient(res)
        return DiffResults.value(res)
    end
    g = zeros(dim)
    gradf!(zeros(dim), g)
    println("A derivative took ")
    @time gradf!(zeros(dim), g)
    println("Adjust your expectations accordingly")

    srange = [(-7.0,7.0) for _ in 1:dim]
    nlopt = NLopt.Opt(alg, length(srange))
    NLopt.lower_bounds!(nlopt, first.(srange))
    NLopt.upper_bounds!(nlopt, last.(srange))
    NLopt.max_objective!(nlopt, gradf!)
    NLopt.xtol_rel!(nlopt, xtol)
    NLopt.maxeval!(nlopt, maxeval)
    return NLoptim(nlopt, grado, tc, dim)
end



function NLoptim(lj::Soss.ConditionalModel; alg=:LD_LBFGS, xtol=1e-8, maxeval::Int=50_000, chunksize::Int=5)
    #Constuct our objective function and gradients
    tc = xform(lj)
    dim = tc.dimension

    function f(x)
        p, logjac = TV.transform_and_logjac(tc, x)
        return logdensity(lj, p) + logjac
    end

    grado = GradObj(f, rand(dim); chunksize=chunksize)
    res = DiffResults.GradientResult(zeros(dim))
    function gradf!(x, grad)
        ForwardDiff.gradient!(res, f, x, grado.cfg)
        grad .= DiffResults.gradient(res)
        return DiffResults.value(res)
    end
    g = zeros(dim)
    gradf!(zeros(dim), g)
    println("A derivative took ")
    @time gradf!(zeros(dim), g)
    println("Adjust your expectations accordingly")

    srange = [(-7.0,7.0) for _ in 1:dim]
    nlopt = NLopt.Opt(alg, length(srange))
    NLopt.lower_bounds!(nlopt, first.(srange))
    NLopt.upper_bounds!(nlopt, last.(srange))
    NLopt.max_objective!(nlopt, gradf!)
    NLopt.xtol_rel!(nlopt, xtol)
    NLopt.maxeval!(nlopt, maxeval)
    return NLoptim(nlopt, grado, tc, dim)
end


function optimize(opt::NLoptim, start=nothing)
    dim = opt.dimension
    nlopt = opt.nlopt
    x0 = zeros(dim)
    lower = nlopt.lower_bounds
    upper = nlopt.upper_bounds
    if isnothing(start)
        x0 = lower .+ rand(dim).*(upper .- lower)
    else
        x0 .= TV.inverse(opt.transform, start)
    end


    (minf,minx,ret) = NLopt.optimize(nlopt, x0)
    println("NLOpt stopped because: $ret")
    return TV.transform(opt.transform, minx), minf
end
