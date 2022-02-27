using ForwardDiff
using DiffResults


struct GradObj{F,G}
    f::F
    cfg::G
end

function GradObj(f::Function, x::AbstractArray; chunksize::Int=5)
    cfg = ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{chunksize}())
    return GradObj(f, cfg)
end


struct LogProbDeriv{T,C,F,G}
    transform::T
    cm::C
    ℓπ::F
    ∇ℓπ::G
end
function LogProbDeriv(cm::Soss.ConditionalModel, chunksize::Int)
    transform = xform(cm)
    function ℓπ(x)
        p, logjac = TV.transform_and_logjac(transform, x)
        return logdensity(cm, p) + logjac
    end
    cfg = ForwardDiff.GradientConfig(ℓπ, rand(transform.dimension), ForwardDiff.Chunk{chunksize}())

    function ∇ℓπ(x)
        ForwardDiff.gradient(ℓπ, x, cfg)
    end
    ∇ℓπ(rand(transform.dimension))
    @time ∇ℓπ(rand(transform.dimension))
    return LogProbDeriv(transform, cm, ℓπ, ∇ℓπ)
end
