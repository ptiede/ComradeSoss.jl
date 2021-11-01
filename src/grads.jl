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
