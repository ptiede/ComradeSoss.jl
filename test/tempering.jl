module Tempering
using MCMCTempering
using AdvancedHMC: DifferentiableDensityModel
export Joint, TemperedJoint


struct Joint{Tℓll} <: Function
    ℓjoint::Tℓll
end

function (joint::Joint)(θ)
    return joint.ℓjoint(θ)
end


struct TemperedJoint{Tℓref, Tℓll, T<:Real} <: Function
    ℓref      :: Tℓref
    ℓjoint :: Tℓll
    β           :: Real
end

function (tj::TemperedJoint)(θ)
    return tj.ℓref(θ) .+ ( (tj.ℓjoint(θ) .- tj.ℓref(θ)) .* tj.β)
end


function MCMCTempering.make_tempered_model(
    model::DifferentiableDensityModel,
    β::Real
)
    ℓπ_β = TemperedJoint(model.ℓπ.ℓref, model.ℓπ.ℓjoint, β)
    ∂ℓπ∂θ_β = TemperedJoint(model.∂ℓπ∂θ.ℓref, model.∂ℓπ∂θ.ℓjoint, β)
    model_β = DifferentiableDensityModel(ℓπ_β, ∂ℓπ∂θ_β)
    return model_β
end

function MCMCTempering.make_tempered_loglikelihood(
    model::DifferentiableDensityModel,
    β::Real
)
    function logπ(z)
        return model.ℓπ.ℓjoint(z) * β
    end
    return logπ
end
end
