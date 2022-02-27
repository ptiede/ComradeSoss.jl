
#Turn off precompilations because of GG bug https://github.com/cscherrer/Soss.jl/issues/267
const fwhmfac = 2*sqrt(2*log(2))

using Soss
using Comrade
using MeasureTheory
import Distributions
const Dists = Distributions


gamps = @model stations, spriors begin
    #Gain amps
    σ ~ For(eachindex(spriors)) do i
        Dists.LogNormal(zero(spriors[i]), spriors[i])
    end
    g = NamedTuple{stations}(σ)
    return g
end

gphases = @model stations, spriors begin
    σ ~ For(2:(length(spriors))) do i
            Dists.truncated(Dists.Normal(zero(spriors[i]), spriors[i]), -π, π)
        end
    g = NamedTuple{stations}(((zero(eltype(σ)),)..., σ...))
    return g
end

function amp_gains(img, g1,g2, u, v)
    return g1*g2*Comrade.amplitude(img, u, v)
end

function _vamps(img, g, s1, s2, uamp, vamp)
    vm = similar(uamp, eltype(g))
    for i in eachindex(uamp)
        vm[i] = amp_gains(img,
                            g[s1[i]],
                            g[s2[i]],
                            uamp[i],
                            vamp[i]
                            )
    end
    return vm
end


va = @model image, gamps, uamp, vamp, s1, s2, erramp begin
    img ~ image
    g ~ gamps
    vm = _vamps(img, g, s1, s2, uamp, vamp)
    amp ~ Dists.MvNormal(vm, erramp)
end

vacp = @model image, gamps, uamp, vamp, s1, s2, erramp,
               u1cp, v1cp, u2cp, v2cp, u3cp, v3cp, errcp begin

    img ~ image
    g ~ gamps
    vm = _vamps(img, g, s1, s2, uamp, vamp)
    amp ~ Dists.MvNormal(vm, erramp)

    cp = Comrade.closure_phase.(Ref(img),
                            u1cp,
                            v1cp,
                            u2cp,
                            v2cp,
                            u3cp,
                            v3cp)

    cphase ~ Comrade.CPVonMises{(:μ,:σ)}(mcp, errcp)
                            #For(eachindex(cp)) do i
    #    Comrade.CPVonMises(cp[i], errcp[i])
    #end

end

function vis_gains(img, ga1,ga2, gp1,gp2, u, v)
    Δθ = gp1 - gp2
    s,c = sincos(Δθ)
    g1 = ga1
    g2 = ga2
    vis = visibility(img, u, v)
    vr = g1*g2*(real(vis)*c + imag(vis)*s)
    vi = g1*g2*(-real(vis)*s + imag(vis)*c)
    return vr+vi*im
end

function _vism(img, ga, gp, s1, s2, u, v)
    vm = similar(u,Complex{eltype(u)})
    for i in eachindex(u)
        vm[i] = vis_gains(img,
                          ga[s1[i]],
                          ga[s2[i]],
                          gp[s1[i]],
                          gp[s2[i]],
                          u[i],
                          v[i]
                        )
    end
    return vm
end



vis = @model image, gamps, gphases, u, v, s1, s2, error begin
    img ~ image
    ga ~ gamps
    gp ~ gphases
    vis = _vism(img, ga, gp, s1, s2, u, v)

    visr ~ Dists.MvNormal(real.(vis), err)
    visi ~ Dists.MvNormal(imag.(vis), err)

end

viswnoise = @model image, gamps, gphases, u, v, s1, s2, error begin
    img ~ image
    ga ~ gamps
    gp ~ gphases
    f ~ Dists.Uniform(0.0, 0.25)
    vis = _vism(img, ga, gp, s1, s2, u, v)
    vamps = abs.(vis)
    errwnoise = hypot.(err, f.*vamps)
    visr ~ Dists.MvNormal(real.(vis), errwnoise)
    visi ~ Dists.MvNormal(imag.(vis), errwnoise)
end



lcacp = @model image,
               u1a,v1a,u2a,v2a,u3a,v3a,u4a,v4a,errcamp,
               u1p,v1p,u2p,v2p,u3p,v3p,errcp begin
    img ~ image

    mlca = Comrade.logclosure_amplitude.(Ref(img), u1a, v1a,
                                               u2a, v2a,
                                               u3a, v3a,
                                               u4a, v4a,
                                    )
    lcamp ~ Dists.MvNormal(mlca, errcamp)
    #For(eachindex(mlca,errcamp)) do i
    #    Dists.Normal(mlca[i], errcamp[i])
    #end

    mcp = Comrade.closure_phase.(Ref(img),
                             u1cp,
                             v1cp,
                             u2cp,
                             v2cp,
                             u3cp,
                             v3cp)
    #cphase ~ Dists.MvNormal(mcp, errcp)
    cphase ~ Comrade.CPVonMises{(:μ,:σ)}(mcp, errcp)
end

#=
mringwb = @model N, M, fov begin
    diam ~ Dists.Uniform(25.0, 85.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    ma ~ Dists.Uniform(0.0, 0.5) |> iid(N)
    mp ~ Dists.Uniform(-1π, 1π) |> iid(N)
    α = ma.*cos.(mp)
    β = ma.*sin.(mp)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)
    f ~ Dists.Uniform(0.8, 1.2)
    mring = renormed(Comrade.MRing{N}(rad, α, β), f-f*floor)
    rimg = smoothed(mring,σ)

    coeff ~ Dists.Dirichlet(fill(M*M, 1.0))
    ϵ ~ Dists.LogNormal(0.0, 0.1)
    bimg = renormed(stretched(Comrade.RImage(coeff, Comrade.SqExpKernel(ϵ)),fov,fov), f*floor)
    img = rimg+bimg
    return img
end
=#
mring = @model N, fmin, fmax begin
    diam ~ Dists.Uniform(25.0, 85.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    ma ~ Dists.Uniform(0.0, 0.5) |> iid(N)
    mp ~ Dists.Uniform(-1π, 1π) |> iid(N)
    α = ma.*cos.(mp)
    β = ma.*sin.(mp)

    f ~ Dists.Uniform(fmin, fmax)
    mring = renormed(Comrade.MRing{N}(rad, α, β), f)
    img = smoothed(mring,σ)
    return img
end

mringwfloor = @model N, fmin, fmax begin
    diam ~ Dists.Uniform(25.0, 85.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    ma ~ Dists.Uniform(0.0, 0.5) |> iid(N)
    mp ~ Dists.Uniform(-1π, 1π) |> iid(N)
    α = ma.*cos.(mp)
    β = ma.*sin.(mp)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)
    f ~ Dists.Uniform(fmin, fmax)

    mring = renormed(Comrade.MRing{N}(rad, α, β), f*(1-floor))
    disk = renormed(stretched(Comrade.Disk(), rad, rad), f*floor)
    img = smoothed(mring+disk,σ)
    return img
end

mringwffloor = @model N, fmin, fmax begin
    diam ~ Dists.Uniform(25.0, 85.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    ma ~ Dists.Uniform(0.0, 0.5) |> iid(N)
    mp ~ Dists.Uniform(-1π, 1π) |> iid(N)
    α = ma.*cos.(mp)
    β = ma.*sin.(mp)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)
    f ~ Dists.Uniform(fmin, fmax)
    σf ~ Dists.Uniform(1.0, 40.0)

    mring = renormed(Comrade.MRing{N}(rad, α, β), f*(1-floor))
    disk = renormed(stretched(Comrade.Disk(), rad, rad), f*floor)
    img = smoothed(mring, σ) +smoothed(disk,σf)
    return img
end

mringwgfloor = @model N, fmin, fmax begin
    diam ~ Dists.Uniform(25.0, 85.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    ma ~ Dists.Uniform(0.0, 0.5) |> iid(N)
    mp ~ Dists.Uniform(-1π, 1π) |> iid(N)
    α = ma.*cos.(mp)
    β = ma.*sin.(mp)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)
    f ~ Dists.Uniform(fmin, fmax)
    dg ~ Dists.Uniform(40.0, 250.0)
    rg = dg/fwhmfac
    mring = smoothed(renormed(Comrade.MRing{N}(rad, α, β), f*(1-floor)), σ)
    g = renormed(stretched(Comrade.Gaussian(), rg, rg), f*floor)
    img = mring + g
    return img
end


smring = @model N, fmin, fmax begin
    diam ~ Dists.Uniform(25.0, 85.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    ma ~ Dists.Uniform(0.0, 0.5) |> iid(N)
    mp ~ Dists.Uniform(-1π, 1π) |> iid(N)
    α = ma.*cos.(mp)
    β = ma.*sin.(mp)

    #Stretch
    τ ~ Dists.Uniform( 0.0, 0.5)
    ξτ ~ Dists.Uniform(-π/2, π/2)
    scx = 1/sqrt(1-τ)
    scy = sqrt(1-τ)

    f ~ Dists.Uniform(fmin, fmax)

    mring = renormed(Comrade.MRing{N}(rad, α, β), f)
    img = smoothed(rotated(stretched(mring,scx,scy),ξτ),σ)
    return img
end


smringwfloor = @model N, fmin, fmax begin
    diam ~ Dists.Uniform(25.0, 85.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    ma ~ Dists.Uniform(0.0, 0.5) |> iid(N)
    mp ~ Dists.Uniform(-1π, 1π) |> iid(N)
    α = ma.*cos.(mp)
    β = ma.*sin.(mp)

    #Stretch
    τ ~ Dists.Uniform(0.0, 0.5)
    ξτ ~ Dists.Uniform(-π/2, π/2)
    scx = 1/sqrt(1-τ)
    scy = sqrt(1-τ)


    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)
    f ~ Dists.Uniform(fmin, fmax)

    mring = renormed(Comrade.MRing{N}(rad, α, β), f*(1-floor))
    disk = renormed(stretched(Comrade.Disk(), rad, rad), f*floor)
    img = smoothed(rotated(stretched(mring+disk,scx,scy),ξτ),σ)
    return img
end
