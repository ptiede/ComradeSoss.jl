
#Turn off precompilations because of GG bug https://github.com/cscherrer/Soss.jl/issues/267
const fwhmfac = 2*sqrt(2*log(2))

using Soss
using ROSE
using MeasureTheory
import Distributions as Dists


gamps = @model stations, spriors begin
    #Gain amps
    σ ~ For(eachindex(stations)) do i
        Dists.LogNormal(0.0, spriors[i])
    end
    return NamedTuple{stations}(σ)
end

gphases = @model stations begin
    σ ~ Dists.truncated(Dists.Normal(0.0, π), -π, π) |> iid(length(stations))
    return NamedTuple{stations}(σ)
end

va = @model image, gamps, uamp, vamp, s1, s2, erramp begin
    img ~ image
    g ~ gamps
    amp ~ For(eachindex(uamp,vamp, erramp)) do i
            g1 = g[s1[i]]
            g2 = g[s2[i]]
            mamp = g1*g2*ROSE.visibility_amplitude(img, uamp[i], vamp[i])
            Dists.Normal(mamp, erramp[i])
    end
end

vacp = @model image, gamps, uamp, vamp, s1, s2, erramp,
               u1cp, v1cp, u2cp, v2cp, u3cp, v3cp, errcp begin

    img ~ image
    g ~ gamps
    amp ~ For(eachindex(uamp,vamp, erramp)) do i
            g1 = g[s1[i]]
            g2 = g[s2[i]]
            mamp = g1*g2*ROSE.visibility_amplitude(img, uamp[i], vamp[i])
            Dists.Normal(mamp, erramp[i])
    end

    cphase ~ For(eachindex(u1cp, errcp)) do i
        mphase = ROSE.closure_phase(img,
                                    u1cp[i],
                                    v1cp[i],
                                    u2cp[i],
                                    v2cp[i],
                                    u3cp[i],
                                    v3cp[i]
                                    )
        ROSE.CPVonMises(mphase, errcp[i])
    end

end


vis = @model image, gamps, gphase, u, v, s1, s2, error begin
    img ~ image
    ga ~ gamps
    gp ~ gphases
    vis = ROSE.visibility.(Ref(img), u,v)

    visr ~ For(eachindex(u, v)) do i
        Δθ = gp[s1[i]] - gp[s2[i]]
        s,c = sincos(Δθ)
        g1 = ga[s1[i]]
        g2 = ga[s2[i]]
        mamp = g1*g2*(real(vis[i])*c + imag(vis[i])*s)
        Dists.Normal(mamp, err[i])
    end

    visi ~ For(eachindex(u, v)) do i
        Δθ = gp[s1[i]] - gp[s2[i]]
        s,c = sincos(Δθ)
        g1 = ga[s1[i]]
        g2 = ga[s2[i]]
        mamp = g1*g2*(-real(vis[i])*s + imag(vis[i])*c)
        Dists.Normal(mamp, err[i])
    end

end


lcacp = @model image,
               u1a,v1a,u2a,v2a,u3a,v3a,u4a,v4a,errcamp,
               u1p,v1p,u2p,v2p,u3p,v3p,errcp begin
    img ~ image

    lcamp ~ For(eachindex(errcamp)) do i
        ca = ROSE.logclosure_amplitude(img,
                                      u1a[i], v1a[i],
                                      u2a[i], v2a[i],
                                      u3a[i], v3a[i],
                                      u4a[i], v4a[i],
                                    )
        Dists.Normal(ca, errcamp[i])
    end

    cphase ~ For(eachindex(u1cp, errcp)) do i
        mphase = ROSE.closure_phase(img,
                                    u1cp[i],
                                    v1cp[i],
                                    u2cp[i],
                                    v2cp[i],
                                    u3cp[i],
                                    v3cp[i]
                                )
        ROSE.CPVonMises(mphase, errcp[i])
    end
end

mring = @model N begin
    diam ~ Dists.Uniform(25.0, 85.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    ma ~ Dists.Uniform(0.0, 0.5) |> iid(N)
    mp ~ Dists.Uniform(-1π, 1π) |> iid(N)
    α = ma.*cos.(mp)
    β = ma.*sin.(mp)

    f ~ Dists.Uniform(0.0, 5.0)
    mring = renormed(ROSE.MRing(rad, Tuple(α), Tuple(β)), f)
    img = smoothed(mring,σ)
    return img
end

mringwfloor = @model N begin
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
    f ~ Dists.Uniform(0.0, 5.0)

    mring = renormed(ROSE.MRing(rad, Tuple(α), Tuple(β)), f-f*floor)
    disk = renormed(stretched(ROSE.Disk(), rad, rad), f*floor)
    img = smoothed(mring+disk,σ)
    return img
end

smring = @model N begin
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

    f ~ Dists.Uniform(0.0, 5.0)

    mring = renormed(ROSE.MRing(rad, Tuple(α), Tuple(β)), f)
    img = smoothed(rotated(stretched(mring,scx,scy),ξτ),σ)
    return img
end


smringwfloor = @model N begin
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
    f ~ Dists.Uniform(0.0, 5.0)

    mring = renormed(ROSE.MRing(rad, Tuple(α), Tuple(β)), f-f*floor)
    disk = renormed(stretched(ROSE.Disk(), rad, rad), f*floor)
    img = smoothed(rotated(stretched(mring+disk,scx,scy),ξτ),σ)
    return img
end
