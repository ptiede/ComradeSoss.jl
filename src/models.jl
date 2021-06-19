
module SossModels

#Turn off precompilations because of GG bug https://github.com/cscherrer/Soss.jl/issues/267
__precompile__(false)
const fwhmfac = 2*sqrt(2*log(2))

using Soss
using ROSE
import Distributions as Dists




"""
    mring2VACP
Returns a fiducial mring model. That is, it contains a 2 mode mring model
with a floor that is a fraction of the flux of the mring. This include
gain amplitudes in fitting and fits visibility amplitudes and closure phases.
"""
mring1VACP = @model uamp, vamp, s1, s2, erramp, u1cp, v1cp, u2cp, v2cp, u3cp, v3cp, errcp begin
    diam ~ Dists.Uniform(25.0, 75.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    #First mring mode
    ma1 ~ Dists.Uniform(0.0,0.5)
    mp1 ~ Dists.Uniform(-1π,1π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)


    #Total flux
    f ~ Dists.Uniform(0.5, 5.0)

    #Gain amps
    AP ~ Dists.LogNormal(0.0, 0.1)
    AZ ~ Dists.LogNormal(0.0, 0.1)
    JC ~ Dists.LogNormal(0.0, 0.1)
    SM ~ Dists.LogNormal(0.0, 0.1)
    AA ~ Dists.LogNormal(0.0, 0.1)
    LM ~ Dists.LogNormal(0.0, 0.2)
    SP ~ Dists.LogNormal(0.0, 0.1)
    PV ~ Dists.LogNormal(0.0, 0.1)
    g = (AP=AP, AZ=AZ, JC=JC, SM=SM, AA=AA, LM=LM, SP=SP, PV=PV)


    mring = renormed(ROSE.MRing(rad, (α1,), (β1,)), f)
    img = smoothed(mring,σ)

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
        CPNormal(mphase, errcp[i])
    end
end

"""
    mring1wfVACP
Returns a fiducial mring model. That is, it contains a 2 mode mring model
with a floor that is a fraction of the flux of the mring. This include
gain amplitudes in fitting and fits visibility amplitudes and closure phases.
"""
mring1wfVACP = @model uamp, vamp, s1, s2, erramp, u1cp, v1cp, u2cp, v2cp, u3cp, v3cp, errcp begin
    diam ~ Dists.Uniform(25.0, 75.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    #First mring mode
    ma1 ~ Dists.Uniform(0.0,0.5)
    mp1 ~ Dists.Uniform(-1π,1π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)

    #Total flux
    f ~ Dists.Uniform(0.5, 5.0)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)

    #Gain amps
    AP ~ Dists.LogNormal(0.0, 0.1)
    AZ ~ Dists.LogNormal(0.0, 0.1)
    JC ~ Dists.LogNormal(0.0, 0.1)
    SM ~ Dists.LogNormal(0.0, 0.1)
    AA ~ Dists.LogNormal(0.0, 0.1)
    LM ~ Dists.LogNormal(0.0, 0.2)
    SP ~ Dists.LogNormal(0.0, 0.1)
    PV ~ Dists.LogNormal(0.0, 0.1)
    g = (AP=AP, AZ=AZ, JC=JC, SM=SM, AA=AA, LM=LM, SP=SP, PV=PV)


    mring = renormed(ROSE.MRing(rad, (α1,), (β1,)), f-f*floor)
    disk = renormed(stretched(ROSE.Disk(), rad, rad), f*floor)
    img = smoothed(mring+disk,σ)

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
        CPNormal(mphase, errcp[i])
    end
end

"""
    smring1wfVACP
Returns a fiducial mring model with a stretch added
"""
smring1wfVACP = @model uamp, vamp, s1, s2, erramp, u1cp, v1cp, u2cp, v2cp, u3cp, v3cp, errcp begin
    diam ~ Dists.Uniform(25.0, 75.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    #First mring mode
    ma1 ~ Dists.Uniform(0.0,0.5)
    mp1 ~ Dists.Uniform(-1π,1π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)

    #Stretch
    τ ~ Dists.truncated(Dists.Normal(0.0, 0.2), 0.0, 1.0)
    ξτ ~ Dists.Uniform(-π/2, π/2)
    scx = 1/sqrt(1-τ)
    scy = sqrt(1-τ)



    #Total flux
    f ~ Dists.Uniform(0.5, 5.0)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)

    #Gain amps
    AP ~ Dists.LogNormal(0.0, 0.1)
    AZ ~ Dists.LogNormal(0.0, 0.1)
    JC ~ Dists.LogNormal(0.0, 0.1)
    SM ~ Dists.LogNormal(0.0, 0.1)
    AA ~ Dists.LogNormal(0.0, 0.1)
    LM ~ Dists.LogNormal(0.0, 0.2)
    SP ~ Dists.LogNormal(0.0, 0.1)
    PV ~ Dists.LogNormal(0.0, 0.1)
    g = (AP=AP, AZ=AZ, JC=JC, SM=SM, AA=AA, LM=LM, SP=SP, PV=PV)


    mring = renormed(ROSE.MRing(rad, (α1,), (β1,)), f-f*floor)
    disk = renormed(stretched(ROSE.Disk(), rad, rad), f*floor)
    img = smoothed(rotated(stretched(mring+disk,scx,scy),ξτ),σ)

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
        CPNormal(mphase, errcp[i])
    end
end

"""
    mring2VACP
Returns a fiducial mring model. That is, it contains a 2 mode mring model
with a floor that is a fraction of the flux of the mring. This include
gain amplitudes in fitting and fits visibility amplitudes and closure phases.
"""
mring2VACP = @model uamp, vamp, s1, s2, erramp, u1cp, v1cp, u2cp, v2cp, u3cp, v3cp, errcp begin
    diam ~ Dists.Uniform(25.0, 75.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    #First mring mode
    ma1 ~ Dists.Uniform(0.0,0.5)
    mp1 ~ Dists.Uniform(-1π,1π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)

    #Second mring mode
    ma2 ~ Dists.Uniform(0.0,0.5)
    mp2 ~ Dists.Uniform(-1π,1π)
    α2 = ma2*cos(mp2)
    β2 = ma2*sin(mp2)


    #Total flux
    f ~ Dists.Uniform(0.5, 5.0)

    #Gain amps
    AP ~ Dists.LogNormal(0.0, 0.1)
    AZ ~ Dists.LogNormal(0.0, 0.1)
    JC ~ Dists.LogNormal(0.0, 0.1)
    SM ~ Dists.LogNormal(0.0, 0.1)
    AA ~ Dists.LogNormal(0.0, 0.1)
    LM ~ Dists.LogNormal(0.0, 0.2)
    SP ~ Dists.LogNormal(0.0, 0.1)
    PV ~ Dists.LogNormal(0.0, 0.1)
    g = (AP=AP, AZ=AZ, JC=JC, SM=SM, AA=AA, LM=LM, SP=SP, PV=PV)


    mring = renormed(ROSE.MRing(rad, (α1,α2), (β1,β2)), f)
    img = smoothed(mring,σ)

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
        CPNormal(mphase, errcp[i])
    end
end

"""
    mring2wfVACP
Returns a fiducial mring model. That is, it contains a 2 mode mring model
with a floor that is a fraction of the flux of the mring. This include
gain amplitudes in fitting and fits visibility amplitudes and closure phases.
"""
mring2wfVACP = @model uamp, vamp, s1, s2, erramp, u1cp, v1cp, u2cp, v2cp, u3cp, v3cp, errcp begin
    diam ~ Dists.Uniform(25.0, 75.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    #First mring mode
    ma1 ~ Dists.Uniform(0.0,0.5)
    mp1 ~ Dists.Uniform(-1π,1π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)

    #Second mring mode
    ma2 ~ Dists.Uniform(0.0,0.5)
    mp2 ~ Dists.Uniform(-1π,1π)
    α2 = ma2*cos(mp2)
    β2 = ma2*sin(mp2)


    #Total flux
    f ~ Dists.Uniform(0.5, 5.0)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)

    #Gain amps
    AP ~ Dists.LogNormal(0.0, 0.1)
    AZ ~ Dists.LogNormal(0.0, 0.1)
    JC ~ Dists.LogNormal(0.0, 0.1)
    SM ~ Dists.LogNormal(0.0, 0.1)
    AA ~ Dists.LogNormal(0.0, 0.1)
    LM ~ Dists.LogNormal(0.0, 0.2)
    SP ~ Dists.LogNormal(0.0, 0.1)
    PV ~ Dists.LogNormal(0.0, 0.1)
    g = (AP=AP, AZ=AZ, JC=JC, SM=SM, AA=AA, LM=LM, SP=SP, PV=PV)


    mring = renormed(ROSE.MRing(rad, (α1,α2), (β1,β2)), f-f*floor)
    disk = renormed(stretched(ROSE.Disk(), rad, rad), f*floor)
    img = smoothed(mring+disk,σ)

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
        CPNormal(mphase, errcp[i])
    end
end

"""
    smring2wfVACP
Returns a fiducial mring model with a stretch added
"""
smring2wfVACP = @model uamp, vamp, s1, s2, erramp, u1cp, v1cp, u2cp, v2cp, u3cp, v3cp, errcp begin
    diam ~ Dists.Uniform(25.0, 75.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    #First mring mode
    ma1 ~ Dists.Uniform(0.0,0.5)
    mp1 ~ Dists.Uniform(-1π,1π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)

    #Second mring mode
    ma2 ~ Dists.Uniform(0.0,0.5)
    mp2 ~ Dists.Uniform(-1π,1π)
    α2 = ma2*cos(mp2)
    β2 = ma2*sin(mp2)

    #Stretch
    τ ~ Dists.truncated(Dists.Normal(0.0, 0.2), 0.0, 1.0)
    ξτ ~ Dists.Uniform(-π/2, π/2)
    scx = 1/sqrt(1-τ)
    scy = sqrt(1-τ)



    #Total flux
    f ~ Dists.Uniform(0.5, 5.0)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)

    #Gain amps
    AP ~ Dists.LogNormal(0.0, 0.1)
    AZ ~ Dists.LogNormal(0.0, 0.1)
    JC ~ Dists.LogNormal(0.0, 0.1)
    SM ~ Dists.LogNormal(0.0, 0.1)
    AA ~ Dists.LogNormal(0.0, 0.1)
    LM ~ Dists.LogNormal(0.0, 0.2)
    SP ~ Dists.LogNormal(0.0, 0.1)
    PV ~ Dists.LogNormal(0.0, 0.1)
    g = (AP=AP, AZ=AZ, JC=JC, SM=SM, AA=AA, LM=LM, SP=SP, PV=PV)


    mring = renormed(ROSE.MRing(rad, (α1,α2), (β1,β2)), f-f*floor)
    disk = renormed(stretched(ROSE.Disk(), rad, rad), f*floor)
    img = smoothed(rotated(stretched(mring+disk,scx,scy),ξτ),σ)

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
        CPNormal(mphase, errcp[i])
    end
end


"""
    mring3wfVACP
Returns a fiducial mring model. That is, it contains a 2 mode mring model
with a floor that is a fraction of the flux of the mring. This include
gain amplitudes in fitting and fits visibility amplitudes and closure phases.
"""
mring3wfVACP = @model uamp, vamp, s1, s2, erramp, u1cp, v1cp, u2cp, v2cp, u3cp, v3cp, errcp begin
    diam ~ Dists.Uniform(25.0, 75.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    #First mring mode
    ma1 ~ Dists.Uniform(0.0,0.5)
    mp1 ~ Dists.Uniform(-1π,1π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)

    #Second mring mode
    ma2 ~ Dists.Uniform(0.0,0.5)
    mp2 ~ Dists.Uniform(-1π,1π)
    α2 = ma2*cos(mp2)
    β2 = ma2*sin(mp2)

    #Second mring mode
    ma3 ~ Dists.Uniform(0.0,0.5)
    mp3 ~ Dists.Uniform(-1π,1π)
    α3 = ma3*cos(mp3)
    β3 = ma3*sin(mp3)

    #Total flux
    f ~ Dists.Uniform(0.5, 5.0)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)

    #Gain amps
    AP ~ Dists.LogNormal(0.0, 0.1)
    AZ ~ Dists.LogNormal(0.0, 0.1)
    JC ~ Dists.LogNormal(0.0, 0.1)
    SM ~ Dists.LogNormal(0.0, 0.1)
    AA ~ Dists.LogNormal(0.0, 0.1)
    LM ~ Dists.LogNormal(0.0, 0.2)
    SP ~ Dists.LogNormal(0.0, 0.1)
    PV ~ Dists.LogNormal(0.0, 0.1)
    g = (AP=AP, AZ=AZ, JC=JC, SM=SM, AA=AA, LM=LM, SP=SP, PV=PV)


    mring = renormed(ROSE.MRing(rad, (α1,α2, α3), (β1,β2, β3)), f-f*floor)
    disk = renormed(stretched(ROSE.Disk(), rad, rad), f*floor)
    img = smoothed(mring+disk,σ)

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
        CPNormal(mphase, errcp[i])
    end
end




"""
    smring3wfVACP
Returns a fiducial mring model with a stretch added
"""
smring3wfVACP = @model uamp, vamp, s1, s2, erramp, u1cp, v1cp, u2cp, v2cp, u3cp, v3cp, errcp begin
    diam ~ Dists.Uniform(25.0, 75.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    #First mring mode
    ma1 ~ Dists.Uniform(0.0,0.5)
    mp1 ~ Dists.Uniform(-1π,1π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)

    #Second mring mode
    ma2 ~ Dists.Uniform(0.0,0.5)
    mp2 ~ Dists.Uniform(-1π,1π)
    α2 = ma2*cos(mp2)
    β2 = ma2*sin(mp2)

    #Third mring mode
    ma3 ~ Dists.Uniform(0.0,0.5)
    mp3 ~ Dists.Uniform(-1π,1π)
    α3 = ma3*cos(mp3)
    β3 = ma3*sin(mp3)


    #Stretch
    τ ~ Dists.truncated(Dists.Normal(0.0, 0.2), 0.0, 1.0)
    ξτ ~ Dists.Uniform(-π/2, π/2)
    scx = 1/sqrt(1-τ)
    scy = sqrt(1-τ)



    #Total flux
    f ~ Dists.Uniform(0.5, 5.0)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)

    #Gain amps
    AP ~ Dists.LogNormal(0.0, 0.1)
    AZ ~ Dists.LogNormal(0.0, 0.1)
    JC ~ Dists.LogNormal(0.0, 0.1)
    SM ~ Dists.LogNormal(0.0, 0.1)
    AA ~ Dists.LogNormal(0.0, 0.1)
    LM ~ Dists.LogNormal(0.0, 0.2)
    SP ~ Dists.LogNormal(0.0, 0.1)
    PV ~ Dists.LogNormal(0.0, 0.1)
    g = (AP=AP, AZ=AZ, JC=JC, SM=SM, AA=AA, LM=LM, SP=SP, PV=PV)


    mring = renormed(ROSE.MRing(rad, (α1,α2,α3), (β1,β2,β3)), f-f*floor)
    disk = renormed(stretched(ROSE.Disk(), rad, rad), f*floor)
    img = smoothed(rotated(stretched(mring+disk,scx,scy),ξτ),σ)

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
        CPNormal(mphase, errcp[i])
    end
end

"""
    mring4wfVACP
Returns a fiducial mring model
"""
mring4wfVACP = @model uamp, vamp, s1, s2, erramp, u1cp, v1cp, u2cp, v2cp, u3cp, v3cp, errcp begin
    diam ~ Dists.Uniform(25.0, 75.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    #First mring mode
    ma1 ~ Dists.Uniform(0.0,0.5)
    mp1 ~ Dists.Uniform(-1π,1π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)

    #Second mring mode
    ma2 ~ Dists.Uniform(0.0,0.5)
    mp2 ~ Dists.Uniform(-1π,1π)
    α2 = ma2*cos(mp2)
    β2 = ma2*sin(mp2)

    #Third mring mode
    ma3 ~ Dists.Uniform(0.0,0.5)
    mp3 ~ Dists.Uniform(-1π,1π)
    α3 = ma3*cos(mp3)
    β3 = ma3*sin(mp3)

    #Third mring mode
    ma4 ~ Dists.Uniform(0.0,0.5)
    mp4 ~ Dists.Uniform(-1π,1π)
    α4 = ma4*cos(mp4)
    β4 = ma4*sin(mp4)



    #Stretch
    τ ~ Dists.truncated(Dists.Normal(0.0, 0.2), 0.0, 1.0)
    ξτ ~ Dists.Uniform(-π/2, π/2)


    #Total flux
    f ~ Dists.Uniform(0.5, 5.0)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)

    #Gain amps
    AP ~ Dists.LogNormal(0.0, 0.1)
    AZ ~ Dists.LogNormal(0.0, 0.1)
    JC ~ Dists.LogNormal(0.0, 0.1)
    SM ~ Dists.LogNormal(0.0, 0.1)
    AA ~ Dists.LogNormal(0.0, 0.1)
    LM ~ Dists.LogNormal(0.0, 0.2)
    SP ~ Dists.LogNormal(0.0, 0.1)
    PV ~ Dists.LogNormal(0.0, 0.1)
    g = (AP=AP, AZ=AZ, JC=JC, SM=SM, AA=AA, LM=LM, SP=SP, PV=PV)


    mring = renormed(ROSE.MRing(rad, (α1,α2,α3, α4), (β1,β2,β3,β4)), f-f*floor)
    disk = renormed(stretched(ROSE.Disk(), rad, rad), f*floor)
    img = smoothed(mring+disk,σ)

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
        CPNormal(mphase, errcp[i])
    end
end


"""
    smring4wfVACP
Returns a fiducial mring model with a stretch added
"""
smring4wfVACP = @model uamp, vamp, s1, s2, erramp, u1cp, v1cp, u2cp, v2cp, u3cp, v3cp, errcp begin
    diam ~ Dists.Uniform(25.0, 75.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    #First mring mode
    ma1 ~ Dists.Uniform(0.0,0.5)
    mp1 ~ Dists.Uniform(-1π,1π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)

    #Second mring mode
    ma2 ~ Dists.Uniform(0.0,0.5)
    mp2 ~ Dists.Uniform(-1π,1π)
    α2 = ma2*cos(mp2)
    β2 = ma2*sin(mp2)

    #Third mring mode
    ma3 ~ Dists.Uniform(0.0,0.5)
    mp3 ~ Dists.Uniform(-1π,1π)
    α3 = ma3*cos(mp3)
    β3 = ma3*sin(mp3)

    #Third mring mode
    ma4 ~ Dists.Uniform(0.0,0.5)
    mp4 ~ Dists.Uniform(-1π,1π)
    α4 = ma4*cos(mp4)
    β4 = ma4*sin(mp4)



    #Stretch
    τ ~ Dists.truncated(Dists.Normal(0.0, 0.2), 0.0, 1.0)
    ξτ ~ Dists.Uniform(-π/2, π/2)
    scx = 1/sqrt(1-τ)
    scy = sqrt(1-τ)



    #Total flux
    f ~ Dists.Uniform(0.5, 5.0)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)

    #Gain amps
    AP ~ Dists.LogNormal(0.0, 0.1)
    AZ ~ Dists.LogNormal(0.0, 0.1)
    JC ~ Dists.LogNormal(0.0, 0.1)
    SM ~ Dists.LogNormal(0.0, 0.1)
    AA ~ Dists.LogNormal(0.0, 0.1)
    LM ~ Dists.LogNormal(0.0, 0.2)
    SP ~ Dists.LogNormal(0.0, 0.1)
    PV ~ Dists.LogNormal(0.0, 0.1)
    g = (AP=AP, AZ=AZ, JC=JC, SM=SM, AA=AA, LM=LM, SP=SP, PV=PV)


    mring = renormed(ROSE.MRing(rad, (α1,α2,α3, α4), (β1,β2,β3,β4)), f-f*floor)
    disk = renormed(stretched(ROSE.Disk(), rad, rad), f*floor)
    img = smoothed(rotated(stretched(mring+disk,scx,scy),ξτ),σ)

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
        CPNormal(mphase, errcp[i])
    end
end



"""
    smring2wfVis
Returns a fiducial mring model with a stretch added
"""
smring2wfVis = @model u, v, s1, s2, err begin
    diam ~ Dists.Uniform(25.0, 75.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    #First mring mode
    ma1 ~ Dists.Uniform(0.0,0.5)
    mp1 ~ Dists.Uniform(0.0,2π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)

    #Second mring mode
    ma2 ~ Dists.Uniform(0.0,0.5)
    mp2 ~ Dists.Uniform(0.0,2π)
    α2 = ma2*cos(mp2)
    β2 = ma2*sin(mp2)

    #Stretch
    τ ~ Dists.Uniform(0.0, 0.99)
    ξτ ~ Dists.Uniform(-π/2, π/2)
    scx = sqrt(1-τ)
    scy = 1/sqrt(1-τ)



    #Total flux
    f ~ Dists.Uniform(0.5, 5.0)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)



    mring = renormed(ROSE.MRing(rad, (α1,α2), (β1,β2)), f-f*floor)
    disk = renormed(stretched(ROSE.Disk(), rad, rad), f*floor)
    img = rotated(stretched(smoothed(mring+disk,σ),1.0,scy),ξτ)

    aAP ~ Dists.LogNormal(0.0, 0.1)
    aAZ ~ Dists.LogNormal(0.0, 0.1)
    aJC ~ Dists.LogNormal(0.0, 0.1)
    aSM ~ Dists.LogNormal(0.0, 0.1)
    aAA ~ Dists.LogNormal(0.0, 0.1)
    aLM ~ Dists.LogNormal(0.0, 0.2)
    aSP ~ Dists.LogNormal(0.0, 0.1)
    aPV ~ Dists.LogNormal(0.0, 0.1)
    ga = (AP=aAP, AZ=aAZ, JC=aJC, SM=aSM, AA=aAA, LM=aLM, SP=aSP, PV=aPV)

    pAP ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pAZ ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pJC ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pSM ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pAA ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pLM ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pSP ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pPV ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    gp = (AP=pAP, AZ=pAZ, JC=pJC, SM=pSM, AA=pAA, LM=pLM, SP=pSP, PV=pPV)


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

"""
    smring3wfVis
Returns a fiducial mring model with a stretch added
"""
smring3wfVis = @model u, v, s1, s2, err begin
    diam ~ Dists.Uniform(25.0, 75.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    #First mring mode
    ma1 ~ Dists.Uniform(0.0,0.5)
    mp1 ~ Dists.Uniform(-1π,1π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)

    #Second mring mode
    ma2 ~ Dists.Uniform(0.0,0.5)
    mp2 ~ Dists.Uniform(-1π,1π)
    α2 = ma2*cos(mp2)
    β2 = ma2*sin(mp2)

    #Third mring mode
    ma3 ~ Dists.Uniform(0.0,0.5)
    mp3 ~ Dists.Uniform(-1π,1π)
    α3 = ma3*cos(mp3)
    β3 = ma3*sin(mp3)


    #Stretch
    τ ~ Dists.truncated(Dists.Normal(0.0, 0.2), 0.0, 1.0)
    ξτ ~ Dists.Uniform(-π/2, π/2)
    scx = 1/sqrt(1-τ)
    scy = sqrt(1-τ)



    #Total flux
    f ~ Dists.Uniform(0.5, 5.0)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)



    mring = renormed(ROSE.MRing(rad, (α1,α2,α3), (β1,β2,β3)), f-f*floor)
    disk = renormed(stretched(ROSE.Disk(), rad, rad), f*floor)
    img = smoothed(rotated(stretched(mring+disk,scx,scy),ξτ),σ)

    aAP ~ Dists.LogNormal(0.0, 0.1)
    aAZ ~ Dists.LogNormal(0.0, 0.1)
    aJC ~ Dists.LogNormal(0.0, 0.1)
    aSM ~ Dists.LogNormal(0.0, 0.1)
    aAA ~ Dists.LogNormal(0.0, 0.1)
    aLM ~ Dists.LogNormal(0.0, 0.2)
    aSP ~ Dists.LogNormal(0.0, 0.1)
    aPV ~ Dists.LogNormal(0.0, 0.1)
    ga = (AP=aAP, AZ=aAZ, JC=aJC, SM=aSM, AA=aAA, LM=aLM, SP=aSP, PV=aPV)

    pAP ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pAZ ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pJC ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pSM ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pAA ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pLM ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pSP ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pPV ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    gp = (AP=pAP, AZ=pAZ, JC=pJC, SM=pSM, AA=pAA, LM=pLM, SP=pSP, PV=pPV)


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




"""
    mring2wfVis
Returns a fiducial mring model that is fit to complex vis including
constant gain amps, and gain phases.
"""

mring2wfVis = @model u, v, s1, s2, err begin
    diam ~ Dists.Uniform(25.0, 75.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    #First mring mode
    ma1 ~ Dists.Uniform(0.0,0.5)
    mp1 ~ Dists.Uniform(-1π,1π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)

    #Second mring mode
    ma2 ~ Dists.Uniform(0.0,0.5)
    mp2 ~ Dists.Uniform(-1π,1π)
    α2 = ma2*cos(mp2)
    β2 = ma2*sin(mp2)


    #Total flux
    f ~ Dists.Uniform(0.5, 5.0)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)

    aAP ~ Dists.LogNormal(0.0, 0.1)
    aAZ ~ Dists.LogNormal(0.0, 0.1)
    aJC ~ Dists.LogNormal(0.0, 0.1)
    aSM ~ Dists.LogNormal(0.0, 0.1)
    aAA ~ Dists.LogNormal(0.0, 0.1)
    aLM ~ Dists.LogNormal(0.0, 0.2)
    aSP ~ Dists.LogNormal(0.0, 0.1)
    PV ~ Dists.LogNormal(0.0, 0.1)
    g = (AP=AP, AZ=AZ, JC=JC, SM=SM, AA=AA, LM=LM, SP=SP, PV=PV)

    pAP ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pAZ ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pJC ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pSM ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pAA ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pLM ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    pSP ~ Dists.truncated(Dists.Normal(0.0, π), -π, π)
    gp = (AP=pAP, AZ=pAZ, JC=pJC, SM=pSM, AA=pAA, LM=pLM, SP=pSP)


    mring = renormed(ROSE.MRing(rad, (α1,α2), (β1,β2)), f-f*floor)
    disk = renormed(stretched(ROSE.Disk(), rad, rad), f*floor)
    img = smoothed(mring+disk,σ)
    vis = visibility.(Ref(img), u, v)

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



mring2wfVA = @model u, v, s1, s2, err begin
    diam ~ Dists.Uniform(25.0, 75.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    #First mring mode
    ma1 ~ Dists.Uniform(0.0,0.5)
    mp1 ~ Dists.Uniform(-1π,1π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)

    #Second mring mode
    ma2 ~ Dists.Uniform(0.0,0.5)
    mp2 ~ Dists.Uniform(-1π,1π)
    α2 = ma2*cos(mp2)
    β2 = ma2*sin(mp2)


    #Total flux
    f ~ Dists.Uniform(0.5, 5.0)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)

    AP ~ Dists.LogNormal(0.0, 0.1)
    AZ ~ Dists.LogNormal(0.0, 0.1)
    JC ~ Dists.LogNormal(0.0, 0.1)
    SM ~ Dists.LogNormal(0.0, 0.1)
    AA ~ Dists.LogNormal(0.0, 0.1)
    LM ~ Dists.LogNormal(0.0, 0.2)
    SP ~ Dists.LogNormal(0.0, 0.1)
    PV ~ Dists.LogNormal(0.0, 0.1)
    g = (AP=AP, AZ=AZ, JC=JC, SM=SM, AA=AA, LM=LM, SP=SP, PV=PV)


    #Build the model
    mring = renormed(ROSE.MRing(rad, (α1,α2), (β1,β2)), f-f*floor)
    disk = renormed(stretched(ROSE.Disk(), rad, rad), f*floor)
    img = smoothed(mring+disk,σ)

    #Observe the model
    amp ~ For(eachindex(u,v, err)) do i
        g1 = g[s1[i]]
        g2 = g[s2[i]]
        mamp = g1*g2*ROSE.visibility_amplitude(img, u[i], v[i])
        Dists.Normal(mamp, err[i])
    end

end


gaussVA = @model u, v, s1, s2, err begin
    fwhm ~ Dists.Uniform(1.0, 50.0)
    f ~ Dists.Uniform(0.5, 5.0)
    σ = fwhm/SossModels.fwhmfac
    img = renormed(stretched(ROSE.Gaussian(), σ, σ), f)

    #Gain amps
    AP ~ Dists.LogNormal(0.0, 0.1)
    AZ ~ Dists.LogNormal(0.0, 0.1)
    JC ~ Dists.LogNormal(0.0, 0.1)
    SM ~ Dists.LogNormal(0.0, 0.1)
    AA ~ Dists.LogNormal(0.0, 0.1)
    LM ~ Dists.LogNormal(0.0, 0.2)
    SP ~ Dists.LogNormal(0.0, 0.1)
    PV ~ Dists.LogNormal(0.0, 0.1)
    g = (AP=AP, AZ=AZ, JC=JC, SM=SM, AA=AA, LM=LM, SP=SP, PV=PV)


    amp ~ For(eachindex(u,v, err)) do i
        g1 = g[s1[i]]
        g2 = g[s2[i]]
        mamp = g1*g2*ROSE.visibility_amplitude(img, u[i], v[i])
        Dists.Normal(mamp, err[i])
    end

end

end #SossModels
