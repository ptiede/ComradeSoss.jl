const fwhmfac = 2*sqrt(2*log(2))



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
    mp1 ~ Dists.Uniform(0.0, 2π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)

    #Second mring mode
    ma2 ~ Dists.Uniform(0.0,0.5)
    mp2 ~ Dists.Uniform(0.0, 2π)
    α2 = ma2*cos(mp2)
    β2 = ma2*sin(mp2)


    #Total flux
    f ~ Dists.Uniform(0.8, 1.0)

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
    g = (AP=AP, AZ=AZ, JC=JC, SM=SM, AA=AA, LM=LM, SP=SP)


    mring = ROSE.MRing(rad, (α1,α2), (β1,β2))
    disk = renormed(stretched(ROSE.Disk(), rad, rad), floor)
    img = renormed(smoothed(mring+disk,σ),f)

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
    mp1 ~ Dists.Uniform(0.0, 2π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)

    #Second mring mode
    ma2 ~ Dists.Uniform(0.0,0.5)
    mp2 ~ Dists.Uniform(0.0, 2π)
    α2 = ma2*cos(mp2)
    β2 = ma2*sin(mp2)

    τ ~ Dists.truncated(Dists.Normal(0.0, 0.2), 0.0, 1.0)
    ξτ ~ Dists.Uniform(-π/2, π/2)
    scx = 1/sqrt(1-τ)
    scy = sqrt(1-τ)



    #Total flux
    f ~ Dists.Uniform(0.8, 1.0)

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
    g = (AP=AP, AZ=AZ, JC=JC, SM=SM, AA=AA, LM=LM, SP=SP)


    mring = ROSE.MRing(rad, (α1,α2), (β1,β2))
    disk = renormed(stretched(ROSE.Disk(), rad, rad), floor)
    img = renormed(smoothed(rotated(stretched(mring+disk,scx,scy),ξτ),σ),f)

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



mring2wfVis = @model u, v, s1, s2, err begin
    diam ~ Dists.Uniform(25.0, 75.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    #First mring mode
    ma1 ~ Dists.Uniform(0.0,0.5)
    mp1 ~ Dists.Uniform(0.0, 2π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)

    #Second mring mode
    ma2 ~ Dists.Uniform(0.0,0.5)
    mp2 ~ Dists.Uniform(0.0, 2π)
    α2 = ma2*cos(mp2)
    β2 = ma2*sin(mp2)


    #Total flux
    f ~ Dists.Uniform(0.8, 1.0)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)

    aAP ~ LogNormal(0.0, 0.1)
    aAZ ~ LogNormal(0.0, 0.1)
    aJC ~ LogNormal(0.0, 0.1)
    aSM ~ LogNormal(0.0, 0.1)
    aAA ~ LogNormal(0.0, 0.1)
    aLM ~ LogNormal(0.0, 0.2)
    aSP ~ LogNormal(0.0, 0.1)
    ga = (AP=aAP, AZ=aAZ, JC=aJC, SM=aSM, AA=aAA, LM=aLM, SP=aSP)

    pAP ~ Dists.truncated(Normal(0.0, π), -π, π)
    pAZ ~ Dists.truncated(Normal(0.0, π), -π, π)
    pJC ~ Dists.truncated(Normal(0.0, π), -π, π)
    pSM ~ Dists.truncated(Normal(0.0, π), -π, π)
    pAA ~ Dists.truncated(Normal(0.0, π), -π, π)
    pLM ~ Dists.truncated(Normal(0.0, π), -π, π)
    pSP ~ Dists.truncated(Normal(0.0, π), -π, π)
    gp = (AP=pAP, AZ=pAZ, JC=pJC, SM=pSM, AA=pAA, LM=pLM, SP=pSP)


    mring = ROSE.MRing(rad, (α1,α2), (β1,β2))
    disk = renormed(stretched(ROSE.Disk(), rad, rad), floor)
    img = renormed(smoothed(mring+disk,σ),f)
    vis = visibility.(Ref(img), u, v)

    visr ~ For(eachindex(u, v)) do i
        Δθ = gp[s1[i]] - gp[s2[i]]
        s,c = sincos(Δθ)
        g1 = ga[s1[i]]
        g2 = ga[s2[i]]
        mamp = g1*g2*(real(vis[i])*c + imag(vis[i])*s)
        Normal(mamp, err[i])
    end

    visi ~ For(eachindex(u, v)) do i
        Δθ = gp[s1[i]] - gp[s2[i]]
        s,c = sincos(Δθ)
        g1 = ga[s1[i]]
        g2 = ga[s2[i]]
        mamp = g1*g2*(-real(vis[i])*s + imag(vis[i])*c)
        Normal(mamp, err[i])
    end
end

test = @model begin
    diam ~ Dists.Uniform(25.0, 75.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    #First mring mode
    ma1 ~ Dists.Uniform(0.0,0.5)
    mp1 ~ Dists.Uniform(0.0, 2π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)

    #Second mring mode
    ma2 ~ Dists.Uniform(0.0,0.5)
    mp2 ~ Dists.Uniform(0.0, 2π)
    α2 = ma2*cos(mp2)
    β2 = ma2*sin(mp2)


    #Total flux
    f ~ Dists.Uniform(0.8, 1.0)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)


    mring = ROSE.MRing(rad, (α1,α2), (β1,β2))
    disk = renormed(stretched(ROSE.Disk(), rad, rad), floor)
    img = renormed(smoothed(mring+disk,σ),f)
    return img
end


mring2wfVA = @model u, v, s1, s2, err begin
    diam ~ Dists.Uniform(25.0, 75.0)
    fwhm ~ Dists.Uniform(1.0, 40.0)
    rad = diam/2
    σ = fwhm/fwhmfac

    #First mring mode
    ma1 ~ Dists.Uniform(0.0,0.5)
    mp1 ~ Dists.Uniform(0.0, 2π)
    α1 = ma1*cos(mp1)
    β1 = ma1*sin(mp1)

    #Second mring mode
    ma2 ~ Dists.Uniform(0.0,0.5)
    mp2 ~ Dists.Uniform(0.0, 2π)
    α2 = ma2*cos(mp2)
    β2 = ma2*sin(mp2)


    #Total flux
    f ~ Dists.Uniform(0.8, 1.0)

    #Fraction of floor flux
    floor ~ Dists.Uniform(0.0, 1.0)

    AP ~ LogNormal(0.0, 0.1)
    AZ ~ LogNormal(0.0, 0.1)
    JC ~ LogNormal(0.0, 0.1)
    SM ~ LogNormal(0.0, 0.1)
    AA ~ LogNormal(0.0, 0.1)
    LM ~ LogNormal(0.0, 0.2)
    SP ~ LogNormal(0.0, 0.1)
    ga = (AP=aAP, AZ=aAZ, JC=aJC, SM=aSM, AA=aAA, LM=aLM, SP=aSP)



    mring = ROSE.MRing(rad, (α1,α2), (β1,β2))
    disk = renormed(stretched(ROSE.Disk(), rad, rad), floor)
    img = renormed(smoothed(mring+disk,σ),f)
    amp ~ For(eachindex(uamp,vamp, erramp)) do i
        g1 = g[s1[i]]
        g2 = g[s2[i]]
        mamp = g1*g2*ROSE.visibility_amplitude(img, uamp[i], vamp[i])
        Dists.Normal(mamp, erramp[i])
    end

end
