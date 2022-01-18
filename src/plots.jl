using .Plots
using .StatsPlots
function plot_im(mean_img)
    p = plot()
    heatmap!(p,
              imagepixels(mean_img)...,
              mean_img,
              aspect_ratio=:equal,
              size=(500,400),
              xflip=true
            )
    xlims!(p,-80.0, 80.0)
    ylims!(p,-80.0, 80.0)
    title!(p,"Mean Image")
    return p
end



function plot_samples(sims)
    anim = @animate for s in sims[1:end]
        heatmap(imagepixels(s)..., s, aspect_ratio=:equal, size=(500,400), xflip=true)
        xlims!(-80.0,80.0)
        ylims!(-80.0, 80.0)
    end
    return anim
end

function plot_vis_comp(visobs, gamps, gphase, m)
    u = Comrade.getdata(visobs, :u)
    v = Comrade.getdata(visobs, :v)
    visr = Comrade.getdata(visobs, :visr)
    visi = Comrade.getdata(visobs, :visi)
    error = Comrade.getdata(visobs, :error)
    bl = Comrade.getdata(visobs, :baselines)

    mvis = Comrade.visibility.(Ref(m), u, v)
    for i in eachindex(u)
        s1,s2 = bl[i]
        ga1 = gamps[s1]
        ga2 = gamps[s2]
        gθ1 = gphase[s1]
        gθ2 = gphase[s2]
        s,c = sincos(gθ1 - gθ2)
        vr = ga1*ga2*(real(mvis[i])*c + imag(mvis[i])*s)
        vi = ga1*ga2*(-real(mvis[i])*s + imag(mvis[i])*c)
        mvis[i] = vr + vi*1im
    end


    p = plot()

    scatter!(p,hypot.(u,v)*rad2μas/1e9, visr,
                  yerr=error,
                  color=:cornflowerblue, label="Real Data", markershape=:square, alpha=0.5)
    scatter!(p,hypot.(u,v)*rad2μas/1e9, visi,
                  yerr=error,
                  color=:orange, label="Imag Data", markershape=:square, alpha=0.5)
    scatter!(p,hypot.(u,v)*rad2μas/1e9, real.(mvis),
                  color=:blue, label="Real Model")
    scatter!(p,hypot.(u,v)*rad2μas/1e9, imag.(mvis),
                  color=:red, label="Imag Model")
    ylabel!("Visibility")
    xlabel!("uvdist Gλ")
    return p

end


function plot_amp_comp(ampobs, gains, m)
    u = Comrade.getdata(ampobs, :u)
    v = Comrade.getdata(ampobs, :v)
    amp = Comrade.getdata(ampobs, :amp)
    error = Comrade.getdata(ampobs, :error)
    bl = Comrade.getdata(ampobs, :baselines)

    vamp = Comrade.amplitude.(Ref(m), u, v)
    for i in eachindex(u)
        s1,s2 = bl[i]
        g1 = gains[s1]
        g2 = gains[s2]
        vamp[i] =  g1*g2*vamp[i]
    end

    p = plot()

    scatter!(p,hypot.(u,v)*rad2μas/1e9, amp,
                  yerr=error,
                  color=:cornflowerblue, label="Data", markershape=:square, alpha=0.5)
    scatter!(p,hypot.(u,v)*rad2μas/1e9, vamp,
                  color=:blue, label="Model")
    ylabel!("V Amplitude")
    xlabel!("uvdist Gλ")
    return p

end

function plot_cp_comp(obscp, m)
    u1 = Comrade.getdata(obscp, :u1)
    v1 = Comrade.getdata(obscp, :v1)
    u2 = Comrade.getdata(obscp, :u2)
    v2 = Comrade.getdata(obscp, :v2)
    u3 = Comrade.getdata(obscp, :u3)
    v3 = Comrade.getdata(obscp, :v3)
    cp = Comrade.getdata(obscp, :phase)
    error = Comrade.getdata(obscp, :error)

    mcp = Comrade.closure_phase.(Ref(m),
                               u1, v1,
                               u2, v2,
                               u3, v3
                            )

    p = plot()

    tridist = @. sqrt(u1^2 + v1^2 + u2^2 + v2^2 + u3^2 + v3^2)*rad2μas/1e9
    scatter!(p, tridist, cp,
                  yerr=error,
                  color=:cornflowerblue, label="Data", markershape=:square, alpha=0.5)
    scatter!(p,tridist, mcp,
                  color=:blue, label="Model")
    ylabel!("Closure Phase")
    xlabel!("tridist Gλ")
    return p

end


function plot_res_density(anres, cpnres)
    p = plot()
    density!(p,anres, label="Amplitude")
    density!(p,cpnres, label="CP")
    plot!(p,x->pdf(Normal(), x), label="Std. Normal")
    xlabel!(p,"Normalized Residuals")
    return p
end
