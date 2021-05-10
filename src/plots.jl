using .Plots
using .StatsPlots
function plot_mean(sims)
    p = plot()
    mean_img = mean(sims)
    heatmap!(p,
              imagepixels(sims[1])...,
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

function plot_vis_comp(visobs, m)
    u = ROSE.getdata(visobs, :u)
    v = ROSE.getdata(visobs, :v)
    visr = ROSE.getdata(visobs, :visr)
    visi = ROSE.getdata(visobs, :visi)
    error = ROSE.getdata(visobs, :error)
    bl = ROSE.getdata(visobs, :baselines)

    vis = ROSE.visibility_amplitude.(Ref(m), u, v)

    p = plot()

    scatter!(p,hypot.(u,v)*rad2μas/1e9, visr,
                  yerr=error,
                  color=:cornflowerblue, label="Real Data", markershape=:square, alpha=0.5)
    scatter!(p,hypot.(u,v)*rad2μas/1e9, visi,
                  yerr=error,
                  color=:orange, label="Imag Data", markershape=:square, alpha=0.5)
    scatter!(p,hypot.(u,v)*rad2μas/1e9, real.(vis),
                  color=:blue, label="Real Model")
    scatter!(p,hypot.(u,v)*rad2μas/1e9, imag.(vis),
                  color=:red, label="Imag Model")
    ylabel!("Visibility")
    xlabel!("uvdist Gλ")
    return p

end


function plot_amp_comp(ampobs, gains, m)
    u = ROSE.getdata(ampobs, :u)
    v = ROSE.getdata(ampobs, :v)
    amp = ROSE.getdata(ampobs, :amp)
    error = ROSE.getdata(ampobs, :error)
    bl = ROSE.getdata(ampobs, :baselines)

    vamp = ROSE.visibility_amplitude.(Ref(m), u, v)
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
    u1 = ROSE.getdata(obscp, :u1)
    v1 = ROSE.getdata(obscp, :v1)
    u2 = ROSE.getdata(obscp, :u2)
    v2 = ROSE.getdata(obscp, :v2)
    u3 = ROSE.getdata(obscp, :u3)
    v3 = ROSE.getdata(obscp, :v3)
    cp = ROSE.getdata(obscp, :phase)
    error = ROSE.getdata(obscp, :error)

    mcp = ROSE.closure_phase.(Ref(m),
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
