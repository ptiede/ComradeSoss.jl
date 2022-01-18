using ComradeSoss
using Documenter

DocMeta.setdocmeta!(ComradeSoss, :DocTestSetup, :(using ComradeSoss); recursive=true)

makedocs(;
    modules=[ComradeSoss],
    authors="ptiede <ptiede91@gmail.com> and contributors",
    repo="https://github.com/ptiede/ComradeSoss.jl/blob/{commit}{path}#{line}",
    sitename="ComradeSoss.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ptiede.github.io/ComradeSoss.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ptiede/ComradeSoss.jl",
)
