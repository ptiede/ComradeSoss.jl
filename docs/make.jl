using ROSESoss
using Documenter

DocMeta.setdocmeta!(ROSESoss, :DocTestSetup, :(using ROSESoss); recursive=true)

makedocs(;
    modules=[ROSESoss],
    authors="ptiede <ptiede91@gmail.com> and contributors",
    repo="https://github.com/ptiede/ROSESoss.jl/blob/{commit}{path}#{line}",
    sitename="ROSESoss.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ptiede.github.io/ROSESoss.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ptiede/ROSESoss.jl",
)
