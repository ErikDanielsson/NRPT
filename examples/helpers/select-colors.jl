using Colors

# We assume that the optimizer is written as '<name>-<lr>'
# or '<name>' for learning rate free methods
function parse_optimizer(key::String)
    if count("-", key) == 2
        _, opt, lr = split(key, "-", limit=3)
        return opt, parse(Float64, lr)
    else
        println(key)
        _, opt = split(key, "-", limit=2)
        return opt, nothing
    end
end

function shade_by_lr(base_color::Colorant, lr, lr_min, lr_max)
    hsl = HSL(base_color)

    # Normalize learning rate to [0, 1]
    t = (log10(lr) - log10(lr_min)) / (log10(lr_max) - log10(lr_min))
    t = clamp(t, 0, 1)

    # Lightness range: lighter → darker
    new_lightness = 0.85 - 0.45 * t

    return HSL(hsl.h, hsl.s, new_lightness)
end

function opt_lrs(opt_keys)
    lrs_by_opt = Dict{String, Vector{Float64}}()
    for k in opt_keys
        opt, lr = parse_optimizer(k)
        if lr !== nothing
            push!(get!(lrs_by_opt, opt, Float64[]), lr)
        end
    end
    return lrs_by_opt
end

function color_per_opt(opts, base_colors)
    opt_keys = keys(opts)
    colors = Dict{String, Colorant}()
    lrs_by_opt = opt_lrs(opt_keys)

    for k in opt_keys
        opt, lr = parse_optimizer(k)

        base = base_colors[opt]

        if lr === nothing
            colors[k] = base
        else
            lr_min = minimum(lrs_by_opt[opt])
            lr_max = maximum(lrs_by_opt[opt])
            colors[k] = shade_by_lr(base, lr, lr_min, lr_max)
        end
    end
    return colors
end
