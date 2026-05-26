using NRPT

function make_opt_state(name, lr)
    if name == "SGD"
        return SGDState, (lr)
    elseif name == "DoG"
        return DoGState, (1e-6, 1e-6)
    elseif name == "DoWG"
        return DoWGState, (1e-6, 1e-6)
    elseif name == "Adam"
        return AdamState, (lr, 1e-6)
    elseif name == "AdaGrad"
        return AdagradState, (lr, 1e-6)
    elseif name == "ScaledAdaGrad"
        return ScaledAdagradState, (lr, 1e-6, x -> exp(x))
    elseif name == "Newton"
        return NewtonTrustRegionState, (AutoForwardDiff(),)
    elseif name == "no_opt"
        return NoOptState, ()
    else
        error("$name is not an optimizer")
    end
end

function make_optimizers(config; targets=Dict(["SKL" => SKLObjective]))
    optimizers = Dict()
    for (oname, obj) in targets
        for (name, maybeLrs) in config
            if maybeLrs === nothing
                cons, args = make_opt_state(name, nothing)
                optimizers["$oname-$name"] = (obj(), cons(args...))
            else
                for lr in maybeLrs
                    cons, args = make_opt_state(name, lr)
                    optimizers["$oname-$name-$lr"] = (obj(), cons(args...))
                end
            end
        end
    end
    return optimizers
end