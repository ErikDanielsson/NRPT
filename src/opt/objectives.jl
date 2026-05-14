abstract type PathObjective end

struct SKLObjective <: PathObjective end
struct BarrierObjective <: PathObjective end

objective_loss(::SKLObjective, problem, ptchains, threaded) =
    SKL_loss(problem, ptchains, Val(threaded))

objective_gradient(::SKLObjective, problem, ptchains, threaded) =
    SKL_gradient(problem, ptchains, Val(threaded))

objective_loss(::BarrierObjective, problem, ptchains) =
    barrier_loss(problem, ptchains, schedule)

objective_gradient(::BarrierObjective, problem, ptchains) =
    barrier_gradient(problem, ptchains, schedule)
