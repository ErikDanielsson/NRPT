abstract type PathObjective end

struct SKLObjective <: PathObjective end
struct BarrierObjective <: PathObjective end

objective_loss(::SKLObjective, problem, ptchains, schedule) =
    SKL_loss(problem, ptchains, schedule)

objective_gradient(::SKLObjective, problem, ptchains, schedule) =
    SKL_gradient(problem, ptchains, schedule)

objective_loss(::BarrierObjective, problem, ptchains, schedule) =
    barrier_loss(problem, ptchains, schedule)

objective_gradient(::BarrierObjective, problem, ptchains, schedule) =
    barrier_gradient(problem, ptchains, schedule)
