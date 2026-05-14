I want you to help me refactor the code in nrpt.jl: 
- I want to move the recording of values from raw arrays, like we already do with samples and the index process
- Furthermore, I want you to set up the split between the round for schedule adaptation and optimization more cleanly.
  - This having separate arrays for the round counts for the two different round types. We want for example to run a few rounds of 
    schedule adaptation before we start optimizing and should thus replace the min_ess requirement with "start round optimization" threshould
    that is a specific number of rounds of only schedule optimization. 
  - We want to record the following jointly
    - Index process
    - Samples
  - and the following disjointly
    - Objective values (like we do in the LossRecorder.jl file)
    - Normalization constant estimators
    - Schedules (this is constant for the optimization round, but we want to bind this more explicitly to a recorder in the adaptation round)
Please explain to me what this entails, and check if you can come up with other ways of making the optimized_nrpt code cleaner! Please ask questions
to make sure I know that you have understood the assignment 
