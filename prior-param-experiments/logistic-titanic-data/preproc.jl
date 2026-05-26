using CSV, DataFrames

df = CSV.read("titanic3.csv", DataFrame)
df2 = df[:, ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare"]]
dropmissing!(df2)
CSV.write("titanic3-clean.csv", df2)