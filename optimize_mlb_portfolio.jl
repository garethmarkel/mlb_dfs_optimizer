using LinearAlgebra, DataFrames, CSV, Distributions,SparseArrays, ArgParse, DelimitedFiles

function parse_param()

    s = ArgParseSettings()

    @add_arg_table s begin
        "--g"
            arg_type = Int
            default = 6
            help = "overlap between lineups"
        "--n"
            arg_type = Int
            default = 150
            help = "Number of lineups to be produced"
        "--n_scenarios"
            arg_type = Int
            default = 10000
            help = "Number of poit scenarios to simulate"
        "--plim"
            arg_type = Int
            default = 1
            help = "an option with an argument"
        "--cormat_path"
            arg_type = String
            required = true
            help = "Path to the correlation matrix for a given DFS contest"
        "--player_path"
            arg_type = String
            required = true
            help = "Path to the csv with player forecasts for a given contest"
        "--team_path"
            arg_type = String
            required = true
            help = "Path to the csv with sampled opponent lineups for a given DFS contest"
        "--payout_path"
            arg_type = String
            required = true
            help = "Path to a csv with payouts and order statistics associated for a given contest, columns 'ranks' and 'payouts'"
        "--output_dir"
            arg_type = String
            required = true
            help = "Output file path"
    end

    return parse_args(s)

end

parsed_args = parse_param()

dt = parsed_args["d"]


mat_cor = Matrix(DataFrame(CSV.File(parsed_args["cormat_path"]; types=(i, name) -> Float64)))
df_forecast = DataFrame(CSV.File(parsed_args["player_path"))
mat_lineups = Matrix(DataFrame(CSV.File(parsed_args["team_path")))
df_payouts = DataFrame(CSV.File(parsed_args["payout_path"))

n_mat = size(mat_cor)[1] + 1
n_tot = size(df_forecast)[1]

nd = parsed_args["n_scenarios"]

# simulate points δ
C = cholesky(mat_cor)
u = hcat([rand(Normal(0,1), size(mat_cor)[1]) for i in 1:nd]...)
δ = C.L*u

δ = δ .+ df_forecast[:,:proj_points]

# cap points at 0
δ[δ.<0.0] .= 0.0

Σ = mat_cor



vec_ord_stats = df_payouts[:,:ranks]
vec_ord_pay = df_payouts[:,:payout]


# Get the order statistics of opponent scoring

mat_ord = Array{Float64}(undef, size(vec_ord_stats)[2], size(δ)[2])

Threads.@threads for i in 1:size(δ)[2]
    tmp = mat_lineups' * δ[:,i]
    tmp = sort(tmp, rev = true)
    for j in 1:size(vec_ord_stats)[2]
        mat_ord[j,i] = tmp[vec_ord_stats[j]]
    end
end

# find the correlation, mean, and variance of all the order statistics 

mat_ord_cor = Array{Float64}(undef, n_tot, size(vec_ord_stats)[2])
mat_ord_mean = Array{Float64}(undef, 1, size(vec_ord_stats)[2])
mat_ord_var = Array{Float64}(undef, 1, size(vec_ord_stats)[2])

for i in 1:size(vec_ord_stats)[2]
    for j in 1:n_tot
        mat_ord_cor[j,i] = cov(δ[j,:], mat_ord[i,:])
    end
    mat_ord_mean[1,i] = mean(mat_ord[i,:])
    mat_ord_var[1,i] = var(mat_ord[i,:])
end

mat_ord_cor_approx = mean(mat_ord_cor, dims = 2)

# free up some memory
mat_ord = 0.0


using JuMP, Gurobi

n_lineups = parsed_args["n"]
γ = parsed_args["g"]
s = parsed_args["s"]


# empty array to fill with lineups
lines = Array{Float64}(undef, n_tot, n_lineups)
lines .= 0.0

# frid of hyperparameters
Λ = collect(0:20)/100
A = Array(Diagonal(ones(n_tot)))

# vector to constrain optimization
B = ones(n_lineups) .* γ

# categorize player positions

vec_1b = ifelse.(contains.("1B",df_forecast[:,:pos]) .|| contains.("C",df_forecast[:,:pos]),1.0,0.0)
vec_2b = ifelse.(contains.("2B",df_forecast[:,:pos]) ,1.0,0.0)
vec_3b = ifelse.(contains.("3B",df_forecast[:,:pos]) ,1.0,0.0)
vec_SS = ifelse.(contains.("SS",df_forecast[:,:pos]) ,1.0,0.0)
vec_OF = ifelse.(contains.("OF",df_forecast[:,:pos]) ,1.0,0.0)
vec_p = ifelse.(contains.("P",df_forecast[:,:pos]) ,1.0,0.0)
vec_util = ifelse.(df_forecast[:,:pos] .!= "P",1.0,0.0)

# categorize teams. we need to create reference vectors to 
# implement team constraints (no more than 4 players from a team)

u_teams = unique(df_forecast[:,:team])
mat_tm = zeros(n_tot,size(u_teams)[1])
mat_tm2 = zeros(n_tot,size(u_teams)[1])

mat_lmda = zeros(n_lineups)

mat_l = zeros(n_tot,size(u_teams)[1])

# create vector to indicate players who appear in the batting order

batting_ord = 1.0 .- Float64.(df_forecast[:,:ord] .=== missing)

# create vector to indicate players who appear at the top fo the batting order
# not used, but could make additional conttraint

big_hitters = [ifelse(x === missing,8,x) for x in df_forecast[:,:ord]]
big_hitters = [ifelse(j > 6, 0, 1) for j in big_hitters]

n_teams = length(u_teams)
for i in 1:n_teams
    mat_tm[df_forecast[:,:team] .== u_teams[i], i] .= 1.0
    mat_tm2[df_forecast[:,:team] .== u_teams[i], i] .= 1.0
    
    #pitchers don't count towards total
    mat_tm[df_forecast[:,:pos] .== "P", i] .= 0.0

end


# now, make each lineup
for k in 1:n_lineups

    # matrix for candidate lineups for each run
    sub_lines = Array{Float64}(undef, n_tot, length(Λ))
    sub_lines .= 0.0

    for i in 1:length(Λ)

        λ = Λ[i]
        
        tst = df_forecast[:,:proj_points] - 2*λ*mat_ord_cor_approx[:,1]
        
        model = Model(Gurobi.Optimizer)
        set_silent(model)

        @variable(model, 1 >= x[1:n_tot] >= 0, Int)

        # lineup position constriants
        @constraint(model, sum(x) == 9)
        @constraint(model, x' * vec_1b <= 2)
        @constraint(model, x' * vec_2b <= 2)
        @constraint(model, x' * vec_3b <= 2)
        @constraint(model, x' * vec_SS <= 2)
        @constraint(model, x' * vec_OF == 3)
        @constraint(model, x' * vec_p == 1)

        @constraint(model, x' * vec_1b >= 1)
        @constraint(model, x' * vec_2b >= 1)
        @constraint(model, x' * vec_3b >= 1)
        @constraint(model, x' * vec_SS >= 1)

        # overlap constraint
        @constraint(model, lines' * x .<= B)

        # team cosntraint and salary constraint
        @constraint(model, mat_tm' * x .<= 4)
        @constraint(model, df_forecast[:,:salary]'*x <= 35000)

        # at least 3 teams have to appear
        @variable(model, z[1:n_teams], Bin)
        @constraint(model,cc[d=1:n_teams], z[d] <= mat_tm2[:,d]' * x)
        @constraint(model, sum(z) >= 3)

        # objective function
        @objective(model, Max, x' * tst + λ* x' * Σ * x)

        # optimize model
        optimize!(model)
        sub_lines[:,i] = value.(x)

    end

    μ_y = Array{Float64}(undef, length(Λ),size(vec_ord_stats)[2])
    σ_y = Array{Float64}(undef, length(Λ),size(vec_ord_stats)[2])

    # which proposed lineup maximizes overall objective function
    for i in 1:size(vec_ord_stats)[2]

        μ_y[:,i] = sub_lines' * df_forecast[:,:proj_points] .- mat_ord_mean[i]
        σ_y[:,i] = diag(sub_lines' * Σ * sub_lines) .- 2 * sub_lines' * mat_ord_cor[:,i] .+ mat_ord_var[i]

    end

    σ_y = sqrt.(σ_y)

    payoffs = zeros(length(Λ))

    for i in 1:length(Λ)

        R = 0

        for j in 1:size(vec_ord_stats)[2]

            R = R + (vec_ord_pay[j] - vec_ord_pay[j+1])*(1 - cdf(Normal(0,1), - μ_y[i,j]/σ_y[i,j]))

        end
        payoffs[i] = R

    end

    # select lineup with max payoff
    indmax = findfirst(x -> x == maximum(payoffs),payoffs)

    lines[:,k] = sub_lines[:,indmax]

    # store lambdas for reference
    mat_lmda[k] = Λ[indmax]

end

out = parsed_args["output_dir"]

writedlm("$out/lineups-$γ-$n_lineups.csv", lines, ",")
writedlm("$out/lambda.csv", mat_lmda, ",")
