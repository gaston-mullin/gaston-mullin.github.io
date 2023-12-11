# Load the relevant packages
using StatsKit, ForwardDiff, Ipopt, NLsolve, Optim, Parameters, Zygote, LinearAlgebra, Random, Plots, BenchmarkTools, StatsBase

# Define the values for the x distribution
x_min=0.0;
x_max=15.0;
x_int=0.05;
x_len=Int64(1+(x_max-x_min)/x_int);
X=range(x_min,x_max,x_len);

# Transition matrix for mileage:
F1=zeros(x_len,x_len);
F1[:,1].=1.0;

x_tday      = repeat(X, 1, x_len); 
x_next      = x_tday';
x_zero      = zeros((x_len, x_len));

F2 = (x_next.>=x_tday) .* exp.(-(x_next - x_tday)) .* (1 .- exp(-x_int));
F2[:,end]=1 .-sum(F2[:,1:(end-1)],dims=2);
F2_cumul=cumsum(F2,dims=2);

F2

# Rows add up to 1
sum(F2,dims=2)

# There are two permanent characteristics s, denoting bus brand.
S=[1, 2];
s_len=Int64(length(S));

"""
This function pefrorms value iteration, which is needed to solve the model.
"""
function value_function_iteration(x::AbstractRange{Float64},S::Vector{Int64},F1::Matrix{Float64},F2::Matrix{Float64},β::Number,θ::Vector;MaxIter=1000)
    x_len=length(x);
    γ=Base.MathConstants.eulergamma;
    value_function2=zeros(x_len,length(S));
    value_diff=1.0;
    tol=1e-5;
    iter=1;
    local v1, v2
    while (value_diff>tol) && (iter<=MaxIter)
        value_function1=value_function2;
        v1=[0.0 + β*F1[j,:]'*value_function1[:,s] for j∈eachindex(X), s∈eachindex(S)];
        v2=[θ[1]+θ[2]*X[j]+θ[3]*S[s] + β*(F2[j,:]'*value_function1[:,s]) for j=1:x_len, s∈eachindex(S)];
        value_function2=[log(exp(v1[j,s])+exp(v2[j,s]))+γ for j=1:x_len, s=1:length(S)];
        iter=iter+1;
        value_diff=maximum((value_function1 .- value_function2).^2);
    end
    ccps=[1/(1+exp(v2[j,s,]-v1[j,s])) for j=1:x_len, s=1:length(S)];
    return (ccps_true=ccps, value_function=value_function2)
end

"""
This function generates data. 
"""
function generate_data(N,T,X,S,F1,F2,F_cumul,β,θ;T_init=10,π=0.4,ex_initial=0)
    if ex_initial==1
        T_init=0;
    end
    x_data=zeros(N,T+T_init);
    x_data_index=Array{Int32}(ones(N,T+T_init));
    if ex_initial==1
        x_data_index[:,1]=rand(1:length(X),N,1);
        x_data[:,1]=X[x_data_index[:,1]];
    end
    s_data=(rand(N) .> π) .+ 1;
    d_data=zeros(N,T+T_init);

    draw_ccp=rand(N,T+T_init);
    draw_x=rand(N,T+T_init);

    (ccps,_)=value_function_iteration(X,S,F1,F2,β,θ);

    for n=1:N
        for t=1:T+T_init
            d_data[n,t]=(draw_ccp[n,t] > ccps[x_data_index[n,t],s_data[n]])+1;
            if t<T+T_init
                x_data_index[n,t+1]=1 + (d_data[n,t]==2)*sum(draw_x[n,t] .> F_cumul[x_data_index[n,t],:]); 
                x_data[n,t+1]=X[x_data_index[n,t+1]];
            end
        end
    end

    return (XData=x_data[:,T_init+1:T+T_init], SData=repeat(s_data,1,T),
        DData=d_data[:,T_init+1:T+T_init],
        XIndexData=x_data_index[:,T_init+1:T_init+T],
        TData=repeat(1:T,N,1),
        NData=repeat((1:N)',1,T)) 
end


# Set the number of buses, time periods and value of parameters
N=2000;
T=15;
θ=[2.0, -0.15, 1.0];
β=0.9;
π=0.4;

Random.seed!(3000);
XData, SData, DData, XIndexData, TData, NData = generate_data(N,T,X,S,F1,F2,F2_cumul,β,θ);

# Showing the mileage of one bus.
plot(XData[1:2,:]',ylabel="Mileage",xlabel="Time",label = ["Bus 1" "Bus 2"])
ylims!(0, 15)

# Showing the frequency of engine replacement in the data.
sort(countmap(sum(DData.==1,dims=2)))

ccp_hat_cell = [mean(vec(DData)[(vec(XIndexData).==j) .& (vec(SData).==s)].==1) for j∈eachindex(X), s∈eachindex(S)];
epsilon=1e-4;

#ccp_hat_cell = zeros(length(X),length(S));
#
#for j in 1:length(X)
#    for s in 1:length(S)
#        ccp_hat_cell[j,s] = mean(vec(DData)[(vec(XIndexData).==j) .& (vec(SData).==s)].==1);
#    end
#end

never_reached=isnan.(ccp_hat_cell); # Keep track of states that never occur in the data
ccp_hat_cell[never_reached] .= 0.5; # Provide a value for the never reached states. 0.5 may not correspond well to what the data sugests...
ccp_hat_cell=max.(epsilon, min.(1-epsilon,ccp_hat_cell)); # Makes sure that all probabilities are in the open interval (0,1), as "1" and "0" observations cause problems.

plot(X[never_reached[:,1].==0],ccp_hat_cell[never_reached[:,1].==0,1],xlabel="Mileage (x)",ylabel="Conditional probability, s=1",label = "Prob")


# Estimate the ccps in a flexible parametric logit
X_logit=[ones(length(XData)) vec(XData) vec(XData).^2 vec(XData).^3/10 vec(SData).==2 (vec(SData).==2).*vec(XData) (vec(SData).==2).*(vec(XData).^2)];
logit_lik(b) = -sum((vec(DData).==1).*(X_logit*b) .- log.(1 .+ exp.(X_logit*b)))
x_0= 0.1.*ones(size(X_logit,2),1);
result=optimize(logit_lik,x_0, LBFGS(),Optim.Options(g_tol = 1e-6); autodiff = :forward)
logit_params_ccp = result.minimizer;
logit_params_ccp


# Fit the ccps from previous regression
xb_logit = zeros(length(X),2);

xb_logit[:,1] = logit_params_ccp[1] .+ logit_params_ccp[2]*X + logit_params_ccp[3]*(X.^2) + logit_params_ccp[4]*(X.^3)/10
xb_logit[:,2] = logit_params_ccp[5] .+ xb_logit[:,1] + logit_params_ccp[6]*X + logit_params_ccp[7]*(X.^2);

ccp_hat_logit = exp.(xb_logit)./(1 .+ exp.(xb_logit));

plot(X,ccp_hat_logit[:,1],xlabel="Mileage (x)",ylabel="Conditional probability, s=1",label = "Prob")


# Get all elements to the correct dimensions

# Stack CCPs in a Vector
ccp_hat_vec = vec(ccp_hat_logit);

#In case you want to check the order of the stack, compare the second element of the vector against the two possible candidates to see how it is working
println(string("Second element of the stacked vector: ",ccp_hat_vec[2]));
println(string("Row 2, column 1: ",ccp_hat_logit[2,1]));
println(string("Row 1, column 2: ",ccp_hat_logit[1,2]));
# Easy to see, elements 1:length(X) are from column 1, elements (length(x)+1):(length(x)+length(x)) are from column 2

# F1 and F2 into square matrices of size X.S
Z_len = length(X)*length(S);
F1_all = vcat(hcat(F1,zeros(size(F1))),hcat(zeros(size(F1)),F1));
F2_all = vcat(hcat(F2,zeros(size(F2))),hcat(zeros(size(F2)),F2));


γ=Base.MathConstants.eulergamma;

psi1 = γ .- log.(ccp_hat_vec);
psi2 = γ .- log.(1 .- ccp_hat_vec);

u2_hat = psi1 - psi2 + β*(F1_all - F2_all)*inv(I(Z_len) - β*F1_all)*psi1;

all_data = hcat(ones(Z_len,1), repeat(X,outer=length(S)), repeat(S,inner=length(X)));

min_distance(b) = sum((all_data*b - u2_hat)'*I(Z_len)*(all_data*b - u2_hat));
x_0= 0.1.*ones(size(all_data,2),1);
result_min_dist=optimize(min_distance,x_0, LBFGS(),Optim.Options(g_tol = 1e-6); autodiff = :forward)
lambda_hat_min_dist = result_min_dist.minimizer;
print(lambda_hat_min_dist);

# Alternatively:

lambda_hat_min_dist2 = inv(all_data'*all_data)*all_data'*u2_hat;
print(lambda_hat_min_dist2)

function ccp_likelihood_renewal(theta,all_data,F1_all,F2_all,data_x_index,data_s,data_d,ccp_hat_vec)
    γ=Base.MathConstants.eulergamma;

    u2 = theta[1] .+ theta[2]*all_data[:,2] + theta[3]*all_data[:,3];
    
    diff_v = u2  + theta[4]*(F1_all*log.(ccp_hat_vec) - F2_all*log.(ccp_hat_vec));

    data_z_index = vec(XIndexData) + (vec(SData) .- 1)*length(X);
    ccp_lik = -sum((vec(data_d).==2).*(diff_v[data_z_index]) - log.(1 .+ exp.(diff_v[data_z_index])));

end


min_renewal(theta) = ccp_likelihood_renewal(theta,all_data,F1_all,F2_all,XIndexData,SData,DData,ccp_hat_vec)
result_renewal = optimize(min_renewal,[0.1;0.1;0.1;0.1], LBFGS(),Optim.Options(g_tol = 1e-10); autodiff = :forward);
theta_renewal = result_renewal.minimizer


function fiml_likelihood(θ,DData,All_States_Index,X,S,F1,F2,β)
    _,value_function = value_function_iteration(X,S,F1,F2,β,θ)
    v1=[0.0 + β*F1[j,:]'*value_function[:,s] for j∈eachindex(X), s∈eachindex(S)];
    v2=[θ[1]+θ[2]*X[j]+θ[3]*S[s] + β*(F2[j,:]'*value_function[:,s]) for j∈eachindex(X), s∈eachindex(S)];

    return -sum((DData[j] == 2.0) * (v2[All_States_Index[j]] - v1[All_States_Index[j]] ) - log(1 + exp(v2[All_States_Index[j]]-v1[All_States_Index[j]])) for j∈eachindex(All_States_Index))

end

function fiml_estimation(DData::Matrix,XData::Matrix,SData::Matrix{Int64},X,S,F1,F2,β)
    x_len = length(X);
    All_States = [X 1.0*ones(x_len,1); X 2.0*ones(x_len,1)];
    All_States_Index=[findfirst(all(All_States .== [XData[j] SData[j]], dims=2)[:,1]) for j∈eachindex(XData)];
    f(θ) = fiml_likelihood(θ, DData,All_States_Index,X,S,F1,F2,β)
    result = optimize(f,[0.1;0.1;0.1], LBFGS(),Optim.Options(g_tol = 1e-6); autodiff = :forward);
    #result = optimize(f,[0.1;0.1;0.1]);
    return result.minimizer
end

theta_hat_fiml = fiml_estimation(DData,XData,SData,X,S,F1,F2,β);
println(theta_hat_fiml)


