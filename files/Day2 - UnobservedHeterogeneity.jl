# Load the relevant packages
using StatsKit, ForwardDiff, Ipopt, NLsolve, Optim, Parameters, Zygote, LinearAlgebra, Random, Plots, BenchmarkTools, StatsBase, Distributions

# Uncomment following lines to install one or several packages that are missing

#using Pkg
#Pkg.add(["StatsKit", "ForwardDiff", "Ipopt", "NLsolve", "Optim", "Parameters", "Zygote", "LinearAlgebra", "Random", "Plots", "BenchmarkTools", "StatsBase", "Distributions"])

# Generate data
Random.seed!(3000);
nobs=1000;
mix_dist = MixtureModel(Poisson[Poisson(2.0), Poisson(8.0)], [0.4, 0.6]);
Y=rand(mix_dist,nobs);

ex_hist = histogram(Y,normalize=:pdf, bins=0:20,label=nothing,xlabel="x",ylabel = "Relative frequency")

poisson_hat = fit_mle(Poisson, Y);
#geometric_hat = fit_mle(Geometric,Y);

histogram(Y,normalize=:pdf, bins=0:20,label="Rel. Freq")
plot!(pdf.(poisson_hat,0:20),linewidth=3,label = "PMF of Poisson fit")

# Step 1 (a): initial values for theta and lambda
theta_0=[1.0,2.0];
pi_0=[0.5,0.5];

# Step 1 (b): specify convergence criteria
iter=1;
maxiter=1000;
lik0=10.0;
normdiff=10.0;
tolerance=1e-6;


EZ = [(pi_0[j]*pdf(Poisson(theta_0[j]),Y[i]))/(sum(pi_0[k]*pdf(Poisson(theta_0[k]),Y[i]) for k∈eachindex(pi_0))) for i ∈ eachindex(Y), j ∈ eachindex(pi_0)];
EZ[1:5,:] # Showing how EZ (matrix NxJ looks, first 5 rows displayed here

# Alternatively:

# For loops (not in array comprehension mode)
EZ_2 = zeros(length(Y),length(pi_0));

for i in eachindex(Y)
    for j in eachindex(pi_0)
        EZ_2[i,j] = pdf(Poisson(theta_0[j]),Y[i]);
    end
end

EZ_2_rowsum = sum(EZ_2,dims=2);
for j in eachindex(pi_0)
    EZ_2[:,j] = EZ_2[:,j]./EZ_2_rowsum;
end

# Vectorized:

EZ_3 = pdf.(Poisson.(theta_0'),Y);
EZ_3_rowsum = repeat(sum(EZ_3,dims=2),1,2);
EZ_3 = EZ_3./EZ_3_rowsum;


pi_0=mean(EZ,dims=1);

theta_fun(theta_0) = -sum(EZ[i,j]*log(pdf(Poisson(theta_0[j]),Y[i])) for i ∈ eachindex(Y), j ∈ eachindex(pi_0));
b=optimize(theta_fun,theta_0);

# Resetting the initial values for optimization

# Step 1 (a): initial values for theta and lambda
theta_0=[1.0,2.0];
pi_0=[0.5,0.5];

# Step 1 (b): specify convergence criteria
iter=1;
maxiter=1000;
lik0=10.0;
normdiff=10.0;
tolerance=1e-6;

# Putting everything in a loop

while normdiff>tolerance && iter<=maxiter
        EZ = [(pi_0[j]*pdf(Poisson(theta_0[j]),Y[i]))/(sum(pi_0[k]*pdf(Poisson(theta_0[k]),Y[i]) for k∈eachindex(pi_0))) for i ∈ eachindex(Y), j ∈ eachindex(pi_0)];
        pi_0=mean(EZ,dims=1);
        theta_fun(theta_0) = -sum(EZ[i,j]*log(pdf(Poisson(theta_0[j]),Y[i])) for i ∈ eachindex(Y), j ∈ eachindex(pi_0));
        b=optimize(theta_fun,theta_0);
        # Calculate likelihood and update for next iteration:
        theta_0=b.minimizer;
        lik1 = -theta_fun(theta_0) + sum(log(pi_0[j])*EZ[i,j] for i∈eachindex(Y), j∈eachindex(pi_0));
        normdiff=abs(lik1-lik0);
        lik0=lik1;
        iter=iter+1;
    end

println(string("π_1, π_2 estimated to:",round.(pi_0,digits=3)))
println(string("θ_1, θ_2 estimated to:",round.(theta_0,digits=3)))


# Maximizing the whole likelihood at once

using OptimizationOptimJL 
#using Pkg
#Pkg.add("OptimizationOptimJL")

whole_lik(params,data) = -sum(log.(params[3].*pdf.(Poisson(params[1]),data) + (1-params[3]).*pdf.(Poisson(params[2]),data)));

p0 =  [1,2,0.5]

optf = OptimizationFunction(whole_lik, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf,p0,Y, lb = [0,0,0], ub = [Inf,Inf,1])
sol = solve(prob, NelderMead());

println(string("θ_1, θ_2, π_1 estimated to:",round.(sol.u,digits=3)))


histogram(Y,normalize=:pdf, bins=0:20,label="Rel. Freq")
plot!(pdf.(Poisson(theta_0[1]),0:20)*pi_0[1] .+ pdf.(Poisson(theta_0[2]),0:20)*pi_0[2],linewidth=3,label = "PMF of mixture distribution")

# Data generation (similar to yesterday's)

function value_function_iteration(X::AbstractRange{Float64},S::Vector{Int64},F1::Matrix{Float64},F2::Matrix{Float64},β::Number,θ::Vector;MaxIter=1000)
    x_len=length(X);
    γ=0.5772;
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
        #value_diff=sum((value_function1 .- value_function2).^2);
        value_diff=maximum((value_function1 .- value_function2).^2);
    end
    ccps=[1/(1+exp(v2[j,s,]-v1[j,s])) for j=1:x_len, s=1:length(S)];
    return (ccps_true=ccps, value_function=value_function2)
end

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

x_min=0.0;
x_max=10.0;
x_int=0.5;
x_len=Int32(1+(x_max-x_min)/x_int);
x=range(x_min,x_max,x_len);

# Transition matrix for mileage:
x_tran       = zeros((x_len, x_len));
x_tran_cumul = zeros((x_len, x_len));
x_tday      = repeat(x, 1, x_len); 
x_next      = x_tday';
x_zero      = zeros((x_len, x_len));

x_tran = (x_next.>=x_tday) .* exp.(-(x_next - x_tday)) .* (1 .- exp(-x_int));
x_tran[:,end]=1 .-sum(x_tran[:,1:(end-1)],dims=2);
x_tran_cumul=cumsum(x_tran,dims=2);

S=[1, 2];
s_len=Int32(length(S));
F1=zeros(x_len,x_len);
F1[:,1].=1.0;
F2=x_tran;

N=1000;
T=40;
X=x;
θ=[2.0, -0.15, 1.0];
β=0.9;
F_cumul=x_tran_cumul;
Random.seed!(3000);
XData, SData, DData, XIndexData, TData, NData = generate_data(N,T,X,S,F1,F2,F_cumul,β,θ; ex_initial=1);
γ=Base.MathConstants.eulergamma;


pi_0=0.5;
theta_0=[0.1,0.1,0.1];
q=[pi_0*ones(N,1) (1-pi_0)*ones(N,1)];

ccp_hat = [sum(repeat(q[:,s],T,1).*(vec(XIndexData) .== j).* (vec(DData) .== 1.0))/sum(repeat(q[:,s],T,1).*(vec(XIndexData) .== j)) for j∈eachindex(X), s∈eachindex(S)];

iter=1;
cond=0;
max_iter=1000;
tol=1e-6;
lik0=1.0;
stored_lik_vals=zeros(max_iter,1);

v1_ccp=repeat(β*(-log.(ccp_hat[1,:]') .+ γ),x_len,1);
v2_ccp=[theta_0[1]+theta_0[2]*X[j] + theta_0[3]*S[s] + β*(F2[j,:]'*(-log.(ccp_hat[:,s])) +γ) for j∈eachindex(X), s∈eachindex(S)];


like_pointwise_1= [((DData[n,t]==2.0)*exp(v2_ccp[XIndexData[n,t],1]-v1_ccp[XIndexData[n,t],1]) +(1-(DData[n,t]==2.0)))/(1+exp(v2_ccp[XIndexData[n,t],1]-v1_ccp[XIndexData[n,t],1])) for n=1:N, t=1:T];
like_pointwise_2= [((DData[n,t]==2.0)*exp(v2_ccp[XIndexData[n,t],2]-v1_ccp[XIndexData[n,t],2]) +(1-(DData[n,t]==2.0)))/(1+exp(v2_ccp[XIndexData[n,t],2]-v1_ccp[XIndexData[n,t],2])) for n=1:N, t=1:T];

ll_pw1=prod(like_pointwise_1,dims=2);
ll_pw2=prod(like_pointwise_2,dims=2);

q[:,1] .= (pi_0.*ll_pw1)./(pi_0.*ll_pw1 .+ (1-pi_0).*ll_pw2);
q[:,2] .= 1.0 .- q[:,1];
pi_0=mean(q[:,1]);

ccp_hat = [sum(repeat(q[:,s],T,1).*(vec(XIndexData) .== j).* (vec(DData) .== 1.0))/sum(repeat(q[:,s],T,1).*(vec(XIndexData) .== j)) for j∈eachindex(X), s∈eachindex(S)];

# Define the inner likelihood, to be optimized to obtain estimates for theta

function ccp_likelihood_inner(θ,N,T,X,S,F2,XIndexData,DData,q,ccp_hat,β)
    γ=Base.MathConstants.eulergamma;
    x_len=length(X);
    v1_ccp=repeat(β*(-log.(ccp_hat[1,:]') .+ γ),x_len,1);
    v2_ccp=[θ[1]+θ[2]*X[j] + θ[3]*S[s] + β*(F2[j,:]'*(-log.(ccp_hat[:,s])) +γ) for j∈eachindex(X), s∈eachindex(S)];
    q_use=repeat(q[:,1],1,T);
    ccp_lik_1=-sum(q_use[n,t]*((DData[n,t]==2.0)*(v2_ccp[XIndexData[n,t],1]-v1_ccp[XIndexData[n,t],1]) - log(1+exp(v2_ccp[XIndexData[n,t],1]-v1_ccp[XIndexData[n,t],1]))) for n=1:N, t=1:T);
    ccp_lik_2=-sum((1-q_use[n,t])*((DData[n,t]==2.0)*(v2_ccp[XIndexData[n,t],2]-v1_ccp[XIndexData[n,t],2]) - log(1+exp(v2_ccp[XIndexData[n,t],2]-v1_ccp[XIndexData[n,t],2]))) for n=1:N, t=1:T);
    return ccp_lik_1 + ccp_lik_2
end

# Optmize likelihood with respect to theta
inner_lik(θ) = ccp_likelihood_inner(θ,N,T,X,S,F2,XIndexData,DData,q,ccp_hat,β);
optim_res = optimize(inner_lik,theta_0,LBFGS(); autodiff = :forward);
theta_0=optim_res.minimizer;



function ccp_likelihood_inner(θ,N,T,X,S,F2,XIndexData,DData,q,ccp_hat,β)
    γ=Base.MathConstants.eulergamma;
    x_len=length(X);
    v1_ccp=repeat(β*(-log.(ccp_hat[1,:]') .+ γ),x_len,1);
    v2_ccp=[θ[1]+θ[2]*X[j] + θ[3]*S[s] + β*(F2[j,:]'*(-log.(ccp_hat[:,s])) +γ) for j∈eachindex(X), s∈eachindex(S)];
    q_use=repeat(q[:,1],1,T);
    ccp_lik_1=-sum(q_use[n,t]*((DData[n,t]==2.0)*(v2_ccp[XIndexData[n,t],1]-v1_ccp[XIndexData[n,t],1]) - log(1+exp(v2_ccp[XIndexData[n,t],1]-v1_ccp[XIndexData[n,t],1]))) for n=1:N, t=1:T);
    ccp_lik_2=-sum((1-q_use[n,t])*((DData[n,t]==2.0)*(v2_ccp[XIndexData[n,t],2]-v1_ccp[XIndexData[n,t],2]) - log(1+exp(v2_ccp[XIndexData[n,t],2]-v1_ccp[XIndexData[n,t],2]))) for n=1:N, t=1:T);
    return ccp_lik_1 + ccp_lik_2
end

while cond==0 && iter<=max_iter
    v1_ccp=repeat(β*(-log.(ccp_hat[1,:]') .+ γ),x_len,1);
    v2_ccp=[theta_0[1]+theta_0[2]*X[j] + theta_0[3]*S[s] + β*(F2[j,:]'*(-log.(ccp_hat[:,s])) +γ) for j∈eachindex(X), s∈eachindex(S)];
    
    # Pointwise likelihood:

    like_pointwise_1= [((DData[n,t]==2.0)*exp(v2_ccp[XIndexData[n,t],1]-v1_ccp[XIndexData[n,t],1]) +(1-(DData[n,t]==2.0)))/(1+exp(v2_ccp[XIndexData[n,t],1]-v1_ccp[XIndexData[n,t],1])) for n=1:N, t=1:T];
    like_pointwise_2= [((DData[n,t]==2.0)*exp(v2_ccp[XIndexData[n,t],2]-v1_ccp[XIndexData[n,t],2]) +(1-(DData[n,t]==2.0)))/(1+exp(v2_ccp[XIndexData[n,t],2]-v1_ccp[XIndexData[n,t],2])) for n=1:N, t=1:T];
    ll_pw1=prod(like_pointwise_1,dims=2);
    ll_pw2=prod(like_pointwise_2,dims=2);

    q[:,1] .= (pi_0.*ll_pw1)./(pi_0.*ll_pw1 .+ (1-pi_0).*ll_pw2);
    q[:,2] .= 1.0 .- q[:,1];
    pi_0=mean(q[:,1]);
    # Maximization
    ccp_hat = [sum(repeat(q[:,s],T,1).*(vec(XIndexData) .== j).* (vec(DData) .== 1.0))/sum(repeat(q[:,s],T,1).*(vec(XIndexData) .== j)) for j∈eachindex(X), s∈eachindex(S)];    inner_lik(θ) = ccp_likelihood_inner(θ,N,T,X,S,F2,XIndexData,DData,q,ccp_hat,β);
    inner_lik(θ) = ccp_likelihood_inner(θ,N,T,X,S,F2,XIndexData,DData,q,ccp_hat,β);
    optim_res = optimize(inner_lik,theta_0,LBFGS(); autodiff = :forward);
    theta_0=optim_res.minimizer;

    stored_lik_vals[iter]=sum(q[:,1].*log.(ll_pw1) + q[:,2].*log.(ll_pw2)) + sum(log(pi_0)*q[:,1] + log(1-pi_0)*q[:,2]);
    if iter>25 
        lik_diff=abs((stored_lik_vals[iter]-stored_lik_vals[iter-25])/stored_lik_vals[iter-25]);
        if lik_diff<tol
            cond=1;
        end
    end
    iter=iter+1;
    if iter>max_iter
        print("Maximum number of iterations reached")
        break
    end
end

println(string("θ_1, θ_2, θ_3 estimated to:",round.(theta_0,digits=3)))
println(string("π_1 estimated to:",round.(pi_0,digits=3)))

