using LinearAlgebra, Plots, DataFrames, SpecialFunctions, Random, GLM

## GENERATE DATA

Nm=3000;
Nf=5;
T=10;
Tl=10;
S=5;
Xn=10;
β=.9;

#structural entry coefficients theta:
bx=-.05;
bx0=0;
bnf=-.2;
bss=.25;
be=-1.5;

# Price coefficients -- not important until there are unobserved states
bp=[7; -.4 ;-.1 ;.3];

# Parameters governing transition of the unobservables
ps=.7;
nps=(1-ps)/(S-1);
trans=ps*I(S)+nps*(1 .-I(S));

ctrans = zeros(S,S-1);
ctrans[:,1]=trans[:,1];

for s=1:S-2
    ctrans[:,s+1]=ctrans[:,s]+trans[:,s+1];
end

nf=(0:Nf)';

# Generate firm-level utility for all state combinations

state_args=(0:Nf,0:Xn-1,0:S-1,0:1);

NC=4;
ii = NC:-1:1;

A = DataFrame(Iterators.product(state_args[1],state_args[2],state_args[3],state_args[4]));
A = Matrix(A);
#A2 = vec(collect(Iterators.product(state_args[1],state_args[2],state_args[3],state_args[4])));

# Fill binomial

bine=zeros(Nf+1,Nf+1);
for n in 1:Nf+1   
    for k in 1:n       
        bine[n,k]=binomial(n-1,k-1);                  
    end        
end

## Generate data

theta=[bnf, bx, bss, be]';
Util=zeros(size(A,1),1);
for j in 1:size(A,1)
    Util[j]=0+bx*A[j,2]+bnf*A[j,1]+bss*A[j,3]+be*(1-A[j,4]);
end

N=Nf;


# Find the fixed point that characterizes the equilibrium.

#out = prob_entry(Util,trans,Nf,Xn,S,bine,β,A);
prob_out,h2,h3 = prob_entry(Util,trans,Nf,Xn,S,bine,β,A);

Random.seed!(2023);

# Using the equilibrium choice probabilities, simulate the data on firm
# choices and states.
Firm,X,State,Y,Lfirm =EntryDataGen(prob_out,ctrans,S,Xn,Nf+1,Nm,T,bp,Tl,A);

## FIRST STAGE ESTIMATION OF CCPs

# Reshape the data
S_vec = repeat(vec(State),Nf + 1); 
X_vec = repeat(vec(X),(Nf + 1)*Xn);

# Add firm-level decisions to get number of active firms in the previous period:
NFirm=dropdims(sum(Firm,dims = 3), dims= 3);
LNFirm=dropdims(sum(Lfirm,dims = 3), dims = 3);

NFirm_vec = repeat(vec(NFirm),Nf + 1); 
LNFirm_vec = repeat(vec(LNFirm),Nf + 1); 

Firm_vec=vec(Firm);
LFirm_vec=vec(Lfirm);

LNFirm_vec=LNFirm_vec-LFirm_vec;

Z = hcat(ones((Nf+1)*Nm*T,1), X_vec, S_vec, 1 .- LFirm_vec,LNFirm_vec);


# Create variables used in logit for reduced-form CCP estimation:
W_ccp = [ones(size(X_vec)) X_vec (X_vec./10).^2  LFirm_vec LNFirm_vec (LNFirm_vec./5).^2 S_vec S_vec.*X_vec./10 LFirm_vec.*S_vec LNFirm_vec.*S_vec./10];
model1 =  glm(W_ccp, Firm_vec, Binomial(), LogitLink());
lambda_hat =  model1.pp.beta0;

# Fit CCPs to each observations:
B = DataFrame(A,["LNFirms","X","S","LFirm"]);

B_fit = [ones(size(B,1),1) B.X (B.X./10).^2 B.LFirm B.LNFirms (B.LNFirms./5).^2 B.S B.S.*B.X./10 B.LFirm.*B.S B.S.*B.LNFirms./10];
ccp_hat=predict(model1,B_fit);


#### FUNCTIONS

function nfirms(pe1,pi1,bine,bini,ne,N)

    Pe=zeros(ne+1,1);
    for j in 1:ne+1
        Pe[j]=bine[j]*(pe1^(j-1))*((1-pe1)^(ne-(j-1)));
    end
    
    Pi=zeros(N-ne+1,1);
    for j in 1:N-ne+1
        Pi[j]=bini[j]*(pi1^(j-1))*((1-pi1)^(N-ne-(j-1)));
    end
    
    BigP=zeros(1,N+1);
    j=0;
    
    while j<ne+1
        k=0;
        while k<N-ne+1
            BigP[j+k+1]=Pe[j+1]*Pi[k+1]+BigP[j+k+1];
            k=k+1;
        end
        j=j+1;
    end
    return BigP;
end


function prob_entry(Util,trans,N,Xn,S,bine,Beta,A)
    BigP=zeros(size(A,1),N+1);
    fv=zeros(size(A,1),1);
    eul=Base.MathConstants.eulergamma;
    
    x1 = (1 .+ exp.(Util));
    p=exp.(Util)./(1 .+ exp.(Util));
    
    p0=zeros((N+1)*Xn*S*2,1);
    p2=p;

    while maximum(abs.(p0-p2))>.0000000001
        p0=p2;

        for j in 1:size(A,1)
            ind1= only(findall(row -> row == [minimum([A[j,1]+A[j,4],N]),A[j,2],A[j,3],0], eachrow(A)));
            ind2= only(findall(row -> row == [maximum([A[j,1]+A[j,4]-1,0]),A[j,2],A[j,3],1], eachrow(A)));
            BigP[j,:]=nfirms(p[ind1],p[ind2],bine[N-A[j,1]+1,:],bine[A[j,1]+1,:],N-A[j,1],N);
    
            v=0;
            for s2=1:S
                toLookIntoP = hcat(collect(0:5),repeat([A[j,2] s2-1 1],outer = [6,1]));
                elementsP = [sum(row) >= 1 for row in eachrow([all(row_a .== row_b) for row_a in eachrow(A), row_b in eachrow(toLookIntoP)])];
                v=v + trans[A[j,3]+1,s2]*(BigP[j,:]'*log.(1 .- p[elementsP]));
            end
            fv[j]=-v;
        
            toLookIntoUtil = hcat(collect(0:5),repeat([A[j,2] A[j,3] A[j,4]],outer = [6,1]))
            elementsUtil = [sum(row) >= 1 for row in eachrow([all(row_a .== row_b) for row_a in eachrow(A), row_b in eachrow(toLookIntoUtil)])];
            tu=BigP[j,:]'*Util[elementsUtil]-Beta*v+Beta*eul;
            p[j]=exp(tu)/(1+exp(tu));
    
        end
        p2=p;
    end
    fv=Beta.*(fv .+ eul);
    return p,BigP,fv;
end

## function below not updated

function EntryDataGen(p,ctrans,S,Xn,Nf,Nm,T,bp,Tl,A)
    Firm=zeros(Nm,T+Tl,Nf);
    Lfirm=zeros(Nm,T+Tl+1,Nf);
    X = rand(0:Xn-1,Nm);
    State=zeros(Nm,T+Tl+1);
    Y=zeros(Nm,T+Tl);
    
    State[:,1]=rand(1:S,Nm);
    
    Draw1=rand(Nm,T+Tl,Nf);
    Draw2=rand(Nm,T+Tl);
    Draw3=randn(Nm,T+Tl);
    
    for nm in 1:Nm
        Nfirm=0;
        for t in 1:T+Tl
            for nf in 1:Nf
                ind = [findfirst(row -> row == [Nfirm-Lfirm[nm,t,nf],X[nm],State[nm,t]-1,Lfirm[nm,t,nf]], eachrow(A))]
                Firm[nm,t,nf]=p[only(ind)]>Draw1[nm,t,nf];
            end
    
            Nfirm=sum(Firm[nm,t,:]);
            Lfirm[nm,t+1,:]=Firm[nm,t,:];
    
            Y[nm,t]=only([1 Nfirm X[nm]-1 State[nm,t]-1]*bp) + Draw3[nm,t];

            State[nm,t+1]=1;
    
            for s=1:S-1
                State[nm,t+1]=State[nm,t+1]+(Draw2[nm,t]>ctrans[Int(State[nm,t]),s]);
            end
        end
    end
    
    Firm=Firm[:,Tl+1:T+Tl,:];
    State=State[:,Tl+1:T+Tl] .- 1;
    Lfirm=Lfirm[:,Tl+1:T+Tl,:];
    Y=Y[:,Tl+1:T+Tl];

    return Firm, X, State, Y, Lfirm
end
    



####
