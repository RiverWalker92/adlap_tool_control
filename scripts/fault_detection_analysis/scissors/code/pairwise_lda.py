"""Batch one/two-feature LDA with the existing lsqr/auto/equal-prior method.

Each pair has its own two-feature Ledoit-Wolf shrinkage, not a submatrix
of a 68-feature LDA. Marginal training/class moments are shared for speed.
Reference: sklearn.discriminant_analysis._cov/_solve_lstsq and
sklearn.covariance.ledoit_wolf_shrinkage. Validate against sklearn before use.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.linalg import lstsq

class PairwiseLDA:
    def __init__(self, first, second):
        self.first=np.asarray(first,int)
        self.second=np.asarray(second,int)
        self.single=self.second<0
        self.safe_second=np.where(self.single,self.first,self.second)

    def fit(self,X,y,k):
        X=np.asarray(X,float);y=np.asarray(y,int)
        self.scaler=StandardScaler().fit(X)
        T=self.scaler.transform(X)
        i=self.first;j=self.safe_second;m=len(i)
        cov=np.zeros((m,2,2));means=np.zeros((m,k,2))
        for label in range(k):
            C=T[y==label]
            if len(C)==0:raise ValueError('A training class is empty.')
            means[:,label,0]=C.mean(0)[i]
            means[:,label,1]=np.where(self.single,0,C.mean(0)[j])
            scale=StandardScaler().fit(C)
            Z=scale.transform(C)
            # Ledoit-Wolf centers the standardised class again.
            Z-=Z.mean(0)
            n=len(Z)
            G=(Z.T@Z)/n
            H=((Z*Z).T@(Z*Z))/n
            a=np.diag(G)[i];b=np.diag(G)[j];c=G[i,j]
            mu=(a+b)/2
            delta0=a*a+b*b+2*c*c
            beta=(np.diag(H)[i]+np.diag(H)[j]+2*H[i,j]-delta0)/(2*n)
            delta=(delta0-2*mu*(a+b)+2*mu*mu)/2
            beta=np.minimum(beta,delta)
            alpha=np.divide(beta,delta,out=np.zeros(m),where=(beta!=0)&(delta!=0))
            alpha[self.single]=0
            si=scale.scale_[i];sj=scale.scale_[j]
            cov[:,0,0]+=((1-alpha)*a+alpha*mu)*si*si/k
            cov[:,1,1]+=np.where(self.single,0,((1-alpha)*b+alpha*mu)*sj*sj/k)
            off=np.where(self.single,0,(1-alpha)*c*si*sj)/k
            cov[:,0,1]+=off;cov[:,1,0]+=off
        # The 2x2 solution uses the same machine-epsilon rank cutoff as scipy lstsq.
        vals,vec=np.linalg.eigh(cov)
        cutoff=np.max(np.abs(vals),axis=1)*np.finfo(float).eps
        inv=np.divide(1.,vals,out=np.zeros_like(vals),where=vals>cutoff[:,None])
        pinv=np.einsum('mab,mb,mcb->mac',vec,inv,vec)
        coef=np.einsum('mab,mkb->mka',pinv,means)
        # Reproduce the reference SVD solve for rank-deficient/ill-conditioned pairs.
        special=(~self.single)&(vals[:,0]<=np.maximum(vals[:,1],0)*1e-10)
        for idx in np.flatnonzero(special):
            coef[idx]=lstsq(cov[idx],means[idx].T)[0].T
        std_bias=-.5*np.einsum('mka,mka->mk',means,coef)+np.log(1/k)
        scales=np.column_stack([self.scaler.scale_[i],self.scaler.scale_[j]])
        offsets=np.column_stack([self.scaler.mean_[i],self.scaler.mean_[j]])
        self.coef=coef/scales[:,None,:]
        self.intercept=std_bias-np.einsum('mka,ma->mk',self.coef,offsets)
        self.k=k
        return self

    def scores(self,X):
        X=np.asarray(X,float)
        V=np.stack([X[:,self.first],X[:,self.safe_second]],axis=-1)
        return np.einsum('nma,mka->nmk',V,self.coef)+self.intercept[None,:,:]

    def predict(self,X):
        return self.scores(X).argmax(axis=-1).astype(np.int8)

    def coefficients(self,model_index):
        co=self.coef[model_index];b=self.intercept[model_index]
        if self.k==2:co=(co[1]-co[0])[None,:];b=np.asarray([b[1]-b[0]])
        if self.single[model_index]:co=co[:,:1]
        return co,b
