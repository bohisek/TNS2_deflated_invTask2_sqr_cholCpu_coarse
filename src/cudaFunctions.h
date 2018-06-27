#ifndef CUDAFUNCTIONS_H_
#define CUDAFUNCTIONS_H_

// declare CUDA fields
T *dT , *dr , *dp , *dq , *dz;
T *dcc, *dss, *dww;  // matrix A center, south, west stencil

T *kww, *kw , *kc;  // (TNS2) matrix M-1 center, south, west stencil
T       *ksw, *ks, *kse, *ksee;
T             *kss, *ksse;

T *dpp;              // matrix P = sqrt(D)
T *drh, *dsg;        // partial dot products
T *dV;               // cell volume
T *dqB;              // Neumann BC bottom



// initialize CUDA fields
template <class T>
void cudaInit( T *hT,
		T *hV,
		T *hcc,
		T *hss,
		T *hww,
		T *hqB,
		const int blocks,
		const int Nx,
		const int Ny)
{
	cudaMalloc((void**)&dT ,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMalloc((void**)&dr ,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMalloc((void**)&dV ,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMalloc((void**)&dp ,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMalloc((void**)&dq ,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMalloc((void**)&dz ,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMalloc((void**)&dpp,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMalloc((void**)&dcc,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMalloc((void**)&dss,sizeof(T)*(Nx*Ny+Nx  ));
	cudaMalloc((void**)&dww,sizeof(T)*(Nx*Ny+1   ));
	cudaMalloc((void**)&drh,sizeof(T)*(blocks    ));
	cudaMalloc((void**)&dsg,sizeof(T)*(blocks    ));
	cudaMalloc((void**)&dqB,sizeof(T)*(Nx        ));

	cudaMalloc((void**)&kc,sizeof(T)*(Nx*Ny+2*Nx)); // 7 diagonals for M^-1 (some diagonals omitted)
	cudaMalloc((void**)&ks,sizeof(T)*(Nx*Ny+Nx  ));
	cudaMalloc((void**)&kw,sizeof(T)*(Nx*Ny+1   ));
	cudaMalloc((void**)&ksw ,sizeof(T)*(Nx*Ny+Nx+1));  //               WW   W   C
	cudaMalloc((void**)&kse ,sizeof(T)*(Nx*Ny+Nx-1));  //                    SW  S  SE SEE
	cudaMalloc((void**)&kss ,sizeof(T)*(Nx*Ny+2*Nx));  //                        SS SSE
	cudaMalloc((void**)&kww ,sizeof(T)*(Nx*Ny+2)   );
	cudaMalloc((void**)&ksee,sizeof(T)*(Nx*Ny+Nx-2));
	cudaMalloc((void**)&ksse,sizeof(T)*(Nx*Ny+2*Nx-1));

	cudaMemcpy(dT ,hT ,sizeof(T)*(Nx*Ny+2*Nx),cudaMemcpyHostToDevice);
	cudaMemcpy(dcc,hcc,sizeof(T)*(Nx*Ny+2*Nx),cudaMemcpyHostToDevice);
	cudaMemcpy(dV, hV ,sizeof(T)*(Nx*Ny+2*Nx),cudaMemcpyHostToDevice);
	cudaMemcpy(dss,hss,sizeof(T)*(Nx*Ny+Nx  ),cudaMemcpyHostToDevice);
	cudaMemcpy(dww,hww,sizeof(T)*(Nx*Ny+1   ),cudaMemcpyHostToDevice);
	cudaMemcpy(dqB,hqB,sizeof(T)*(Nx        ),cudaMemcpyHostToDevice);

	cudaMemset(dr ,0,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMemset(dp ,0,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMemset(dq ,0,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMemset(dz ,0,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMemset(dpp,0,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMemset(drh,0,sizeof(T)*(blocks    ));
	cudaMemset(dsg,0,sizeof(T)*(blocks    ));

	cudaMemset(kc, 0,sizeof(T)*(Nx*Ny+2*Nx)); // 7 diagonals for M^-1 (some diagonals omitted)
	cudaMemset(ks, 0,sizeof(T)*(Nx*Ny+Nx  ));
	cudaMemset(kw, 0,sizeof(T)*(Nx*Ny+1   ));
	cudaMemset(ksw,0,sizeof(T)*(Nx*Ny+Nx+1));  //               WW   W   C
	cudaMemset(kse,0,sizeof(T)*(Nx*Ny+Nx-1));  //                    SW  S  SE SEE
	cudaMemset(kss,0,sizeof(T)*(Nx*Ny+2*Nx));  //                        SS SSE
	cudaMemset(kww,0,sizeof(T)*(Nx*Ny+2)   );
	cudaMemset(ksee,0,sizeof(T)*(Nx*Ny+Nx-2));
	cudaMemset(ksse,0,sizeof(T)*(Nx*Ny+2*Nx-1));

}



// destroy CUDA fields
void cudaFinalize()
{
	cudaFree(dT);
	cudaFree(dr);
	cudaFree(dp);
	cudaFree(dq);
	cudaFree(dz);
	cudaFree(dV);
	cudaFree(dpp);
	cudaFree(dcc);
	cudaFree(dss);
	cudaFree(dww);
	cudaFree(drh);
	cudaFree(dsg);
	cudaFree(dqB);

	cudaFree(kc); // TNS2
	cudaFree(ks);
	cudaFree(kw);
	cudaFree(ksw);
	cudaFree(kse);
	cudaFree(kss);
	cudaFree(kww);
	cudaFree(ksee);
	cudaFree(ksse);

	cudaDeviceReset();
}

// AXPY (y := alpha*x + beta*y)
template <class T>
__global__ void AXPY(T *y,
		const T *x,
		const T alpha,
		const T beta,
		const int Nx)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x + Nx;
	y[tid] = alpha * x[tid] + beta * y[tid];
}

// SPMV (sparse matrix-vector multiplication)
template <class T>
__global__ void SpMVv1(T *y,
		const T *stC,
		const T *stS,
		const T *stW,
		const T *x,
		const int Nx)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x + Nx;

	y[tid]  = stC[tid]      * x[tid]     // center
	    	+ stS[tid]      * x[tid+Nx]  // north               N
	        + stW[tid-Nx+1] * x[tid+1]   // east              W C E
	        + stS[tid-Nx]   * x[tid-Nx]  // south               S
	        + stW[tid-Nx]   * x[tid-1];  // west
}

// SPMV (sparse matrix-vector multiplication)
template <class T>
__global__ void SpMVv2(T *y,
		const T *stC,
		const T *stS,
		const T *stW,
		const T *stSW,
		const T *stSE,
		const T *stSS,
		const T *stWW,
		const T *stSEE,
		const T *stSSE,
		const T *x,
		const int Nx,
		const int Ny)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	y[tid+Nx]  = stC[tid+Nx]    * x[tid+Nx]    // C                NNW NN                OK **
	    	+ stS[tid+Nx]    * x[tid+2*Nx]     // N            NWW NW  N  NE             OK **
	        + stW[tid+1]     * x[tid+Nx+1]     // E            WW  W   C  E  EE          OK **
	        + stS[tid]       * x[tid]          // S                SW  S  SE SEE         OK **
	        + stW[tid]       * x[tid+Nx-1]     // W                    SS SSE            OK **
	        + stSE[tid]      * x[tid+1]        // SE                                     OK'**
	        + stWW[tid]      * x[tid+Nx-2]     // WW                                     OK'**
	        + stWW[tid+2]    * x[tid+Nx+2]     // EE                                     OK'**
	        + stSE[tid+Nx-1] * x[tid+2*Nx-1]   // NW                                     OK'**
	        + stSEE[tid]     * x[tid+2]        // SEE
	        + stSEE[tid+Nx-2]* x[tid+2*Nx-2];  // NWW

	// Note: it seems the solver is a bit faster without SEE(NWW) and SSE(NNW), although the number of iterations is slightly higher.

	if (tid>0)             y[tid+Nx]  += stSW[tid]        * x[tid-1];        // SW         OK'**
	if (tid>Nx-1)          y[tid+Nx]  += stSS[tid]        * x[tid-Nx];       // SS         OK'**
	if (tid>Nx-2)          y[tid+Nx]  += stSSE[tid]       * x[tid-Nx+1];     // SSE
	if (tid<Nx*Ny-1)       y[tid+Nx]  += stSW[tid+Nx+1]   * x[tid+2*Nx+1];   // NE         OK'**
	if (tid<Nx*Ny-Nx)      y[tid+Nx]  += stSS[tid+2*Nx]   * x[tid+3*Nx];     // NN         OK'**
	if (tid<Nx*Ny-Nx+1)    y[tid+Nx]  += stSSE[tid+2*Nx-1]* x[tid+3*Nx-1];   // NNW

}

// DOT PRODUCT
template <class T, unsigned int blockSize>
__global__ void DOTGPU(T *c,
		const T *a,
		const T *b,
		const int Nx,
		const int Ny)
{
	extern __shared__ T cache[];

	unsigned int tid = threadIdx.x;
	unsigned int i = tid + blockIdx.x * (blockSize * 2);
	unsigned int gridSize = (blockSize*2)*gridDim.x;


	cache[tid] = 0;

	while(i<Nx*Ny) {
		cache[tid] += a[i+Nx] * b[i+Nx] + a[i+Nx+blockSize] * b[i+Nx+blockSize];
		i += gridSize;
	}

	__syncthreads();

	if(blockSize >= 512) {	if(tid < 256) { cache[tid] += cache[tid + 256]; } __syncthreads(); }
	if(blockSize >= 256) {	if(tid < 128) { cache[tid] += cache[tid + 128]; } __syncthreads(); }
	if(blockSize >= 128) {	if(tid < 64 ) { cache[tid] += cache[tid + 64 ]; } __syncthreads(); }

	if(tid < 32) {
		cache[tid] += cache[tid + 32];
		cache[tid] += cache[tid + 16];
		cache[tid] += cache[tid + 8];
		cache[tid] += cache[tid + 4];
		cache[tid] += cache[tid + 2];
		cache[tid] += cache[tid + 1];
	}

	if (tid == 0) c[blockIdx.x] = cache[0];
}


//
template <class T>
__global__ void elementWiseMul(T *x,
		const T *p,
		const int Nx)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x + Nx;
	x[tid] *= p[tid];
}


// Truncated Neumann series 2
template <class T>
__global__ void makeTNS2(T *smC,
		T *smS,
		T *smW,
		T *smSW,
		T *smSE,
		T *smSS,
		T *smWW,
		T *smSEE,
		T *smSSE,
		const T *stC,
		const T *stS,
		const T *stW,
		const int Nx,
		const int Ny)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	T tstC2 = 0.;
	T tstC3 = 0.;
	T tstC4 = 0.;
	T tstC5 = 0.;
	T tstC6 = 0.;
	T tstC7 = 0.;
	T tstC8 = 0.;
	T tstC9 = 0.;
	T                       tstC1 = 1. / stC[tid+Nx];     //OK*  0
	if (tid < Nx*Ny-1)      tstC3 = 1. / stC[tid+Nx+1];   //OK*  1
	if (tid < Nx*Ny-2)      tstC5 = 1. / stC[tid+Nx+2];   //  *  2
	if (tid < Nx*Ny-Nx+2)   tstC9 = 1. / stC[tid+2*Nx-2];
	if (tid < Nx*Ny-Nx+1)   tstC4 = 1. / stC[tid+2*Nx-1]; //  *
	if (tid < Nx*Ny-Nx)     tstC2 = 1. / stC[tid+2*Nx];   //OK*.
	if (tid < Nx*Ny-Nx-1)   tstC7 = 1. / stC[tid+2*Nx+1]; //  *
	if (tid < Nx*Ny-2*Nx+1) tstC8 = 1. / stC[tid+3*Nx-1]; //  *
	if (tid < Nx*Ny-2*Nx)   tstC6 = 1. / stC[tid+3*Nx];   //  *

	smC[tid+Nx] = 1 * tstC1                                                                                               // C     ok     OK*
			+  (stS[tid+Nx] * tstC1)                          *  (stS[tid+Nx] * tstC1)                          * tstC2   // S     ok     OK*
			+  (stW[tid+1]  * tstC1)                          *  (stW[tid+1]  * tstC1)                          * tstC3   // W     ok     OK*
			+ ((stW[tid+1]  * tstC1) * (stW[tid+2]  * tstC3)) * ((stW[tid+1]  * tstC1) * (stW[tid+2]  * tstC3)) * tstC5   // WW    ok       *
			+ ((stS[tid+Nx] * tstC1) * (stS[tid+2*Nx]*tstC2)) * ((stS[tid+Nx] * tstC1) * (stS[tid+2*Nx]*tstC2)) * tstC6   // SS    ok       *
			+ ((stW[tid+1]  * tstC1) * (stS[tid+Nx+1]*tstC3)) * ((stW[tid+1]  * tstC1) * (stS[tid+Nx+1]*tstC3)) * tstC7   // SW(a) ok       *
			+ ((stS[tid+Nx] * tstC1) * (stW[tid+Nx+1]*tstC2)) * ((stS[tid+Nx] * tstC1) * (stW[tid+Nx+1]*tstC2)) * tstC7;  // SW(b) ok       *

	smW[tid+1]  = -stW[tid+1]  * tstC1 * tstC3    // W   (a)                                                             ok    OK *
			  + ((stW[tid+1]  * tstC1) * (stW[tid+2]  * tstC3)) * tstC5      *       (-stW[tid+2]  * tstC3)    // (b)    ok      *
			  + ((stW[tid+1]  * tstC1) * (stS[tid+Nx+1]*tstC3)) * tstC7      *       (-stS[tid+Nx+1]*tstC3)    // (c)    ok      *
			  + ((stS[tid+Nx] * tstC1) * (stW[tid+Nx+1]*tstC2)) * tstC7      *       (-stS[tid+Nx+1]*tstC3);   //        ok      *

	smS[tid+Nx] = -stS[tid+Nx] * tstC1 * tstC2   // S   (a)                                                              ok    OK*
			  + ((stW[tid+1]  * tstC1) * (stS[tid+Nx+1]*tstC3)) * tstC7      *       (-stW[tid+Nx+1]*tstC2)    // (b)    ok      *
			  + ((stS[tid+Nx] * tstC1) * (stW[tid+Nx+1]*tstC2)) * tstC7      *       (-stW[tid+Nx+1]*tstC2)    // (b)    ok      *
			  + ((stS[tid+Nx] * tstC1) * (stS[tid+2*Nx]*tstC2)) * tstC6      *       (-stS[tid+2*Nx]*tstC2);   // (c)    ok      *


	smSE[tid+Nx-1] = (-stS[tid+Nx] * tstC1) * tstC2      *  (-stW[tid+Nx] * tstC4) // (a)                                                               ok*
			+ ((stW[tid+1]  * tstC1) * (stS[tid+Nx+1]*tstC3)) * tstC7      *    ((stW[tid+Nx]    * tstC4) * (stW[tid+Nx+1]  * tstC2))      // (b)       ok*
			+ ((stS[tid+Nx] * tstC1) * (stW[tid+Nx+1]*tstC2)) * tstC7      *    ((stW[tid+Nx]    * tstC4) * (stW[tid+Nx+1]  * tstC2))      // (b)       ok*
			+ ((stS[tid+Nx] * tstC1) * (stS[tid+2*Nx]*tstC2)) * tstC6      *    ((stW[tid+Nx]    * tstC4) * (stS[tid+2*Nx]  * tstC2))      // (c)       ok*
			+ ((stS[tid+Nx] * tstC1) * (stS[tid+2*Nx]*tstC2)) * tstC6      *    ((stS[tid+2*Nx-1]* tstC4) * (stW[tid+2*Nx]  * tstC8));     // (c)       ok*


	smSW[tid+Nx+1]    = + ((stW[tid+1]  * tstC1) * (stS[tid+Nx+1]*tstC3))  * tstC7   // (a)   ok*
				        + ((stS[tid+Nx] * tstC1) * (stW[tid+Nx+1]*tstC2))  * tstC7;  // (a)   ok*


    smSS[tid+2*Nx]    = ((stS[tid+Nx] * tstC1) * (stS[tid+2*Nx]* tstC2))   * tstC6;   // ok*

    smWW[tid+2]       = ((stW[tid+1]  * tstC1) * (stW[tid+2]   * tstC3))   * tstC5;   // ok*

    smSEE[tid+Nx-2]   = (-stS[tid+Nx] * tstC1) * tstC2 * ((stW[tid+Nx-1]*tstC9) * (stW[tid+Nx]*tstC4));

    smSSE[tid+2*Nx-1] = ((stS[tid+Nx] * tstC1) * (stS[tid+2*Nx]* tstC2))   * tstC6  *  (-stW[tid+2*Nx]*tstC8);

}


// for thermal boundary condition
template <class T>
__global__ void addNeumannBC(T *x,
		const T *Q,
		const T HeatFlux,
		const int Nx)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	x[tid+Nx] += HeatFlux * Q[tid];
}

#endif /* CUDAFUNCTIONS_H_ */
