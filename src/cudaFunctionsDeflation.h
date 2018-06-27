#ifndef CUDAFUNCTIONSDEFLATION_H_
#define CUDAFUNCTIONSDEFLATION_H_

// deflation
T *dazc, *dazw, *dazs;
T *decc, *deww, *dess;
T *dyZ, *drZ, *dpZ , *dqZ;

// initialize CUDA fields
template <class T>
void cudaInitDeflation( T *azc,
		T *azw,
		T *azs,
		T *ecc,
		T *eww,
		T *ess,
		const int NxZ,
		const int nDV,
		const int Nx,
		const int Ny)
{
	cudaMalloc((void**)&dazc ,sizeof(T)*(Nx*Ny));
	cudaMalloc((void**)&dazw ,sizeof(T)*(Nx*Ny+1));
	cudaMalloc((void**)&dazs ,sizeof(T)*(Nx*Ny+Nx));
	cudaMalloc((void**)&dyZ  ,sizeof(T)*(nDV+2*NxZ));
	cudaMalloc((void**)&drZ  ,sizeof(T)*(nDV+2*NxZ));
	cudaMalloc((void**)&dpZ  ,sizeof(T)*(nDV+2*NxZ));
	cudaMalloc((void**)&dqZ  ,sizeof(T)*(nDV+2*NxZ));
	cudaMalloc((void**)&decc ,sizeof(T)*(nDV+2*NxZ));
	cudaMalloc((void**)&deww ,sizeof(T)*(nDV+1));
	cudaMalloc((void**)&dess ,sizeof(T)*(nDV+NxZ));


	cudaMemcpy(dazc ,azc ,sizeof(T)*(Nx*Ny)    ,cudaMemcpyHostToDevice);
	cudaMemcpy(dazw ,azw ,sizeof(T)*(Nx*Ny+1)  ,cudaMemcpyHostToDevice);
	cudaMemcpy(dazs ,azs ,sizeof(T)*(Nx*Ny+Nx) ,cudaMemcpyHostToDevice);
	cudaMemcpy(decc ,ecc ,sizeof(T)*(nDV+2*NxZ),cudaMemcpyHostToDevice);
	cudaMemcpy(deww ,eww ,sizeof(T)*(nDV+1)    ,cudaMemcpyHostToDevice);
	cudaMemcpy(dess ,ess ,sizeof(T)*(nDV+NxZ)  ,cudaMemcpyHostToDevice);

	cudaMemset(dyZ ,0,sizeof(T)*(nDV+2*NxZ));
	cudaMemset(drZ ,0,sizeof(T)*(nDV+2*NxZ));
	cudaMemset(dpZ ,0,sizeof(T)*(nDV+2*NxZ));
	cudaMemset(dqZ ,0,sizeof(T)*(nDV+2*NxZ));
}

void cudaFinalizeDeflation()
{
	cudaFree(dazc);
	cudaFree(dazw);
	cudaFree(dazs);
	cudaFree(dyZ);
	cudaFree(drZ);
	cudaFree(dpZ);
	cudaFree(dqZ);
	cudaFree(decc);
	cudaFree(deww);
	cudaFree(dess);
}

// ---- y1 = Z' * y (works!) ----
template <class T>
__global__ void ZTransXYDeflation(T *y,
		const T *x,
		const int nRowsZ,
		const int NxZ,
		const int Nx)
{
	// nDV (number of deflation vectors) threads must be launched!
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int id0 = nRowsZ*Nx*(tid/NxZ) + nRowsZ*(tid%NxZ);         // a global index of the first cell in a coarse cell  (parenthesis are necessary!)

	y[tid+NxZ] = 0.;	                                      // reset

	for (int j=0; j<nRowsZ*nRowsZ; j++)                       // loop over cells of a single coarse cell
	{
		y[tid+NxZ] += x[id0 + Nx*(j/nRowsZ) + j%nRowsZ + Nx];
	}
}

// ---- y (= Py) = y - AZ * y2 ----
template <class T>
__global__ void YMinusAzXYDeflation(T *y,
		const T *x,
		const T *azc,
		const T *azw,
		const T *azs,
		const int nRowsZ,
		const int NxZ,
		const int Nx)
{
		// Nx*Ny threads must be launched!
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		int ixZ = (tid%Nx)/nRowsZ;        // x-index of coarse system
		int jyZ = (tid/Nx)/nRowsZ;        // y-index
		int idZ = jyZ*NxZ + ixZ;          // linear index of course system

		idZ += NxZ;

		y[tid+Nx]  -= + azc[tid]    * x[idZ]  // center
		         + azw[tid]    * x[idZ-1]     // west
		         + azw[tid+1]  * x[idZ+1]     // east
		         + azs[tid]    * x[idZ-NxZ]   // south
		         + azs[tid+Nx] * x[idZ+NxZ];  // north
	}

#endif /* CUDAFUNCTIONSDEFLATION_H_ */
