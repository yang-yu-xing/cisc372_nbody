#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

__global__ void parallel(vector3 *hPos, vector3 *accels, double *mass){
	int col = (blockDim.x * blockIdx.x) + threadIdx.x;
	int row = (blockDim.y * blockIdx.y) + threadIdx.y;
	int ind = (NUMENTITIES * row) + col; 
	int i = row; 
	int j = col; 
	if(ind < NUMENTITIES * NUMENTITIES){
		if (i == j){
			FILL_VECTOR(accels[ind], 0, 0, 0);
		} else{
			vector3 dist; 
			for (int k = 0; k < 3; k++){
				dist[k] = hPos[i][k] - hPos[j][k];
			}
			double mag_sq = dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2];
			double mag = sqrt(mag_sq);
			double acc = -1 * GRAV_CONSTANT * mass[j] /mag_sq;
			FILL_VECTOR(accels[ind], acc * dist[0] / mag, acc * dist[1] / mag, acc * dist[2]/ mag);
		}
	}
} 