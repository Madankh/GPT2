/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

// all the individual layers forward and backward passes
void encoder_forward(float* out, int* inp, float* wte, float* wpe,
                    int B, int T, int C){
    for(int b = 0; b < B; b++){
        for(int t = 0; t < T; t++){
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            // get the index of the token inp[b,t]
            int ix = inp[b * T + t];
            
            float* wte_ix = wte + ix * C;
            float* wpe_t = wpe + t * C;

            for(int c = 0; c < C; c++){
                out_bt[c] = wte_ix[c] + wpe_t[c];
            }

        }
    }
}


void encoder_backward(float* dwte, float *dwpe,
                     float* dout, int* inp, 
                     int B, int T, int C){
    for(int b = 0; b<B; b++){
        for(int t = 0; t<T; t++){

            float* dout_bt = dout * b * T * C + t * C;
            int ix = inp[b * T + t]
            float* dwte_ixx = dwte + ix * C;
            float* dwpe_t = dwpe + t * C
            for(int i=0; i<C; i++){
                float d = dout_bt[i]
                dwte_ix[i] += d;
                dwpe_ix[i] += d;
            }
        }
    }
                
}
