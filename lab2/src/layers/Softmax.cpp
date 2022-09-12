#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "Softmax.h"
#include <algorithm>
<<<<<<< HEAD
#include <math.h>
=======
>>>>>>> 236e96b6ce27fe0965d388c08138bf3f8786a945


namespace ML {
    // --- Begin Student Code ---

    // Compute the convultion for the layer data
    void SoftMaxLayer::computeNaive(const LayerData &dataIn) const {
        // TODO: Your Code Here...
        Array1D_fp32 inputData = dataIn.getData<Array1D_fp32>();
        Array1D_fp32 outputData = getOutputData().getData<Array1D_fp32>();
        int X = getInputParams().dims[0];
        // printf("%d\n\r", I);
<<<<<<< HEAD

        fp32 sum;
        for(int i = 0; i < X; ++i) {
            sum += inputData[i];
            // printf("%lf\n\r", sum);
        }

        for(int i = 0; i < X; ++i) {
            outputData[i] = exp(inputData[i]) / sum;
            // printf("%lf\n\r", outputData[i]);
        }

=======
        for(int i = 0; i < X; ++i) {

        }

        
>>>>>>> 236e96b6ce27fe0965d388c08138bf3f8786a945
        printf("Softmax Finished\n\r");
    }


    // Compute the convolution using threads
    void SoftMaxLayer::computeThreaded(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the convolution using a tiled approach
    void SoftMaxLayer::computeTiled(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the convolution using SIMD
    void SoftMaxLayer::computeSIMD(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }
};