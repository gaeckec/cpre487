#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "Flatten.h"
#include <algorithm>


namespace ML {
    // --- Begin Student Code ---

    // Compute the convultion for the layer data
    void FlattenLayer::computeNaive(const LayerData &dataIn) const {
        // TODO: Your Code Here...
        Array3D_fp32 inputData = dataIn.getData<Array3D_fp32>();

        int X = getInputParams().dims[0];
        int Y = getInputParams().dims[1];
        int Z  = getInputParams().dims[2];

        Array1D_fp32 outputData = getOutputData().getData<Array1D_fp32>();

        for(int i = 0; i < X; ++i) {
            for(int j = 0; j < Y; ++j) {
                for(int k = 0; k < Z; ++k) {
                    outputData[(X*Y*i) + (Y*j) + k] = inputData[i][j][k];
                }
            }
        }
        printf("Flatten Finished\n\r");
    }


    // Compute the convolution using threads
    void FlattenLayer::computeThreaded(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the convolution using a tiled approach
    void FlattenLayer::computeTiled(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the convolution using SIMD
    void FlattenLayer::computeSIMD(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }
};