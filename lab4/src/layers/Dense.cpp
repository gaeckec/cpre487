#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "Dense.h"
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

namespace ML {
    // --- Begin Student Code ---

    // Compute the full connected for the layer data
    void DenseLayer::computeNaive(const LayerData &dataIn) const {
        // TODO: Your Code Here...
       
        //Define Parameters
        int input_height = getInputParams().dims[0];
        int num_input_channels = getWeightParams().dims[1];

        //Probably have a variable assicated with this        
        int batch_size = 1;

        //Preset LayerDate Tpye
        LayerData Weight_data = getWeightData();
        LayerData Bias_data = getBiasData();
        LayerData Output_data = getOutputData();

        //Map values to memory
        Array2D_fp32 denseWeightData = Weight_data.getData<Array2D_fp32>();
        Array1D_fp32 denseBiasData = Bias_data.getData<Array1D_fp32>();
        Array1D_fp32 denseInputData = dataIn.getData<Array1D_fp32>();
        Array1D_fp32 denseOutputData = Output_data.getData<Array1D_fp32>();

        Array2D_i8 denseWeightData_q;
        Array1D_i32 denseBiasData_q;
        Array1D_ui8 denseInputData_q;
        Array1D_i32 denseOutputData_q;

        //predeclair variables
        int n,m,h;
        int x,y,z,w;

        //Create Scales for quantization
        fp32 weights_max = denseWeightData[0][0]; 
        fp32 inputs_max = denseInputData[0];

        for(x = 0; x < getWeightParams().dims[0]; x++) {
            for(y = 0; y < getWeightParams().dims[1]; y++) {
                weights_max = std::max(weights_max, std::abs(denseWeightData[x][y]));
            }
        }
        for(x = 0; x < getInputParams().dims[0]; x++) {
            inputs_max = std::max(inputs_max, std::abs(denseInputData[x]));
        }

        int8_t scale_weight = 127 / weights_max;
        uint8_t scale_input = 255 / inputs_max;
        int32_t scale_biases = scale_input * scale_weight;

        //Quantize inputs, weights, and biases with scales
        for(x = 0; x < getWeightParams().dims[0]; x++) {
            for(y = 0; y < getWeightParams().dims[1]; y++) {
                denseWeightData_q[x][y] = std::round(denseWeightData[x][y] * scale_weight);
            }
        }
        for(x = 0; x < getInputParams().dims[0]; x++) {
            denseInputData_q[x] = std::round(denseInputData[x] * scale_input);
        }
        for(x = 0; x < getBiasParams().dims[0]; x++) {
            denseBiasData_q[x] = std::round(denseBiasData[x] * scale_biases);
        }

        //loop through and perform the opperation
        for(n = 0; n < batch_size; n++){
            for(m = 0; m < num_input_channels; m++){
                for(h = 0; h < input_height; h++){                                    
                    denseOutputData_q[m] += denseInputData_q[h] * denseWeightData_q[h][m];                                
                }
                denseOutputData_q[m] += denseBiasData_q[m];

                if(denseOutputData_q[m] < 0) { denseOutputData_q[m] = 0.0; }
            } 
        }

        //De-quantize output values back to fp32
        for(x = 0; x < getOutputParams().dims[0]; x++) {
            denseOutputData[x] = denseOutputData_q[x] / scale_biases;
        }
    }
    


    // Compute the Dense using threads
    void DenseLayer::computeThreaded(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        std::cout << "\n\n\nhello Dense thread\n\n\n";

    }

    // Compute the Dense using a tiled approach
    void DenseLayer::computeTiled(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        std::cout << "\n\n\nhello Dense Tiled\n\n\n";

    }


    // Compute the Dense using SIMD
    void DenseLayer::computeSIMD(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        std::cout << "\n\n\nhello Dense SIMD\n\n\n";

    }
};