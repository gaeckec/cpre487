#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "Convolutional.h"
#include <chrono>

using namespace std::chrono;

namespace ML {
    // --- Begin Student Code ---

    // Compute the convultion for the layer data
    void ConvolutionalLayer::computeNaive(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        bool debug = false;

        if (debug)
            std::cout << "\n\n\n";

        // std::cout << getInputParams().dims[3];

        //Define Parameters
        int input_height = getInputParams().dims[0];
        int input_width = getInputParams().dims[1];
        int num_input_channels = getInputParams().dims[2];
        int filter_height = getWeightParams().dims[0];
        int filter_width = getWeightParams().dims[1];
        int num_filter_channels = getWeightParams().dims[3];
        int output_height, output_width;

        //Probably have a variable assicated with this        
        int batch_size = 1;
        int stide_size = 1;

        output_height = ((input_height - filter_height + stide_size) / stide_size);
        output_width = ((input_width - filter_width + stide_size) / stide_size);

        //check
        if(debug){
            std::cout << "Input Height = " << input_height << ", Input Width = " << input_width << ", num_input_channels " << num_input_channels << "\n"
                    << "Filter Height = " << filter_height << ", Filter Width = " << filter_width << ", num_filter_channels " << num_filter_channels << "\n"
                    << "Output Height = " << output_height << ", Output Width = " << output_width << ", num_output_channels " << num_filter_channels << "\n"
                    << "\n\n";
        }

        //Preset LayerDate Tpye
        LayerData Weight_data = getWeightData();
        LayerData Bias_data = getBiasData();
        LayerData Output_data = getOutputData();

        //Map values to memory
        Array4D_fp32 convWeightData = Weight_data.getData<Array4D_fp32>();
        Array1D_fp32 convBiasData = Bias_data.getData<Array1D_fp32>();
        Array3D_fp32 convInputData = dataIn.getData<Array3D_fp32>();
        Array3D_fp32 convOutputData = Output_data.getData<Array3D_fp32>();

        Array3D_i8 convWeightData_q;
        Array4D_ui8 convInputData_q;
        Array1D_i32 convBiasData_q;

        //predeclair variables
        int n,m,p,q,c,r,s;
        int x,y,z,w;
        //Debugging Var - var[x][y][z]
        int input_x, input_y;

        //Create Scales for quantization
        fp32 weights_max = convWeightData[0][0][0][0]; 
        fp32 inputs_max = convInputData[0][0][0];

        for(x = 0; x < getWeightParams().dims[0]; x++) {
            for(y = 0; y < getWeightParams().dims[1]; y++) {
                for(z = 0; z < getWeightParams().dims[2]; z++) {
                    for(w = 0; w < getWeightParams().dims[4]; w++) {
                        weights_max = std::max(weights_max, std::abs(convWeightData[x][y][z][w]));
                    }
                }
            }
        }
        for(x = 0; x < getWeightParams().dims[0]; x++) {
            for(y = 0; y < getWeightParams().dims[1]; y++) {
                for(z = 0; z < getWeightParams().dims[2]; z++) {
                    inputs_max = std::max(inputs_max, std::abs(convInputData[x][y][z]));
                }
            }
        }

        uint8_t scale_weight = 127 / weights_max;
        int8_t scale_input = 255 / inputs_max;
        int32_t scale_biases = scale_input * scale_weight;

        //Quantize inputs, weights, and biases with scales
        for(x = 0; x < getWeightParams().dims[0]; x++) {
            for(y = 0; y < getWeightParams().dims[1]; y++) {
                for(z = 0; z < getWeightParams().dims[2]; z++) {
                    for(w = 0; w < getWeightParams().dims[4]; w++) {
                        convWeightData_q[x][y][z][w] = std::round(convWeightData[x][y][z][w] * scale_weight);
                    }
                }
            }
        }
        for(x = 0; x < getWeightParams().dims[0]; x++) {
            for(y = 0; y < getWeightParams().dims[1]; y++) {
                for(z = 0; z < getWeightParams().dims[2]; z++) {
                    convInputData_q[x][y][z] = std::round(convInputData[x][y][z] * scale_input);
                }
            }
        }
        for(x = 0; x < getBiasParams().dims[0]; x++) {
            convBiasData_q[x] = std::round(convBiasData[x] * scale_biases);
        }


        //Start time for profiling
        auto start = high_resolution_clock::now();

        for(n = 0; n < batch_size; n++){
            for(m = 0; m < num_filter_channels; m++){
                for(p = 0; p < output_height; p++){
                    for(q = 0; q < output_width; q++){
                       for(c = 0; c < num_input_channels; c++){
                            for(r = 0; r < filter_height; r++){
                                for(s = 0; s < filter_width; s++){
                                    
                                    input_x = stide_size * q + s;
                                    input_y = stide_size * p + r;

                                    convOutputData[q][p][m] += convInputData[input_x][input_y][c] * convWeightData[s][r][c][m];
                                }
                            }
                        } 
                        convOutputData[q][p][m] += convBiasData[m];

                        if(convOutputData[q][p][m] < 0){
                            convOutputData[q][p][m] = 0.0f;
                        }
                        // convOutputData[q][p][m] += convBiasData[m];
                    }
                } 
            }
        }

        // printf("\nDim 0 = %d, Dim 1 = %d, Dim 2 = %d\n", output_height, output_width, num_filter_channels);
        //printf("Convolution Finished\n\r");
        auto end = high_resolution_clock::now();

        auto total = duration_cast<microseconds>(end - start);
        printf("Convolution Finished in %d us\n\r", total.count());
    }


    // Compute the convolution using threads
    void ConvolutionalLayer::computeThreaded(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        std::cout << "\n\n\nhello Convolution thread\n\n\n";

    }


    // Compute the convolution using a tiled approach
    void ConvolutionalLayer::computeTiled(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        std::cout << "\n\n\nhello Convolution Tiled\n\n\n";

    }


    // Compute the convolution using SIMD
    void ConvolutionalLayer::computeSIMD(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        std::cout << "\n\n\nhello Convolution SIMD\n\n\n";

    }
};