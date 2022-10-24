#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "Convolutional.h"
#include <chrono>
#include <cmath>
#include <algorithm>

using namespace std::chrono;
using namespace std;

namespace ML {
    // --- Begin Student Code ---

    // Compute the convultion for the layer data
    void ConvolutionalLayer::computeNaive(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        bool debug = false;
        int quant_length = getQuantLength();

        if (debug) {
            
        }
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

        //predeclair variables
        int n,m,p,q,c,r,s;
        int x,y,z,w;

        //Preset LayerDate Tpye
        LayerData Weight_data = getWeightData();
        LayerData Bias_data = getBiasData();
        LayerData Output_data = getOutputData();

        //Map values to memory
        Array4D_fp32 convWeightData = Weight_data.getData<Array4D_fp32>();
        Array1D_fp32 convBiasData = Bias_data.getData<Array1D_fp32>();
        Array3D_fp32 convInputData = dataIn.getData<Array3D_fp32>();
        Array3D_fp32 convOutputData = Output_data.getData<Array3D_fp32>();
        fp32 convOutputData_2[output_height][output_width][num_filter_channels];
            
        Array4D_i8 convWeightData_q = getWeightData_q().getData<Array4D_i8>();
        Array1D_i32 convBiasData_q = getBiasData_q().getData<Array1D_i32>();
        Array3D_ui8 convInputData_q = getInputData_q().getData<Array3D_ui8>();
        Array3D_i32 convOutputData_q = getOutputData_q().getData<Array3D_i32>();

        //Debugging Var - var[x][y][z]
        int input_x, input_y;

        fp32 weights_max = 0;
        fp32 inputs_max = 0;
        

        if(isQuantized()) {
            weights_max = getWeightsMax();
            inputs_max = getInputsMax();
            //printf("Weight Max Val: %lf\n\rInput Max Val: %lf\n\r", weights_max, inputs_max);
        }

        if(debug){
            fp32 weights_max = convWeightData[0][0][0][0];
            fp32 inputs_max = convInputData[0][0][0];
            for(x = 0; x < getWeightParams().dims[0]; x++) {
                for(y = 0; y < getWeightParams().dims[1]; y++) {
                    for(z = 0; z < getWeightParams().dims[2]; z++) {
                        for(w = 0; w < getWeightParams().dims[3]; w++) {
                            weights_max = std::max(weights_max, std::abs(convWeightData[x][y][z][w]));
                        }
                    }
                }
            }
            for(x = 0; x < getInputParams().dims[0]; x++) {
                for(y = 0; y < getInputParams().dims[1]; y++) {
                    for(z = 0; z < getInputParams().dims[2]; z++) {
                        inputs_max = std::max(inputs_max, std::abs(convInputData[x][y][z]));
                    }
                }
            }

            printf("Weight Max Val: %lf\n\rInput Max Val: %lf\n\r", weights_max, inputs_max);
        }

        //Create Scales for quantization
        i8 scale_weight = std::pow(2,quant_length-1)-1 / weights_max;
        ui8 scale_input = std::pow(2,quant_length)-1 / inputs_max;
        i32 scale_biases = scale_input * scale_weight;
        //printf("scale_weight: %d\n\rscale_input: %d\n\rscale_biases: %d\n\r", scale_weight, scale_input, scale_biases);

        if(isQuantized()) {

            //Quantize inputs, weights, and biases with scales
            //printf("Starting Quantization...\n\r");
            for(x = 0; x < getWeightParams().dims[0]; x++) {
                for(y = 0; y < getWeightParams().dims[1]; y++) {
                    for(z = 0; z < getWeightParams().dims[2]; z++) {
                        for(w = 0; w < getWeightParams().dims[3]; w++) {
                            convWeightData_q[x][y][z][w] = static_cast<i8>(std::round(convWeightData[x][y][z][w] * scale_weight));
                            //printf("convWeightData_q: %d\n\r", convWeightData_q[x][y][z][w]);
                        }
                    }
                }
            }

            for(x = 0; x < getInputParams().dims[0]; x++) {
                for(y = 0; y < getInputParams().dims[1]; y++) {
                    for(z = 0; z < getInputParams().dims[2]; z++) {
                        convInputData_q[x][y][z] = static_cast<ui8>(std::round(convInputData[x][y][z] * scale_input));
                        //printf("convInputData_q: %d\n\r", convInputData_q[x][y][z]);
                    }
                }
            }
            for(x = 0; x < getBiasParams().dims[0]; x++) {
                convBiasData_q[x] = static_cast<i32>(std::round(convBiasData[x] * scale_biases));
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

                                        convOutputData_q[q][p][m] += convInputData_q[input_x][input_y][c] * convWeightData_q[s][r][c][m];
                                        //printf("convOutputData_q: %d\n\r", convOutputData_q[q][p][m]);
                                    }
                                }
                            } 
                            convOutputData_q[q][p][m] += convBiasData_q[m];
                        }
                    } 
                }
            }

            // printf("\nDim 0 = %d, Dim 1 = %d, Dim 2 = %d\n", output_height, output_width, num_filter_channels);
            //printf("Convolution Finished\n\r");
            auto end = high_resolution_clock::now();
            auto total = duration_cast<microseconds>(end - start);
            printf("Convolution Finished in %d us\n\r", total.count());

            //De-quantize output values back to fp32
            for(x = 0; x < getOutputParams().dims[0]; x++) {
                for(y = 0; y < getOutputParams().dims[1]; y++) {
                    for(z = 0; z < getOutputParams().dims[2]; z++) {
                        //printf("%d\n\r", convOutputData_q[x][y][z]);
                        convOutputData[x][y][z] = convOutputData_q[x][y][z] / static_cast<fp32>(scale_biases);
                        if(convOutputData[x][y][z] < 0) { convOutputData[x][y][z] = 0.0f; }
                    }
                }
                //printf("%lf\n\r", convOutputData[x][0][0]);
            }
        } 
        else {  
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

                                        convOutputData_2[p][q][m] += convInputData[input_x][input_y][c] * convWeightData[s][r][c][m];
                                    }
                                }
                            } 
                            convOutputData_2[p][q][m] += convBiasData[m];
                            convOutputData_2[p][q][m] = std::max(float(0), convOutputData_2[p][q][m]);
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

        // fp32 err = 0;
        // for(x = 0; x < output_height; ++x) {
        //     for(y = 0; y < output_width; ++y) {
        //         for(z = 0; z < num_filter_channels; ++z) {
        //             err += (convOutputData[x][y][z] - convOutputData_2[x][y][z]);
        //         }
        //     }
        //     //printf("DE Quantized Value: %lf, Original Value: %lf\n\r", convOutputData[0][x][x], convOutputData_2[0][x][x]);
        //     // //printf("DE Quantized Value: %lf\n\r", convOutputData[0][x][x]);
        // }
        // printf("err: %lf\n\r", err);
    }


    // Compute the convolution using threads
    void ConvolutionalLayer::computeThreaded(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        std::cout << "\n\n\nhello Convolution thread\n\n\n";

    }


    // Compute the convolution using a tiled approach
    void ConvolutionalLayer::computeTiled(const LayerData &dataIn) const {
        // TODO: Your Code Here...
        std::cout << "\nhello Convolution Tiled\n";
        int block_size = 8;
        bool tiled = false;

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
        int stride_size = 1;

        int input_x, input_y;

        output_height = ((input_height - filter_height + stride_size) / stride_size);
        output_width = ((input_width - filter_width + stride_size) / stride_size);

        //predeclair variables
        int n,m,mm,p,pp,q,qq,c,cc,r,rr,s,ss;
        int x,y,z,w;

        //Preset LayerDate Tpye
        LayerData Weight_data = getWeightData();
        LayerData Bias_data = getBiasData();
        LayerData Output_data = getOutputData();

        //Map values to memory
        Array4D_fp32 convWeightData = Weight_data.getData<Array4D_fp32>();
        Array1D_fp32 convBiasData = Bias_data.getData<Array1D_fp32>();
        Array3D_fp32 convInputData = dataIn.getData<Array3D_fp32>();
        Array3D_fp32 convOutputData = Output_data.getData<Array3D_fp32>();

        //Start time for profiling
        auto start = high_resolution_clock::now();

        if(!tiled) {
            for(n = 0; n < batch_size; n++){
                for(p = 0; p < output_height; p++){
                    for(q = 0; q < output_width; q++){
                        for(m = 0; m < num_filter_channels; m++) {
                            // std::cout << "\n" << p << q << m << "\n";
                            for(r = 0; r < filter_height; r++){
                                input_x = stride_size * p + r;

                                for(s = 0; s < filter_width; s++) { 
                                    input_y = stride_size * q + s;  

                                    for(c = 0; c < num_input_channels; c++) {
                                        // std::cout << input_x << input_y << c << "\n";
                                        convOutputData[p][q][m] += convInputData[input_x][input_y][c] * convWeightData[s][r][c][m];
                                    }
                                }
                            } 
                            convOutputData[p][q][m] += convBiasData[m];
                            convOutputData[p][q][m] = std::max(float(0), convOutputData[p][q][m]);
                        }
                    } 
                }
            }
        }
        else {
            for(n = 0; n < batch_size; n++){
                for(p = 0; p < output_height; p+=block_size){
                    for(q = 0; q < output_width; q+=block_size){
                        for(pp = p; pp < p+block_size && pp < output_height; pp++) {
                            for(qq = q; qq < q+block_size && qq < output_width; qq++) {
                                for(m = 0; m < num_filter_channels; m++) {
                                    // std::cout << "\n" << pp << " " << qq << " " << m << "\n";
                                    for(r = 0; r < filter_height; r++) {
                                        input_x = stride_size * pp + r;

                                        for(s = 0; s < filter_width; s++) { 
                                            input_y = stride_size * qq + s;  

                                            for(c = 0; c < num_input_channels; c++) {
                                                // std::cout << input_x << " " << input_y << " " << c << "\n";
                                                convOutputData[pp][qq][m] += convInputData[input_x][input_y][c] * convWeightData[s][r][c][m];
                                            }
                                        }
                                    }
                                    convOutputData[pp][qq][m] += convBiasData[m];
                                    convOutputData[pp][qq][m] = std::max(float(0), convOutputData[pp][qq][m]); 
                                }
                            }
                        }
                    } 
                }
            }
        }

        auto end = high_resolution_clock::now();
        auto total = duration_cast<microseconds>(end - start);
        printf("Convolution Finished in %d us\n\r", total.count());
    }


    // Compute the convolution using SIMD
    void ConvolutionalLayer::computeSIMD(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        std::cout << "\n\n\nhello Convolution SIMD\n\n\n";

    }
};