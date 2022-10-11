#pragma once

#include "Layer.h"
#include "../Utils.h"
#include "../Types.h"

namespace ML {
    class DenseLayer : public Layer {
        public:
        DenseLayer(const LayerParams inParams, const LayerParams outParams,
                               const LayerParams weightParams, const LayerParams biasParams, const fp32 weights_max, const fp32 inputs_max, const bool quantize)
                : Layer(inParams, outParams, LayerType::DENSE), weightParam(weightParams), weightData(weightParams),
                  biasParam(biasParams), biasData(biasParams), weights_max(weights_max), inputs_max(inputs_max), quantize(quantize) {}

        // Getters
            const LayerParams& getWeightParams() const { return weightParam; }
            const LayerParams& getBiasParams() const { return biasParam; }
            const LayerData& getWeightData() const { return weightData; }
            const LayerData& getBiasData() const { return biasData; }
            const fp32& getWeightsMax() const { return weights_max; }
            const fp32& getInputsMax() const { return inputs_max; }
            const bool& isQuantized() const { return quantize; }

            // Allocate all resources needed for the layer & Load all of the required data for the layer
            template<typename T>
            void allocateLayer() {
                Layer::allocateOutputBuffer<Array3D<T>>();
                weightData.loadData<Array4D<T>>();
                biasData.loadData<Array1D<T>>();
            }

            // Fre all resources allocated for the layer
            template<typename T>
            void freeLayer() {
                Layer::freeOutputBuffer<Array3D<T>>();
                weightData.freeData<Array4D<T>>();
                biasData.freeData<Array1D<T>>();
                weights_max = 0;
                inputs_max = 0;
            }

            // Virtual functions
            virtual void computeNaive(const LayerData &dataIn) const override;
            virtual void computeThreaded(const LayerData &dataIn) const override;
            virtual void computeTiled(const LayerData &dataIn) const override;
            virtual void computeSIMD(const LayerData &dataIn) const override;

        private:
            LayerParams weightParam;
            LayerData weightData;

            LayerParams biasParam;
            LayerData biasData;

            fp32 weights_max;
            fp32 inputs_max;
            bool quantize;
    };
}