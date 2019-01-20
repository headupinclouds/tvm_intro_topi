// https://github.com/dmlc/tvm/blob/master/apps/howto_deploy/cpp_deploy.cc

/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Example code on load and run TVM module.s
 * \file cpp_deploy_example.cc
 */
#include <cstdio>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include <random>
#include <iostream>

inline int64_t sum(int64_t values[], int64_t num)
{
    int64_t value = values[0];
    for(int64_t i = 1; i < num; i++)
    {
        value *= values[i];
    }
    return value;
}
void Verify(tvm::runtime::Module mod, std::string fname)
{
  // Get the function from the module.
  tvm::runtime::PackedFunc f = mod.GetFunction(fname);
  CHECK(f != nullptr);
  // Allocate the DLPack data structures.
  //
  // Note that we use TVM runtime API to allocate the DLTensor in this example.
  // TVM accept DLPack compatible DLTensors, so function can be invoked
  // as long as we pass correct pointer to DLTensor array.
  //
  // For more information please refer to dlpack.
  // One thing to notice is that DLPack contains alignment requirement for
  // the data pointer and TVM takes advantage of that.
  // If you plan to use your customized data container, please
  // make sure the DLTensor you pass in meet the alignment requirement.
  //
  DLTensor* data;
  DLTensor* kern;
  DLTensor* relu;

  DLTensor* gamma;
  DLTensor* beta;
  DLTensor* mean;
  DLTensor* var;

  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;

  // If you receive an error stating "device_type
#if defined(TVM_OPENCL_RUNTIME)
    constexpr int device_type = kDLOpenCL;
#elif defined(TVM_OPENGL_RUNTIME)
    constexpr int device_type = 11; // kDLOpenGL;
#elif defined(TVM_VULKAN_RUNTIME)
    constexpr int device_type = kDLVulkan;
#elif defined(TVM_METAL_RUNTIME)
    constexpr int device_type = kDLMetal;
#elif defined(TVM_CUDA_RUNTIME)
    constexpr int device_type = kDLGPU;
#elif defined(TVM_CPU_RUNTIME)
    constexpr int device_type = kDLCPU;
#else
#  error Must define a valid TVM_<KIND>_RUNTIME flag, see CMakeLists.txt
#endif

  int device_id = 0;

  int64_t data_shape[4] = { 1, 3, 224, 224 };
  int64_t data_size = sum(data_shape, 4);
  TVMArrayAlloc(data_shape, 4, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &data);

  int64_t kern_shape[4] = { 64, 3, 7, 7, };
  int64_t kern_size = sum(kern_shape, 4);
  TVMArrayAlloc(kern_shape, 4, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &kern);

  int64_t relu_shape[4] = { 1, 64, 112, 112 };
  int64_t relu_size = sum(relu_shape, 4);
  TVMArrayAlloc(relu_shape, 4, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &relu);

  int64_t bn_shape[1] = { 64 };
  int64_t bn_size = sum(bn_shape, 1);
  TVMArrayAlloc(bn_shape, 1, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &gamma);
  TVMArrayAlloc(bn_shape, 1, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &beta);
  TVMArrayAlloc(bn_shape, 1, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &mean);
  TVMArrayAlloc(bn_shape, 1, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &var);

  std::vector<float> data_mem(data_size);
  std::vector<float> kern_mem(kern_size);
  std::vector<float> relu_mem(relu_size);

  std::vector<float> gamma_mem(64);
  std::vector<float> beta_mem(64);
  std::vector<float> mean_mem(64);
  std::vector<float> var_mem(64);

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  for (int i = 0; i < data_size; ++i)
  {
    data_mem[i] = dis(gen);
    std::cout << "data[" << i << "] = " << data_mem[i] << std::endl;
  }

  TVMArrayCopyFromBytes(data, data_mem.data(), data_size * sizeof(float));

  for (int i = 0; i < kern_size; ++i)
  {
    kern_mem[i] = dis(gen);
    std::cout << "kern[" << i << "] = " << kern_mem[i] << std::endl;
  }

  TVMArrayCopyFromBytes(kern, kern_mem.data(), kern_size * sizeof(float));

  for (int i = 0; i < 64; ++i)
  {
    gamma_mem[i] = 1.f;
    beta_mem[i] = 0.f;
    mean_mem[i] = 0.f;
    var_mem[i] = 1.f;
  }
  TVMArrayCopyFromBytes(gamma, gamma_mem.data(), 64 * sizeof(float));
  TVMArrayCopyFromBytes(beta, beta_mem.data(), 64 * sizeof(float));
  TVMArrayCopyFromBytes(mean, mean_mem.data(), 64 * sizeof(float));
  TVMArrayCopyFromBytes(var, var_mem.data(), 64 * sizeof(float));

  // Invoke the function
  // PackedFunc is a function that can be invoked via positional argument.
  // The signature of the function is specified in tvm.build
  auto result = f(data, kern, gamma, beta, mean, var, relu);

  TVMArrayCopyToBytes(relu, relu_mem.data(), relu_size * sizeof(float));

  // Print out the reluut
  for (int i = 0; i < relu_size; ++i)
  {
    std::cout << "relu[" << i << "]=" << relu_mem[i] << std::endl;
  }

  LOG(INFO) << "Free arrays ...";      
  TVMArrayFree(data);
  TVMArrayFree(kern);

  TVMArrayFree(gamma);
  TVMArrayFree(beta);
  TVMArrayFree(mean);
  TVMArrayFree(var);
  
  TVMArrayFree(relu);
      
  LOG(INFO) << "Finish verification...";
}

int main(int argc, char **argv) try
{
  std::cout << "argv[1] " << argv[1] << " argv[2] " << argv[2] << std::endl;
  tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile(argv[1]);
  LOG(INFO) << "Verify dynamic loading from intro_topi.so";
  Verify(mod_dylib, argv[2]);
  LOG(INFO) << "OK HERE";
  
  throw std::runtime_error("exit"); // workaround
}
catch(std::exception &e)
{
    std::cerr << "exception " << e.what() << std::endl;
    abort();
}
