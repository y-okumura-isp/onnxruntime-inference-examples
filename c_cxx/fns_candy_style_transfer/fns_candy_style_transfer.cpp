#include <assert.h>
#include <png.h>
#include <stdio.h>

#include "onnxruntime_cxx_api.h"

#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

#define tcscmp strcmp

/**
 * convert input from HWC format to CHW format
 * \param input A single image. The byte array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A float array. should be freed by caller after use
 * \param output_count Array length of the `output` param
 */
static void hwc_to_chw(const png_byte* input, size_t h, size_t w, float** output, size_t* output_count) {
  size_t stride = h * w;
  *output_count = stride * 3;
  float* output_data = (float*)malloc(*output_count * sizeof(float));
  for (size_t i = 0; i != stride; ++i) {
    for (size_t c = 0; c != 3; ++c) {
      output_data[c * stride + i] = input[i * 3 + c];
    }
  }
  *output = output_data;
}

/**
 * convert input from CHW format to HWC format
 * \param input A single image. This float array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A byte array. should be freed by caller after use
 */
static void chw_to_hwc(const float* input, size_t h, size_t w, png_bytep* output) {
  size_t stride = h * w;
  png_bytep output_data = (png_bytep)malloc(stride * 3);
  for (int c = 0; c != 3; ++c) {
    size_t t = c * stride;
    for (size_t i = 0; i != stride; ++i) {
      float f = input[t + i];
      if (f < 0.f || f > 255.0f) f = 0;
      output_data[i * 3 + c] = (png_byte)f;
    }
  }
  *output = output_data;
}

/**
 * \param out should be freed by caller after use
 * \param output_count Array length of the `out` param
 */
static int read_png_file(const char* input_file, size_t* height, size_t* width, float** out, size_t* output_count) {
  png_image image; /* The control structure used by libpng */
  /* Initialize the 'png_image' structure. */
  memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  if (png_image_begin_read_from_file(&image, input_file) == 0) {
    return -1;
  }
  png_bytep buffer;
  image.format = PNG_FORMAT_BGR;
  size_t input_data_length = PNG_IMAGE_SIZE(image);
  if (input_data_length != 720 * 720 * 3) {
    printf("input_data_length:%zd\n", input_data_length);
    return -1;
  }
  buffer = (png_bytep)malloc(input_data_length);
  memset(buffer, 0, input_data_length);
  if (png_image_finish_read(&image, NULL /*background*/, buffer, 0 /*row_stride*/, NULL /*colormap*/) == 0) {
    return -1;
  }
  hwc_to_chw(buffer, image.height, image.width, out, output_count);
  free(buffer);
  *width = image.width;
  *height = image.height;
  return 0;
}

/**
 * \param tensor should be a float tensor in [N,C,H,W] format
 */
static int write_tensor_to_png_file(Ort::Value & tensor, const char* output_file) {
  Ort::TensorTypeAndShapeInfo shape_info = tensor.GetTensorTypeAndShapeInfo();
  auto dims = shape_info.GetShape();
  assert(dims.size() == 4);
  auto f = tensor.GetTensorMutableData<float>();

  // write to png file
  png_bytep model_output_bytes;
  png_image image;
  memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  image.format = PNG_FORMAT_BGR;
  image.height = (png_uint_32)dims[2];
  image.width = (png_uint_32)dims[3];
  chw_to_hwc(f, image.height, image.width, &model_output_bytes);
  int ret = 0;
  if (png_image_write_to_file(&image, output_file, 0 /*convert_to_8bit*/, model_output_bytes, 0 /*row_stride*/,
                              NULL /*colormap*/) == 0) {
    printf("write to '%s' failed:%s\n", output_file, image.message);
    ret = -1;
  }
  free(model_output_bytes);
  return ret;
}

static void usage() { printf("usage: <model_path> <input_file> <output_file> [cpu|cuda|dml] \n"); }

int run_inference(Ort::Session & session, const std::string & input_file, const std::string & output_file) {
  size_t input_height;
  size_t input_width;
  float* model_input;
  size_t model_input_ele_count;
  const char* output_file_p = output_file.c_str();
  const char* input_file_p = input_file.c_str();

  // load png data to model_input
  if (read_png_file(input_file_p, &input_height, &input_width, &model_input, &model_input_ele_count) != 0) {
    return -1;
  }
  if (input_height != 720 || input_width != 720) {
    printf("please resize to image to 720x720\n");
    free(model_input);
    return -1;
  }

  const int64_t input_shape[] = {1, 3, 720, 720};
  const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);

  // ステップ1: GPU 転送は内部に任せる
  // 最終: IO を GPU メモリのポインタとする
  int device_id = 0;
  Ort::IoBinding io_binding{session};
  std::cout << "hello" << std::endl;
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::cout << "device_id: " << memory_info.GetDeviceId() << std::endl;
  auto input_tensor = Ort::Value::CreateTensor<float>(
      memory_info,
      model_input,
      model_input_ele_count,
      input_shape,
      input_shape_len);
  io_binding.BindInput("inputImage", input_tensor);

  auto out_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  auto outlen = std::accumulate(input_shape, input_shape + input_shape_len, 1, std::multiplies<float>());
  std::vector<float> out_data(outlen);
  auto output_tensor = Ort::Value::CreateTensor<float>(
      out_memory_info,
      out_data.data(),
      out_data.size(),
      input_shape,
      input_shape_len);
  io_binding.BindOutput("outputImage", output_tensor);

  session.Run(Ort::RunOptions(),
              io_binding);

  int ret = 0;
  if (write_tensor_to_png_file(output_tensor, output_file_p) != 0) {
    ret = -1;
  }
  free(model_input);
  return ret;
}

void verify_input_output_count(const Ort::Session & session) {
  assert(session.GetInputCount() == 1);
  assert(session.GetOutputCount() == 1);
}

int enable_cuda(Ort::SessionOptions & session_options) {
  OrtCUDAProviderOptions o;
  o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  o.gpu_mem_limit = SIZE_MAX;
  session_options.AppendExecutionProvider_CUDA(o);
  return 0;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    usage();
    return -1;
  }

  std::string model_path{argv[1]};
  std::string input_file{argv[2]};
  std::string output_file{argv[3]};
  // By default it will try CUDA first. If CUDA is not available, it will run all the things on CPU.
  // But you can also explicitly set it to DML(directml) or CPU(which means cpu-only).
  std::string execution_provider;
  if (argc >= 5) {
    execution_provider = argv[4];
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions session_options;

  int ret = 0;

  if (!execution_provider.empty()) {
    if (execution_provider == "cpu") {
      // Nothing; this is the default
    } else if (execution_provider == "dml") {
      puts("DirectML is not enabled in this build.");
      return -1;
    } else {
      usage();
      puts("Invalid execution provider option.");
      return -1;
    }
  } else {
    printf("Try to enable CUDA first\n");
    ret = enable_cuda(session_options);
    if (ret) {
      fprintf(stderr, "CUDA is not available\n");
    } else {
      printf("CUDA is enabled\n");
    }
  }

  Ort::Session session(env, model_path.c_str(), session_options);
  verify_input_output_count(session);
  ret = run_inference(session, input_file, output_file);
  if (ret != 0) {
    fprintf(stderr, "fail\n");
  }

  return ret;
}
