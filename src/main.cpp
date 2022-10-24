#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <cassert>
#include <fstream>
#include <iterator>

cl_device_id
get_gpu_device_id(cl_int& err_code)
{
    /* search thru platforms (IDKs) */
    cl_platform_id platforms[64];
    uint platformCount;
    err_code = clGetPlatformIDs(64, platforms, &platformCount);
    if (err_code != CL_SUCCESS)
        return nullptr;

    /* search thru devices and found any kind of GPU */
    cl_device_id device = nullptr;
    for (int i = 0; i < platformCount && device == nullptr; ++i) {
        cl_device_id devices[64];
        uint deviceCount;
        err_code = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 64, devices, &deviceCount);
        if (err_code == CL_SUCCESS)
            device = devices[i];
    }
    return device;
}

cl_program
get_program_from_file(const std::string filename, cl_context context, cl_int& err_code)
{
    std::ifstream ifs(filename);
    std::string src(std::istreambuf_iterator<char>{ifs}, {});

    const char *source = src.c_str();
    size_t length = src.length();

    /* program initialization */
    return clCreateProgramWithSource(context, 1, &source, &length, &err_code);
}

int 
main() 
{
    cl_int res = 0;
    cl_device_id device = get_gpu_device_id(res);
    /* context initalization */
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &res);
    assert(res == CL_SUCCESS);

    /* command queue initialization */
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &res);
    assert(res == CL_SUCCESS);

    cl_program program = get_program_from_file("src/functions.cl", context, res);
    assert(res == CL_SUCCESS);

    /* check program build info */
    res = clBuildProgram(program, 1, &device, "", nullptr, nullptr);    
    if (res != CL_SUCCESS) {
        char log[256];
        size_t log_len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_len, log, &log_len);
        return 1;
    }

    /* kernel initalization */
    cl_kernel kernel = clCreateKernel(program, "sobel_filter", &res);
    assert(res == CL_SUCCESS);

    size_t width = 6;
    size_t height = 7;

    uint32_t in_vec_data[] {
        0, 11, 11, 11, 0, 0,
        0, 11, 11, 11, 0, 0,
        0, 11, 11, 11, 0, 0,
        0, 11, 11, 11, 0, 0,
        0, 11, 11, 11, 0, 0,
        0, 11, 11, 11, 0, 0,
        0, 11, 11, 11, 0, 0
    };

    /* read output buffer */
    uint32_t out_vec_data[width * height] {0};

    /* input buffer prepare */
    cl_mem in_vec = clCreateBuffer(context, CL_MEM_READ_ONLY,  width * height * sizeof(*in_vec_data), nullptr, &res);
    assert(res == CL_SUCCESS);

    /* input buffer fill */
    res = clEnqueueWriteBuffer(queue, in_vec, CL_TRUE, 0, width * height * sizeof(*in_vec_data), in_vec_data, 0, nullptr, nullptr);
    assert(res == CL_SUCCESS);

    /* output buffer C prepare */
    cl_mem out_vec = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(*out_vec_data), nullptr, &res);
    assert(res == CL_SUCCESS);

    /* add all buffers to kernel */
    res |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_vec);
    res |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_vec);
    res |= clSetKernelArg(kernel, 2, sizeof(int), &(width));
    res |= clSetKernelArg(kernel, 3, sizeof(int), &(height));
    assert(res == CL_SUCCESS);

    /* dimension size specification */
    size_t globalWorkSize[2] {width, height};
    res |= clEnqueueNDRangeKernel(queue, kernel, 2, 0, globalWorkSize, nullptr, 0, nullptr, nullptr);
    res |= clEnqueueReadBuffer(queue, out_vec, CL_TRUE, 0, width * height * sizeof(*out_vec_data), out_vec_data, 0, nullptr, nullptr);
    assert(res == CL_SUCCESS);

    /* wait to all tasks ends */
    clFinish(queue);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::cout << out_vec_data[x + y * width] << ' ';
        }
        std::cout << '\n';
    }
    return EXIT_SUCCESS;
}