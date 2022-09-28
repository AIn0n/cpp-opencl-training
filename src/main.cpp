#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>

int main() 
{
    /* search thru platforms (IDKs) */
    cl_platform_id platforms[64];
    uint platformCount;
    cl_int res = clGetPlatformIDs(64, platforms, &platformCount);
    if (res != CL_SUCCESS) {
        std::cout << "error: " << res << " LINE: " << __LINE__ << '\n';
        return 1;
    }
    /* search thru devices and found any kind of GPU */
    cl_device_id device = nullptr;
    for (int i = 0; i < platformCount && device == nullptr; ++i) {
        cl_device_id devices[64];
        uint deviceCount;
        res = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 64, devices, &deviceCount);
        if (res == CL_SUCCESS)
            device = devices[i];
    }

    /* context initalization */
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &res);
    if (res != CL_SUCCESS) {
        std::cout << "error: " << res << " LINE: " << __LINE__ << '\n';
        return 1;
    }

    /* command queue initialization */
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &res);
    if (res != CL_SUCCESS) {
        std::cout << "error: " << res << " LINE: " << __LINE__ << '\n';
        return 1;
    }

    std::string src = "\
    __kernel void vector_sum(__constant float* a, __constant float* b, __global float* c) {   \
    int i = get_global_id(0);       \
    float sum = a[i] + b[i];        \
    c[i] = sum;                     \
    }";

    const char *source = src.c_str();
    size_t length = src.length();

    

    /* program initialization */
    cl_program program = clCreateProgramWithSource(context, 1, &source, &length, &res);
    if (res != CL_SUCCESS) {
        std::cout << "error: " << res << " LINE: " << __LINE__ << '\n';
        return 1;
    }

    /* check program build info */
    res = clBuildProgram(program, 1, &device, "", nullptr, nullptr);    
    if (res != CL_SUCCESS) {
        char log[256];
        size_t log_len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_len, log, &log_len);
        return 1;
    }

    /* kernel initalization */
    cl_kernel kernel = clCreateKernel(program, "vector_sum", &res);
    if (res != CL_SUCCESS) {
        std::cout << "error: " << res << " LINE: " << __LINE__ << '\n';
        return 1;
    }

    /* input buffer A prepare */
    cl_mem veca = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * sizeof(float), nullptr, &res);
        if (res != CL_SUCCESS) {
        std::cout << "error: " << res << " LINE: " << __LINE__ << '\n';
        return 1;
    }

    /* input buffer A fill */
    float vecaData[] {15.0f, 4.0f};
    clEnqueueWriteBuffer(queue, veca, CL_TRUE, 0, 2 * sizeof(float), vecaData, 0, nullptr, nullptr);
    if (res != CL_SUCCESS) {
        std::cout << "error: " << res << " LINE: " << __LINE__ << '\n';
        return 1;
    }

    /* input buffer B prepare */
    cl_mem vecb = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * sizeof(float), nullptr, &res);
        if (res != CL_SUCCESS) {
        std::cout << "error: " << res << " LINE: " << __LINE__ << '\n';
        return 1;
    }

    /* input buffer B fill */
    float vecbData[] {5.0f, 6.0f};
    clEnqueueWriteBuffer(queue, vecb, CL_TRUE, 0, 2 * sizeof(float), vecbData, 0, nullptr, nullptr);
    if (res != CL_SUCCESS) {
        std::cout << "error: " << res << " LINE: " << __LINE__ << '\n';
        return 1;
    }

    /* output buffer C prepare */
    cl_mem vecc = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * sizeof(float), nullptr, &res);
    if (res != CL_SUCCESS) {
        std::cout << "error: " << res << " LINE: " << __LINE__ << '\n';
        return 1;
    }

    /* add all buffers to kernel */
    res = clSetKernelArg(kernel, 0, sizeof(cl_mem), &veca);
    if (res != CL_SUCCESS) {
        std::cout << "error: " << res << " LINE: " << __LINE__ << '\n';
        return 1;
    }
    res = clSetKernelArg(kernel, 1, sizeof(cl_mem), &vecb);
    if (res != CL_SUCCESS) {
        std::cout << "error: " << res << " LINE: " << __LINE__ << '\n';
        return 1;
    }
    res = clSetKernelArg(kernel, 2, sizeof(cl_mem), &vecc);
    if (res != CL_SUCCESS) {
        std::cout << "error: " << res << " LINE: " << __LINE__ << '\n';
        return 1;
    }

    /* dimension size specification */
    size_t globalWorkSize = 2;
    size_t localWorkSize = 2;
    res = clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalWorkSize,  &localWorkSize, 0, nullptr, nullptr);
    if (res != CL_SUCCESS) {
        std::cout << "error: " << res << " LINE: " << __LINE__ << '\n';
        return 1;
    }

    /* read output buffer */
    float veccData[2] {0};
    res = clEnqueueReadBuffer(queue, vecc, CL_TRUE, 0, 2 * sizeof(float), veccData, 0, nullptr, nullptr);
    if (res != CL_SUCCESS) {
        std::cout << "error: " << res << "LINE: " << __LINE__ << '\n';
        return 1;
    }

    /* wait to all tasks ends */
    clFinish(queue);
    for (int i = 0; i < 2; ++i)
        std::cout << veccData[i] << '\n';
}