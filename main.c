#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>

void calculate_on_gpu(const char *source, float f1, float f2);

int main() {
    float number = 0.0f;

    union {
        float f;
        unsigned int i;
    } a, b;

    a.i = 0x3F800001;
    b.i = 0xBF800002;

    __asm__  (
    "VMUL.F32 S3, %[input_j], %[input_i];"
    "VADD.F32 %[result], S3, %[input_k];"
    : [result] "=t"(number)
    : [input_i] "t"(a.f), [input_j] "t"(a.f), [input_k] "t"(b.f));
    printf("result: %.9g\n", number);

    __asm__  (
    "VFMA.F32 %[input_k], %[input_j], %[input_i];"
    "VMOV %[result], %[input_k];"
    : [result] "=t"(number)
    : [input_i] "t"(a.f), [input_j] "t"(a.f), [input_k] "t"(b.f));
    printf("result: %.9g\n", number);

    __asm__  (
    "VMLA.F32 %[input_k], %[input_j], %[input_i];"
    "VMOV %[result], %[input_k];"
    : [result] "=t"(number)
    : [input_i] "t"(a.f), [input_j] "t"(a.f), [input_k] "t"(b.f));
    printf("result: %.9g\n", number);

    const char *source = "__kernel void vector_add(__global float *A, __global float *B, __global float *C) {\n"
                                "A[0] = B[0] * B[0] + C[0];\n"
                         "}";
    /**
     *  %0 = load float, float addrspace(1)* %B, align 4, !tbaa !8
        %mul = fmul float %0, %0
        %1 = load float, float addrspace(1)* %C, align 4, !tbaa !8
        %add = fadd float %mul, %1
        store float %add, float addrspace(1)* %A, align 4, !tbaa !8
        ret void
     */
    calculate_on_gpu(source, a.f, b.f);
    return 0;
}

void calculate_on_gpu(const char *source, float f1, float f2) {
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1,
                         &device_id, &ret_num_devices);
    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

// Create memory buffers on the device for each vector
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      1 * sizeof(float), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      1 * sizeof(float), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                      1 * sizeof(float), NULL, &ret);

    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
                               1 * sizeof(float), &f1, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
                               1 * sizeof(float), &f2, 0, NULL, NULL);

    size_t source_len = strlen(source);

    cl_program program = clCreateProgramWithSource(context, 1,
                                                   (const char **) &source, (const size_t *) &source_len, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &c_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &a_mem_obj);

    size_t global_item_size = 1;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                                 &global_item_size, NULL, 0, NULL, NULL);

    float *output = malloc(1 * sizeof(float));
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
                              1 * sizeof(float), output, 0, NULL, NULL);
    float result = output[0];
    free(output);
    printf("result: %.9g\n", result);
}