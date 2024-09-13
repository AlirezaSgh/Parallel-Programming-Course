#ifdef __aarch64__
#include "sse2neon.h" // Include this header for ARM64 architecture
#else
#include <x86intrin.h> // Include this header for x86 architecture
#endif

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <pthread.h>
#include <omp.h>
#include <iostream>

using namespace std;

struct ret_data
{
    double time;
    float avg;
    float std;
    friend ostream &operator<<(ostream &os, const ret_data &data)
    {
        os << "\tTime: " << data.time << ", \n"
           << "\tAverage: " << data.avg << ", \n"
           << "\tStandard Deviation: " << data.std;
        return os;
    }
};

struct Thread_Arg
{
    float *data;
    int start;
    int fin;
    float avg = 0;
};

const int MAX_SIZE = 1 << 20;
const int NUM_THREADS = 4;

ret_data compute_serial(float *);

ret_data compute_simd(float *);

ret_data compute_omp(float *);

ret_data compute_pthread(float *);

float calculate_avg_ser(float *);

float calculate_std_ser(float, float *);

float calculate_avg_simd(float *);

float calculate_std_simd(float, float *);

float calculate_avg_omp(float *);

float calculate_std_omp(float, float *);

void *calc_thread_sum(void *);

void *calc_thread_std_sum(void *);

float calculate_avg_pthread(float *);

float calculate_std_pthread(float, float *);

int main()
{
    ret_data ser, simd, omp, pthread;
    float numbers[MAX_SIZE];
    srand(time(NULL));
    for (int i = 0; i < MAX_SIZE; i++)
    {
        do
        {
            numbers[i] = (float)rand() / (float)(RAND_MAX);
        } while (numbers[i] == 0.0f);
    }

    ser = compute_serial(numbers);
    simd = compute_simd(numbers);
    omp = compute_omp(numbers);
    pthread = compute_pthread(numbers);

    cout << "Serial:\n"
         << ser << endl;
    cout << "SIMD:\n"
         << simd << endl;
    cout << "OpenMP:\n"
         << omp << endl;
    cout << "Posix:\n"
         << pthread << endl;
}

ret_data compute_serial(float numbers[MAX_SIZE])
{
    double start, end, time;
    ret_data data;
    float avg, std;
    start = omp_get_wtime();
    avg = calculate_avg_ser(numbers);
    std = calculate_std_ser(avg, numbers);
    end = omp_get_wtime();
    data.avg = avg;
    data.std = std;
    data.time = end - start;
    return data;
}

ret_data compute_simd(float numbers[MAX_SIZE])
{
    double start, end, time;
    ret_data data;
    float avg, std;
    start = omp_get_wtime();
    avg = calculate_avg_simd(numbers);
    std = calculate_std_simd(avg, numbers);
    end = omp_get_wtime();
    data.avg = avg;
    data.std = std;
    data.time = end - start;
    return data;
}

ret_data compute_omp(float numbers[MAX_SIZE])
{
    double start, end, time;
    ret_data data;
    float avg, std;
    start = omp_get_wtime();
    avg = calculate_avg_omp(numbers);
    std = calculate_std_omp(avg, numbers);
    end = omp_get_wtime();
    data.avg = avg;
    data.std = std;
    data.time = end - start;
    return data;
}

ret_data compute_pthread(float numbers[MAX_SIZE])
{
    double start, end, time;
    ret_data data;
    float avg, std;
    start = omp_get_wtime();
    avg = calculate_avg_pthread(numbers);
    std = calculate_std_pthread(avg, numbers);
    end = omp_get_wtime();
    data.avg = avg;
    data.std = std;
    data.time = end - start;
    return data;
}

float calculate_avg_ser(float numbers[MAX_SIZE])
{
    float sum[4] = {0, 0, 0, 0};
    for (int i = 0; i < MAX_SIZE; i += 4)
    {
        sum[0] += numbers[i];
        sum[1] += numbers[i + 1];
        sum[2] += numbers[i + 2];
        sum[3] += numbers[i + 3];
    }
    for (int i = 0; i < 4; i++)
        sum[i] = sum[i] / MAX_SIZE;
    float total_sum = sum[0] + sum[1] + sum[2] + sum[3];
    return total_sum;
}

float calculate_std_ser(float avg, float numbers[MAX_SIZE])
{
    float sum[4] = {0, 0, 0, 0};
    float temp;
    for (int i = 0; i < MAX_SIZE; i += 4)
    {
        for (int j = 0; j < 4; j++)
        {
            temp = avg - numbers[i + j];
            temp = temp * temp;
            sum[j] += temp;
        }
    }
    for (int i = 0; i < 4; i++)
        sum[i] = sum[i] / MAX_SIZE;
    float total_sum = sum[0] + sum[1] + sum[2] + sum[3];
    float std_div = sqrt(total_sum);
    return std_div;
}

float calculate_avg_simd(float numbers[MAX_SIZE])
{
    __m128 sum, temp;
    sum = _mm_set_ps(0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < MAX_SIZE; i += 4)
    {
        temp = _mm_load_ps(&numbers[i]);
        sum = _mm_add_ps(sum, temp);
    }
    temp = _mm_set_ps1(float(MAX_SIZE));
    __m128 avg = _mm_div_ps(sum, temp);
    avg = _mm_hadd_ps(avg, avg);
    avg = _mm_hadd_ps(avg, avg);
    return _mm_cvtss_f32(avg);
}

float calculate_std_simd(float avg_f, float numbers[MAX_SIZE])
{
    __m128 temp, sum, avg;
    avg = _mm_set1_ps(avg_f);
    sum = _mm_setzero_ps();
    for (int i = 0; i < MAX_SIZE; i += 4)
    {
        temp = _mm_load_ps(&numbers[i]);
        temp = _mm_sub_ps(temp, avg);
        temp = _mm_mul_ps(temp, temp);
        sum = _mm_add_ps(sum, temp);
    }
    temp = _mm_set_ps1(float(MAX_SIZE));
    __m128 standard_div = _mm_div_ps(sum, temp);
    standard_div = _mm_hadd_ps(standard_div, standard_div);
    standard_div = _mm_hadd_ps(standard_div, standard_div);
    standard_div = _mm_sqrt_ps(standard_div);
    return _mm_cvtss_f32(standard_div);
}

float calculate_avg_omp(float numbers[MAX_SIZE])
{
    float avg = 0;
#pragma omp parallel shared(numbers) reduction(+ : avg) num_threads(NUM_THREADS)
    {
        float sum = 0;
#pragma omp for
        for (int i = 0; i < MAX_SIZE; i++)
            sum += numbers[i];
        avg = float(sum) / MAX_SIZE;
    }
    return avg;
}

float calculate_std_omp(float avg_f, float numbers[MAX_SIZE])
{
    float sum = 0;
#pragma omp parallel reduction(+ : sum) num_threads(NUM_THREADS)
    {
        float temp = 0;
#pragma omp for
        for (int i = 0; i < MAX_SIZE; i++)
        {
            temp = avg_f - numbers[i];
            temp *= temp;
            sum += temp;
        }
        sum /= MAX_SIZE;
    }
    float std_div = sqrt(sum);
    return std_div;
}

void *calc_thread_sum(void *arg)
{
    Thread_Arg *input;
    float *sum;
    float *avg;
    sum = (float *)malloc(sizeof(float));
    input = (Thread_Arg *)arg;
    *sum = 0;
    for (int i = input->start; i < input->fin; i++)
        *sum += input->data[i];
    // *avg = sum / MAX_SIZE;
    pthread_exit(sum);
}

void *calc_thread_std_sum(void *args)
{
    Thread_Arg *input = (Thread_Arg *)args;
    float *sum = (float *)malloc(sizeof(float));
    float temp;
    *sum = 0;
    for (int i = input->start; i < input->fin; i++)
    {
        temp = input->avg - input->data[i];
        temp *= temp;
        *sum += temp;
    }
    pthread_exit(sum);
}

float calculate_avg_pthread(float numbers[MAX_SIZE])
{
    pthread_t threads[NUM_THREADS];
    Thread_Arg thread_inputs[NUM_THREADS];
    int chunk_size = MAX_SIZE / NUM_THREADS;
    float sum = 0;
    void *return_ptr;
    float *return_sum;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        thread_inputs[i].start = i * chunk_size;
        thread_inputs[i].fin = (i + 1) * chunk_size;
        thread_inputs[i].data = numbers;
        pthread_create(&threads[i], NULL, calc_thread_sum, &thread_inputs[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], &return_ptr);
        return_sum = (float *)return_ptr;
        sum += *return_sum;
        free(return_sum);
    }
    return sum / MAX_SIZE;
}

float calculate_std_pthread(float avg_f, float numbers[MAX_SIZE])
{
    pthread_t threads[NUM_THREADS];
    Thread_Arg thread_inputs[NUM_THREADS];
    int chunk_size = MAX_SIZE / NUM_THREADS;
    float sum = 0;
    void *return_ptr;
    float *return_sum;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        thread_inputs[i].start = i * chunk_size;
        thread_inputs[i].fin = (i + 1) * chunk_size;
        thread_inputs[i].data = numbers;
        thread_inputs[i].avg = avg_f;
        pthread_create(&threads[i], NULL, calc_thread_std_sum, &thread_inputs[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], &return_ptr);
        return_sum = (float *)return_ptr;
        sum += *return_sum;
        free(return_sum);
    }
    sum = sum / MAX_SIZE;
    float std_div = sqrt(sum);
    return std_div;
}
