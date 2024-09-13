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
    int indx;
    float data;
    friend ostream &operator<<(ostream &os, const ret_data &data)
    {
        os << "\tTime: " << data.time << ", \n"
           << "\tData: " << data.data << ", \n"
           << "\tIndex: " << data.indx << endl;
        return os;
    }
};

struct Func_Return
{
    int index;
    float smallest;
    double time;
};

struct Thread_Args
{
    float *data;
    int start;
    int end;
};

struct Thread_Return
{
    float smallest;
    int index = 0;
};

const int MAX_SIZE = 1 << 20;
const int NUM_THREADS = 4;

ret_data find_smallest_serial(float *);

ret_data find_smallest_omp(float *);

void *find_smallest_parallel_thread(void *);

ret_data find_smallest_pthread(float *);

int main()
{
    float numbers[MAX_SIZE];
    srand(time(NULL));
    for (int i = 0; i < MAX_SIZE; i++)
    {
        do
        {
            numbers[i] = (float)rand() / (float)(RAND_MAX);
        } while (numbers[i] == 0.0f);
    }
    ret_data ser, omp, simd, pthread;
    ser = find_smallest_serial(numbers);
    omp = find_smallest_omp(numbers);
    pthread = find_smallest_pthread(numbers);

    cout << "Serial:\n"
         << ser
         << "SIMD:\n"
         << simd
         << "OpenMP:\n"
         << omp
         << "Pthread:\n"
         << pthread;
}

ret_data find_smallest_omp(float data[])
{
    float smallest_gl = data[0];
    int smallest_i = -1;
    double start, end;
    start = omp_get_wtime();
    int smallest_index = -1;
    float smallest = data[0];
#pragma omp parallel shared(data, smallest_gl, smallest_i) private(smallest_index, smallest) num_threads(NUM_THREADS)
    {
        smallest_index = -1;
        smallest = data[0];

#pragma omp for
        for (int i = 0; i < MAX_SIZE; i++)
        {
            if (data[i] < smallest)
            {
                smallest = data[i];
                smallest_index = i;
            }
        }
#pragma omp critical
        {
            if (smallest < smallest_gl)
            {
                smallest_gl = smallest;
                smallest_i = smallest_index;
            }
        }
    }
    end = omp_get_wtime();
    return {end - start, smallest_i, smallest_gl};
}

ret_data find_smallest_serial(float data[])
{
    int smallest_index = 0;
    float smallest = data[0];
    double start, end;
    start = omp_get_wtime();
    for (int i = 0; i < MAX_SIZE; i++)
    {
        if (data[i] < smallest)
        {
            smallest_index = i;
            smallest = data[i];
        }
    }
    end = omp_get_wtime();
    return {end - start, smallest_index, smallest};
}

void *find_smallest_parallel_thread(void *args)
{
    struct Thread_Args *tArgs = (struct Thread_Args *)args;
    Thread_Return *return_data = (Thread_Return *)malloc(sizeof(Thread_Return));
    return_data->smallest = tArgs->data[0];
    for (int i = tArgs->start; i < tArgs->end; i++)
    {
        if (tArgs->data[i] < return_data->smallest)
        {
            return_data->index = i;
            return_data->smallest = tArgs->data[i];
        }
    }

    pthread_exit(return_data);
}

ret_data find_smallest_pthread(float data[])
{
    float smallest_gl = data[0];
    int smallest_i = -1;
    double start_time, end_time;
    start_time = omp_get_wtime();

    pthread_t threads[NUM_THREADS];
    Thread_Args Thread_Args[NUM_THREADS];
    Thread_Return *return_val;
    void *thread_return_val;
    int chunk_size = MAX_SIZE / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; i++)
    {
        Thread_Args[i].data = data;
        Thread_Args[i].start = i * chunk_size;
        Thread_Args[i].end = (i == NUM_THREADS - 1) ? MAX_SIZE : (i + 1) * chunk_size;

        pthread_create(&threads[i], NULL, find_smallest_parallel_thread, (void *)&Thread_Args[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], &thread_return_val);
        return_val = (Thread_Return *)thread_return_val;
        if (return_val->smallest < smallest_gl)
        {
            smallest_gl = return_val->smallest;
            smallest_i = return_val->index;
        }
        free(return_val);
    }

    end_time = omp_get_wtime();
    return {end_time - start_time, smallest_i, smallest_gl};
}
