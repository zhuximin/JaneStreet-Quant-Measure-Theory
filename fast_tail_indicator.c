/*
 * Low-Latency Tail Risk Indicator
 * Robust outlier detection for high-frequency data
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Comparator for double array sorting
int cmp_double(const void *a, const void *b) {
    double diff = (*(double*)a - *(double*)b);
    return (diff > 0) - (diff < 0);
}

// Calculate median (modifies input array)
double get_median(double* data, int len) {
    if (len == 0) return 0.0;
    qsort(data, len, sizeof(double), cmp_double);
    return (len % 2 == 0) ? (data[len/2 - 1] + data[len/2]) / 2.0 : data[len/2];
}

// Compute tail risk indicators (1 = outlier, 0 = normal)
// Uses MAD for robust thresholding
void compute_tail_indicators(double* feat_arr, int len, double thresh, int* out_idx) {
    double* temp = (double*)malloc(len * sizeof(double));
    for (int i = 0; i < len; i++) temp[i] = feat_arr[i];
    
    double median = get_median(temp, len);
    
    for (int i = 0; i < len; i++) temp[i] = fabs(feat_arr[i] - median);
    double mad = get_median(temp, len);
    free(temp);
    
    if (mad == 0.0) {
        for (int i = 0; i < len; i++) out_idx[i] = 0;
        return;
    }
    
    double limit = thresh * mad;
    double* p_feat = feat_arr;
    int* p_out = out_idx;
    
    // Fast pointer iteration
    for (int i = 0; i < len; i++) {
        *p_out = (fabs(*p_feat - median) > limit) ? 1 : 0;
        p_feat++;
        p_out++;
    }
}

// Unit test for tail risk detection
int main() {
    // Test data with normal values and extreme outliers
    double test_data[] = {1.0, 2.0, 2.5, 100.0, 3.0, -50.0, 2.1};
    int len = sizeof(test_data) / sizeof(test_data[0]);
    int out[len];

    compute_tail_indicators(test_data, len, 3.5, out);

    printf("[poc_test] input (f):  ");
    for(int i=0; i<len; i++) printf("%6.1f ", test_data[i]);
    
    printf("\n[poc_test] out (chi):  ");
    for(int i=0; i<len; i++) printf("%6d ", out[i]);
    printf("\n");

    return 0;
}