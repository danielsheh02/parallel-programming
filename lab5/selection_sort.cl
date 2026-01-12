__kernel void selection_sort_chunks(
    __global double* arr,
    const int n,
    const int chunk_size
) {
    int gid = get_global_id(0);

    int start = gid * chunk_size;
    int end = start + chunk_size;

    if (gid == get_global_size(0) - 1) {
        end = n;
    }

    if (start >= n) return;

    for (int i = start; i < end - 1; ++i) {
        int min_idx = i;
        for (int j = i + 1; j < end; ++j) {
            if (arr[j] < arr[min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            double tmp = arr[i];
            arr[i] = arr[min_idx];
            arr[min_idx] = tmp;
        }
    }
}
