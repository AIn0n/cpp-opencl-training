__kernel void binary_threshold(__constant uint* in, __global uint* out, const int width, const int height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const uint idx = x + y * width;
    out[idx] = in[idx] > 10;
}

uint get_idx(__constant uint* a, int x, int y, int width)
{
    return a[x + y * width];
}

void
insertion_sort(uint *arr, int n)
{
    uint key;
    int j;
    for (int i = 1; i < n; ++i) {
        key = arr[i];
        j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

__kernel void median_filter_3(__constant uint* in, __global uint* out, const int width, const int height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x + 1 < width && y + 1 < height && x > 0 && y > 0) {
        uint buffer[9] = {
            get_idx(in, x - 1,  y - 1,  width),
            get_idx(in, x,      y - 1,  width),
            get_idx(in, x + 1,  y - 1,  width),
            get_idx(in, x - 1,  y,      width),
            get_idx(in, x,      y,      width),
            get_idx(in, x + 1,  y,      width),
            get_idx(in, x - 1,  y + 1,  width),
            get_idx(in, x,      y + 1,  width),
            get_idx(in, x + 1,  y + 1,  width)
        };
        insertion_sort(buffer, 9);
        out[x + y * width] = buffer[4];
    } else {
        out[x + y * width] = 0;
    }
}
