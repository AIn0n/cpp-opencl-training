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

__kernel void median_filter_5(__constant uint* in, __global uint* out, const int width, const int height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x + 2 >= width || y + 2 >= height || x - 2 < 0 || y -2 < 0) {
        out[x + y * width] = 0;
    }
    uint buffer[25];
    int i = 0;
    for (int ky = -2; ky < 3; ++ky) {
        for (int kx = -2; kx < 3; ++kx) {
            buffer[i++] = get_idx(in, x + kx, y + ky, width);
        }
    }
    insertion_sort(buffer, 25);
    out[x + y * width] = buffer[12];
}

__kernel void sobel_filter(__constant uint *in, __global uint *out, const int width, const int height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    /*  filter shape
     *  -1 0 1
     *  -2 0 1
     *  -1 0 1
     */
    if (x + 1 < width && y + 1 < height && x > 0 && y > 0) {
        out[x + y * width] = abs(
            (int)get_idx(in, x - 1, y - 1,   width)  *    -1 +
            (int)get_idx(in, x + 1, y + 1,   width)          +
            (int)get_idx(in, x - 1, y,       width)  *    -2 +
            (int)get_idx(in, x + 1, y,       width)  *    2  +
            (int)get_idx(in, x - 1, y + 1,   width)  *    -1 +
            (int)get_idx(in, x + 1, y + 1,   width));
    } else {
        out[x + y * width] = 0;
    }
}