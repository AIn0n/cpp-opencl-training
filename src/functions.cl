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

__kernel void median_filter_3(__constant uint* in, __global uint* out, const int width, const int height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x + 1 < width && y + 1 < height && x > 0 && y > 0) {
        out[x + y * width] = 
            ((get_idx(in, x - 1, y - 1, width) > 0) +
            (get_idx(in, x, y - 1, width) > 0) +
            (get_idx(in, x + 1, y - 1, width) > 0) +
            (get_idx(in, x - 1, y, width) > 0) +
            (get_idx(in, x, y, width) > 0) +
            (get_idx(in, x + 1, y, width) > 0) +
            (get_idx(in, x - 1, y + 1, width) > 0) +
            (get_idx(in, x, y + 1, width) > 0) +
            (get_idx(in, x + 1, y + 1, width) > 0)) > 4 ? 1 : 0;
    } else {
        out[x + y * width] = 0;
    }
}