__kernel void binary_threshold(__constant uint* in, __global uint* out, const int width, const int height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const uint idx = x + y * width;
    out[idx] = in[idx] > 10;
}