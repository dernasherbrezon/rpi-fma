# About

This small project tests [FMA](https://en.wikipedia.org/wiki/Multiply–accumulate_operation#Fused_multiply–add) support on RaspberryPI 3 CPU and GPU. It computes the following:

```C
1.0000001F * 1.0000001F - 1.0000002F
```

# Results


<table>
    <thead>
        <tr>
            <th>Test name</th>
            <th>Result</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>VMUL + VADD</td>
            <td>0</td>
        </tr>
        <tr>
            <td>VFMA</td>
            <td>1.42108547e-14</td>
        </tr>
        <tr>
            <td>VMLA</td>
            <td>0</td>
        </tr>
        <tr>
            <td>GPU: FMUL + FADD</td>
            <td>0</td>
        </tr>
    </tbody>
</table>

# Dependencies

 * [VC4CL](https://github.com/doe300/VC4CL)

# Build

```bash
mkdir build
cd build
cmake ..
make
```

# Run

```bash
./rpi_fma
```
