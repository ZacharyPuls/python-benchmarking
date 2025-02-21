# python-benchmarking
Generic repository I use to benchmark various Python algorithms.

## Running

```bash
pip install py-cpuinfo numpy scipy
python benchmark.py
```

## Example Output

```bash
(.venv) (base) zpuls@DESKTOP-E86VPTT:~/map-generation$ python benchmark.py
2025-02-21T00:57:59-0600 [INFO]: Operating System:
                                 Name: Linux 5.15.167.4-microsoft-standard-WSL2
                                 Version: #1 SMP Tue Nov 5 00:21:55 UTC 2024
                                 Architecture: 64bit
2025-02-21T00:58:00-0600 [INFO]: CPU Information:
                                 Processor: Intel(R) Core(TM) i9-14900K
                                 Cores: 16
                                 Threads: 32
                                 Current Frequency: 3187.20 MHz
                                 Advertised Frequency: 3.1872 GHz
2025-02-21T00:58:00-0600 [INFO]: Running benchmarks: [naive_implementation,naive_implementation_inline,numpy_ccw,numpy_ccw_inline,scipy_pdist], each benchmark will be run 25 times, for 100000 iterations each.
2025-02-21T00:58:00-0600 [INFO]: --------------------------------------------------------------------------------------------------------------------------------
2025-02-21T00:58:03-0600 [INFO]: Benchmark: naive_implementation
                                 Best result:  1µs
                                 All results: [ 1µs, 1µs, 1µs, 1µs, 1µs,
                                                1µs, 1µs, 1µs, 1µs, 1µs,
                                                1µs, 1µs, 1µs, 1µs, 1µs,
                                                1µs, 1µs, 1µs, 1µs, 1µs,
                                                1µs, 1µs, 1µs, 1µs, 1µs]
--------------------------------------------------------------------------------------------------------------------------------
2025-02-21T00:58:05-0600 [INFO]: Benchmark: naive_implementation_inline
                                 Best result:  1µs
                                 All results: [ 1µs, 1µs, 1µs, 1µs, 1µs,
                                                1µs, 1µs, 1µs, 1µs, 1µs,
                                                1µs, 1µs, 1µs, 1µs, 1µs,
                                                1µs, 1µs, 1µs, 1µs, 1µs,
                                                1µs, 1µs, 1µs, 1µs, 1µs]
--------------------------------------------------------------------------------------------------------------------------------
2025-02-21T00:58:33-0600 [INFO]: Benchmark: numpy_ccw
                                 Best result:  10µs
                                 All results: [ 12µs, 11µs, 11µs, 11µs, 11µs,
                                                11µs, 11µs, 11µs, 11µs, 11µs,
                                                11µs, 10µs, 11µs, 11µs, 11µs,
                                                11µs, 11µs, 10µs, 11µs, 11µs,
                                                11µs, 11µs, 11µs, 11µs, 11µs]
--------------------------------------------------------------------------------------------------------------------------------
2025-02-21T00:58:59-0600 [INFO]: Benchmark: numpy_ccw_inline
                                 Best result:  10µs
                                 All results: [ 11µs, 11µs, 11µs, 11µs, 11µs,
                                                11µs, 11µs, 11µs, 11µs, 10µs,
                                                11µs, 11µs, 11µs, 11µs, 11µs,
                                                11µs, 10µs, 11µs, 10µs, 11µs,
                                                11µs, 12µs, 11µs, 10µs, 11µs]
--------------------------------------------------------------------------------------------------------------------------------
2025-02-21T00:59:21-0600 [INFO]: Benchmark: scipy_pdist
                                 Best result:  8µs
                                 All results: [ 10µs, 8µs, 9µs, 9µs, 8µs,
                                                9µs, 9µs, 8µs, 8µs, 8µs,
                                                8µs, 8µs, 8µs, 8µs, 8µs,
                                                8µs, 8µs, 8µs, 8µs, 8µs,
                                                8µs, 8µs, 8µs, 8µs, 8µs]
--------------------------------------------------------------------------------------------------------------------------------
```
