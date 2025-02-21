import datetime
from functools import partial
import logging
import platform
from typing import Final
import timeit
import cpuinfo
import psutil


def naive_implementation(input_data: list[tuple[float, float]]):
    import math

    def cross_product(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    def euclidean_distance(p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    p1, p2, p3, p4 = [point for point in input_data]

    d1 = cross_product(p3, p4, p1)
    d2 = cross_product(p3, p4, p2)
    d3 = cross_product(p1, p2, p3)
    d4 = cross_product(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        denominator = float(d2 - d1)
        if abs(denominator) < 1e-8:
            return None

        t = d3 / denominator
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        intersection = (x, y)

        distance_to_endpoints = min(
            [euclidean_distance(x, intersection) for x in (p1, p2, p3, p4)]
        )

        return 1.0 / (distance_to_endpoints + 1e-6)

    return 0


def naive_implementation_inline(input_data: list[tuple[float, float]]):
    import math

    p1, p2, p3, p4 = [point for point in input_data]

    d1 = (p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])
    d2 = (p4[0] - p3[0]) * (p2[1] - p3[1]) - (p4[1] - p3[1]) * (p2[0] - p3[0])
    d3 = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    d4 = (p2[0] - p1[0]) * (p4[1] - p1[1]) - (p2[1] - p1[1]) * (p4[0] - p1[0])

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        denominator = float(d2 - d1)
        if abs(denominator) < 1e-8:
            return None

        t = d3 / denominator
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        intersection = (x, y)

        distance_to_endpoints = min(
            [
                math.sqrt((intersection[0] - x[0]) ** 2 + (intersection[1] - x[1]) ** 2)
                for x in (p1, p2, p3, p4)
            ]
        )

        return 1.0 / (distance_to_endpoints + 1e-6)

    return 0


def numpy_ccw(input_data: list[tuple[float, float]]):
    import numpy as np

    def ccw(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> bool:
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    p1, p2, p3, p4 = [np.array(point) for point in input_data]

    lines_segments_intersect = ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(
        p1, p2, p3
    ) != ccw(p1, p2, p4)

    if not lines_segments_intersect:
        return 0

    denom = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])
    if np.abs(denom) < 1e-8:
        return 0

    ua = ((p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])) / denom

    intersection = p1 + ua * (p2 - p1)
    distance_to_endpoints = np.min(
        [np.linalg.norm(intersection - x) for x in (p1, p2, p3, p4)]
    )

    return 1.0 / (distance_to_endpoints + 1e-6)


def numpy_ccw_inline(input_data: list[tuple[float, float]]):
    import numpy as np

    p1, p2, p3, p4 = [np.array(point) for point in input_data]

    lines_segments_intersect = (
        (p4[1] - p1[1]) * (p3[0] - p1[0]) > (p3[1] - p1[1]) * (p4[0] - p1[0])
    ) != ((p4[1] - p2[1]) * (p3[0] - p2[0]) > (p3[1] - p2[1]) * (p4[0] - p2[0])) and (
        (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])
    ) != (
        (p4[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p4[0] - p1[0])
    )

    if not lines_segments_intersect:
        return 0

    denom = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])
    if np.abs(denom) < 1e-8:
        return 0

    ua = ((p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])) / denom

    intersection = p1 + ua * (p2 - p1)
    distance_to_endpoints = np.min(
        [np.linalg.norm(intersection - x) for x in (p1, p2, p3, p4)]
    )

    return 1.0 / (distance_to_endpoints + 1e-6)


def scipy_pdist(input_data: list[tuple[float, float]]):
    import numpy as np
    import scipy

    p1, p2, p3, p4 = [np.array(point) for point in input_data]

    lines_segments_intersect = (
        (p4[1] - p1[1]) * (p3[0] - p1[0]) > (p3[1] - p1[1]) * (p4[0] - p1[0])
    ) != ((p4[1] - p2[1]) * (p3[0] - p2[0]) > (p3[1] - p2[1]) * (p4[0] - p2[0])) and (
        (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])
    ) != (
        (p4[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p4[0] - p1[0])
    )

    if not lines_segments_intersect:
        return 0

    denom = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])
    if np.abs(denom) < 1e-8:
        return 0

    ua = ((p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])) / denom

    intersection = p1 + ua * (p2 - p1)
    distance_to_endpoints = np.min(
        scipy.spatial.distance.cdist([intersection], [p1, p2, p3, p4])
    )

    return 1.0 / (distance_to_endpoints + 1e-6)


def pretty_timedelta(delta: datetime.timedelta) -> str:
    timedelta_seconds = delta.total_seconds()
    sign_string = "-" if timedelta_seconds < 0 else ""

    days, seconds = divmod(abs(int(timedelta_seconds)), 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    milliseconds = float((timedelta_seconds % 1) * 1000)
    microseconds = delta.microseconds

    return f"{sign_string} {["", f"{days}d "][days > 0]}{["", f"{hours}h "][hours > 0]}{["", f"{minutes}m "][minutes > 0]}{["", f"{seconds}s "][seconds > 0]}{["", f"{milliseconds:.2f}ms"][int(milliseconds) > 0]}{["", f"{microseconds}µs"][int(milliseconds) == 0]}"


def format_list_to_string(
    input: list[datetime.timedelta], indent: int, cast=str
) -> str:
    return f"[{(',\n' + ' '*int(indent+14)).join([','.join([cast(value) for value in input[i:i + 5]]) for i in range(0, len(input), 5)])}]"


def format_function_to_string(input: callable) -> str:
    return input.__name__


if __name__ == "__main__":
    NUMBER_OF_ITERATIONS_PER_BENCHMARK: Final[int] = 100_000
    NUMBER_OF_BENCHMARKS_PER_FUNCTION: Final[int] = 25
    LOGGING_INDENT: Final[int] = 33
    LOGGING_HORIZONTAL_SEPARATOR_SIZE: Final[int] = 128
    # TODO: add actual, realistic input data - points that don't cross, parallel and perpindicular lines, lines that do cross, very large and small coordinates, etc.
    BENCHMARK_INPUT_DATA: Final[list[tuple[float, float]]] = [
        (6503.0, 2054.0),
        (2855.0, 1933.0),
        (5499.0, 2075.0),
        (5307.0, 945.0),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    benchmarks_to_run = [
        naive_implementation,
        naive_implementation_inline,
        numpy_ccw,
        numpy_ccw_inline,
        scipy_pdist,
    ]

    logging.info(
        f"Operating System:\n{' '*LOGGING_INDENT}Name: {platform.system()} {platform.release()}\n{' '*LOGGING_INDENT}Version: {platform.version()}\n{' '*LOGGING_INDENT}Architecture: {platform.architecture()[0]}"
    )
    cpu = cpuinfo.get_cpu_info()
    logging.info(
        f"CPU Information:\n{' '*LOGGING_INDENT}Processor: {cpu['brand_raw']}\n{' '*LOGGING_INDENT}Cores: {psutil.cpu_count(logical=False)}\n{' '*LOGGING_INDENT}Threads: {psutil.cpu_count(logical=True)}\n{' '*LOGGING_INDENT}Current Frequency: {psutil.cpu_freq().current:.2f} MHz\n{' '*LOGGING_INDENT}Advertised Frequency: {cpu["hz_advertised_friendly"]}"
    )

    logging.info(
        f"Running benchmarks: {format_list_to_string(benchmarks_to_run, indent=LOGGING_INDENT, cast=format_function_to_string)}, each benchmark will be run {NUMBER_OF_BENCHMARKS_PER_FUNCTION} times, for {NUMBER_OF_ITERATIONS_PER_BENCHMARK} iterations each."
    )
    logging.info(f"{'-'*LOGGING_HORIZONTAL_SEPARATOR_SIZE}")

    for benchmark in benchmarks_to_run:
        timer = timeit.Timer(partial(benchmark, BENCHMARK_INPUT_DATA))
        all_results = [
            datetime.timedelta(seconds=(result / NUMBER_OF_ITERATIONS_PER_BENCHMARK))
            for result in timer.repeat(
                repeat=NUMBER_OF_BENCHMARKS_PER_FUNCTION,
                number=NUMBER_OF_ITERATIONS_PER_BENCHMARK,
            )
        ]
        best_result = min(all_results)
        logging.info(
            f"Benchmark: {format_function_to_string(benchmark)}\n{' '*LOGGING_INDENT}Best result: {pretty_timedelta(best_result)}\n{' '*LOGGING_INDENT}All results: {format_list_to_string(all_results, indent=LOGGING_INDENT, cast=pretty_timedelta)}\n{'-'*LOGGING_HORIZONTAL_SEPARATOR_SIZE}"
        )
