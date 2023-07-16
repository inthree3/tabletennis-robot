import multiprocessing

def square(x):
    return x * x


if __name__ == '__main__':
    # Pool 객체 초기화
    pool = multiprocessing.Pool()
    # pool = multiprocessing.Pool(processes=4)

    # Pool.map
    inputs = [0, 1, 2, 3, 4]
    outputs = pool.map(square, inputs)

    print(outputs)

    # Pool.map_async
    outputs_async = pool.map_async(square, inputs)
    outputs = outputs_async.get()

    print(outputs)

    # Pool.apply_async
    results_async = [pool.apply_async(square, [i]) for i in range(100)]
    results = [r.get() for r in results_async]

    print(results)