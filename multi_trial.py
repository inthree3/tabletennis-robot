from multiprocessing import Process,Queue
import numpy as np

def func1(q):
    a = np.ones((3,2))
    q.put(a)


def func2():
    print("b")


def func3(q):

    x = q.get()
    c = x

    print(c)





if __name__ == '__main__':
    q = Queue()

    # 프로세스를 생성합니다
    p1 = Process(target=func1, args=(q,))
    p2 = Process(target=func2)
    p3 = Process(target=func3, args=(q,))

    # start로 각 프로세스를 시작합니다. func1이 끝나지 않아도 func2가 실행됩니다.
    p1.start()
    p2.start()
    p3.start()

    q.close()
    q.join_thread()

    # # join으로 각 프로세스가 종료되길 기다립니다 p1.join()이 끝난 후 p2.join()을 수행합니다
    # p1.join()
    # p2.join()
    # p3.join()
