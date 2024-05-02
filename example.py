from flame import flame_sim

# TODO : density shapes generator
# TODO : ignition gen
# TODO : GPU parralelization GEN
# Check : after successful implementation go to NO network
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Pool, set_start_method, freeze_support
import time


def simulate_flame(f, *args):
    f.simulate(*args)


def main():
    tstart = time.time()
    try:
        set_start_method('spawn')  # Warning! : must be called ('spawn')
    except RuntimeError:
        pass

    f1 = flame_sim(no_frames=80)
    # f2 = flame_sim(no_frames=80)

    t1 = Process(target=simulate_flame, args=(f1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1))
    # t2 = Process(target=simulate_flame, args=(f2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1))

    t1.start()
    # t2.start()

    t1.join()
    # t2.join()

    tstop = time.time()
    total = tstop - tstart
    print(total)


if __name__ == '__main__':
    freeze_support()
    main()
