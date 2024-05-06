import time
from torch.multiprocessing import set_start_method
from flameEngine.flame import flame_sim


def simulate_flame(f, *args):
    f.simulate(*args)


def main():
    tstart = time.time()
    try:
        set_start_method('spawn')  # Warning! ('spawn') must be called
    except RuntimeError:
        pass
    # f1 = flame_sim(no_frames=1500)
    # f2 = flame_sim(no_frames=1500)
    #
    # t1 = Process(target=simulate_flame, args=(f1, 1, 0, 20, 0, 0, 0, 0, 0, 0, 0, 1, 1))
    # t2 = Process(target=simulate_flame, args=(f2, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 1, 1))
    #
    # t1.start()
    # t2.start()
    #
    # t1.join()
    # t2.join()
    f1 = flame_sim(no_frames=500,frame_skip=25)

    f1.simulate( plot=0, save_animation=0, save_v=1, save_u=0, save_vu_mag=0, save_fuel=0,save_oxidizer=0,save_product=0, save_pressure=0, save_temperature=0, save_rgb=0, save_alpha=0)
    tstop = time.time()
    total = tstop - tstart
    print(round(total, 2), '[s]')


if __name__ == '__main__':
    # freeze_support()
    main()
