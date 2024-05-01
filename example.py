from flame import flame_sim

f1 = flame_sim(no_frames=100)
f1.simulate(frame_skip=20, plot=1, save_v=1,save_rgb=1)
print(f1.rgb_result)