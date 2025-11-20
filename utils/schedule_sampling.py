import math
import numpy as np

def reserve_schedule_sampling_exp(args, itr, max_iter):
    if itr < max_iter // 4:
        r_eta = 0.5
    elif itr < max_iter // 2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - max_iter // 4) / args.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < max_iter // 4:
        eta = 0.5
    elif itr < max_iter // 2:
        eta = 0.5 - (0.5 / (max_iter // 2 - max_iter // 4)) * (itr - max_iter // 4)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (args.batch_size, args.input_length))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length))
    true_token = (random_flip < eta)

    h, w = args.grid_size
    
    ones = np.ones((h, w, args.img_channel))
    zeros = np.zeros((h, w, args.img_channel))

    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length):
            if j < args.input_length:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - args.input_length]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length,
                                  h,
                                  w,
                                  args.img_channel))
    return real_input_flag