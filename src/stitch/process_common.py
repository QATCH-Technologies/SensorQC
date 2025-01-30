NUM_TILES = 306
NUM_ROWS = 18
SCALE_BY = 1.0
CORRECT_DF = True


def CheckStd(raw, avg, std):

    raw_x = raw[0]
    raw_y = raw[1]
    avg_x = avg[0]
    avg_y = avg[1]
    std_x = std[0]
    std_y = std[1]

    allowable_deviation_factor = 4

    if (abs(avg_x - raw_x) <=
            allowable_deviation_factor * std_x):
        check_x = True
    else:
        check_x = False

    if (abs(avg_y - raw_y) <=
            allowable_deviation_factor * std_y):
        check_y = True
    else:
        check_y = False

    return (check_x, check_y)
