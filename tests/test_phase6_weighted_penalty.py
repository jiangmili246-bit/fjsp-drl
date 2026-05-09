
def weighted_abs_penalty(w, C, D):
    return w * abs(C - D)


def main():
    C = 20
    D = 15  # same lateness = 5
    regular = weighted_abs_penalty(1, C, D)
    urgent = weighted_abs_penalty(4, C, D)
    assert urgent > regular, 'urgent penalty must exceed regular penalty for same lateness'
    print('Phase6 weighted penalty test passed:', regular, urgent)


if __name__ == '__main__':
    main()
