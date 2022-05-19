MPIC = {
    2: {
        2: 6.5,
        4: 4.,
        8: 2.2,
    },
    4: {
        2: 3.9,
        4: 3.5,
        8: 2.1,
    },
    8: {
        2: 2.5,
        4: 2.3,
        8: 2.1,
    },
}

def mpic_lut(a_bit, w_bit):
    return MPIC[a_bit][w_bit]