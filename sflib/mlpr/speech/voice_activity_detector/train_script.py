from .train import train

# --- 2020.2.17 001 ---
def trainVAD11T12F03SIAE12T13V02():
    train('cuda:0',
          ([11, 12, 3, ([12, 13, 2], {})], {}), ([], {'backprop_len': 50 * 60}))

# --- 2020.2.17 001 ---
def trainVAD11T11F03SIAE12T13V02():
    train('cuda:2',
          ([11, 11, 3, ([12, 13, 2], {})], {}), ([], {'backprop_len': 50 * 60}))

def trainVAD11T19F03SIAE12T13V02():
    train('cuda:2',
          ([11, 19, 3, ([12, 13, 2], {})], {}), ([], {'backprop_len': 50 * 60}))

# --- 2020.2.17 001 ---
def trainVAD01T99F03SIAE12T13V02():
    train('cuda:2',
          ([1, 99, 3, ([12, 13, 2], {})], {}), ([], {'backprop_len': 50 * 60}))

    
# --- 2020.2.16 001 ---
def trainVAD01T01F04SIAE12T13V02():
    train('cuda:1',
          ([1, 1, 4, ([12, 13, 2], {})], {}), ([], {'backprop_len': 50 * 60}))
    

def trainVAD01T01F03SIAE12T13V02():
    train('cuda:0',
          ([1, 1, 3, ([12, 13, 2], {})], {}), ([], {'backprop_len': 50 * 60}))
