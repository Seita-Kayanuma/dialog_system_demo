from .train import train


# --- 2020.2.10 ---
# VADをやり直したので学習し直してみる
def trainTD32T31F03SIAE12T13V02X01():
    train('cuda:0', ([32, 31, 3, ([12, 13, 2], {})], {}))

    
def trainTD32T32F03SIAE12T13V02X01():
    train('cuda:2', ([32, 32, 3, ([12, 13, 2], {})], {}))

    
# --- 2019.12.9-10 ---
def trainTD32T31F03SIAE12T13V02():
    train('cuda:0', ([32, 31, 3, ([12, 13, 2], {})], {}))


def trainTD32T32F03SIAE12T13V02():
    train('cuda:2', ([32, 32, 3, ([12, 13, 2], {})], {}))


# --- 2019.12.9 ---
def trainTD31T31F03SIAE12T13V02():
    train('cuda:0', ([31, 31, 3, ([12, 13, 2], {})], {}))

    
def trainTD32T31F02PTW05T04F03SIAE12T13V02():
    train('cuda:1',
          ([32, 31, 2, ([5, 4, 3, 0, 12, 13, 2], {})], {}))

    
def trainTD31T32F03SIAE12T13V02():
    train('cuda:2', ([31, 32, 3, ([12, 13, 2], {})], {}))

    
def trainTD32T32F02PTW05T04F03SIAE12T13V02():
    train('cuda:3',
          ([32, 32, 2, ([5, 4, 3, 0, 12, 13, 2], {})], {}))
    

# --- 2019.12.6-7 ---
def trainTD23T28F03SIAE12T13V02():
    train('cuda:0', ([23, 28, 3, ([12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
def trainTD23T28F02PTW05T04F03SIAE12T13V02():
    train('cuda:1',
          ([23, 28, 2, ([5, 4, 3, 0, 12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
def trainTD24T28F03SIAE12T13V02():
    train('cuda:2', ([24, 28, 3, ([12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
def trainTD24T28F02PTW05T04F03SIAE12T13V02():
    train('cuda:3',
          ([24, 28, 2, ([5, 4, 3, 0, 12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
# --- 2019.12.5-6 ---
def trainTD23T26F03SIAE12T13V02():
    train('cuda:0', ([23, 26, 3, ([12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
def trainTD23T27F03SIAE12T13V02():
    train('cuda:0', ([23, 27, 3, ([12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
def trainTD23T27F02PTW05T04F03SIAE12T13V02():
    train('cuda:1',
          ([23, 27, 2, ([5, 4, 3, 0, 12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
def trainTD24T27F03SIAE12T13V02():
    train('cuda:2', ([24, 27, 3, ([12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
def trainTD24T27F02PTW05T04F03SIAE12T13V02():
    train('cuda:3',
          ([24, 27, 2, ([5, 4, 3, 0, 12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
# --- 2019.12.5 ---
def trainTD23T25F03SIAE12T13V02():
    train('cuda:0', ([23, 25, 3, ([12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
def trainTD23T25F02PTW05T04F03SIAE12T13V02():
    train('cuda:1',
          ([23, 25, 2, ([5, 4, 3, 0, 12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
def trainTD24T25F03SIAE12T13V02():
    train('cuda:2', ([24, 25, 3, ([12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
def trainTD24T25F02PTW05T04F03SIAE12T13V02():
    train('cuda:3',
          ([24, 25, 2, ([5, 4, 3, 0, 12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
# --- 2019.12.3 ---
def trainTD21T23F03SIAE12T13V02():
    train('cuda:3', ([21, 23, 3, ([12, 13, 2], {})], {}), ([], {'backprop_len': 500}))


def trainTD21T24F03SIAE12T13V02():
    train('cuda:0', ([21, 24, 3, ([12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
def trainTD21T24F02PTW05T04F03SIAE12T13V02():
    train('cuda:1',
          ([21, 24, 2, ([5, 4, 3, 0, 12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
def trainTD22T24F03SIAE12T13V02():
    train('cuda:2', ([22, 24, 3, ([12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
def trainTD22T24F02PTW05T04F03SIAE12T13V02():
    train('cuda:3',
          ([22, 24, 2, ([5, 4, 3, 0, 12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
# --- 2019.12.3 ---
def trainTD21T22F03SIAE12T13V02():
    train('cuda:0', ([21, 22, 3, ([12, 13, 2], {})], {}))

    
def trainTD21T22F02PTW05T04F03SIAE12T13V02():
    train('cuda:1',
          ([21, 22, 2, ([5, 4, 3, 0, 12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
def trainTD22T22F03SIAE12T13V02():
    train('cuda:2', ([22, 22, 3, ([12, 13, 2], {})], {}))

    
def trainTD22T22F02PTW05T04F03SIAE12T13V02():
    train('cuda:3',
          ([22, 22, 2, ([5, 4, 3, 0, 12, 13, 2], {})], {}), ([], {'backprop_len': 500}))


# --- 2019.12.1 ---
def trainTD21T21F03SIAE12T13V02():
    train('cuda:0', ([21, 21, 3, ([12, 13, 2], {})], {}))

    
def trainTD21T21F02PTW05T04F03SIAE12T13V02():
    train('cuda:1',
          ([21, 21, 2, ([5, 4, 3, 0, 12, 13, 2], {})], {}), ([], {'backprop_len': 500}))

    
def trainTD22T21F03SIAE12T13V02():
    train('cuda:2', ([22, 21, 3, ([12, 13, 2], {})], {}))

    
def trainTD22T21F02PTW05T04F03SIAE12T13V02():
    train('cuda:3',
          ([22, 21, 2, ([5, 4, 3, 0, 12, 13, 2], {})], {}), ([], {'backprop_len': 500}))
