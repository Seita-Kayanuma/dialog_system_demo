from . import base as b
from . import run as r


def demoTD21T24F02PTW05T04F03SIAE12T13V02():
    td = b.construct_turn_detector(21, 24, 2, ([5, 4, 3, 0, 12, 13, 2], {}))
    td.load(download=True)
    r.demo24(td)

    
def demoTD21T24F03SIAE12T13V02():
    td = b.construct_turn_detector(21, 24, 3, ([12, 13, 2], {}))
    td.load(download=True)
    r.demo24(td)

    
def demoTD22T22F02PTW05T04F03SIAE12T13V02():
    td = b.construct_turn_detector(22, 22, 2, ([5, 4, 3, 0, 12, 13, 2], {}))
    td.load(download=True)
    r.demo22(td)

    
def demoTD21T21F03SIAE12T13V02V01():
    td = b.construct_turn_detector(21, 21, 3, ([12, 13, 2], {}))
    td.load(download=True)
    r.demo2X(td)

    
def demoTD21T21F02PTW05T04F03SIAE12T13V02():
    td = b.construct_turn_detector(21, 21, 2, ([5, 4, 3, 0, 12, 13, 2], {}))
    td.load(download=True)
    r.demo2X(td)


def demoTD22T21F03SIAE12T13V02():
    td = b.construct_turn_detector(22, 21, 3, ([12, 13, 2], {}))
    td.load(download=True)
    r.demo2X(td)


def demoTD22T21F02PTW05T04F03SIAE12T13V02():
    td = b.construct_turn_detector(22, 21, 2, ([5, 4, 3, 0, 12, 13, 2], {}))
    td.load(download=True)
    r.demo2X(td)
    


