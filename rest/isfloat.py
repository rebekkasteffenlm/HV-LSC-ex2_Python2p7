#!/usr/bin/python

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

