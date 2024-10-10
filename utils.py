from numpy import exp, log

def clamp(x, minimum, maximum):

    return max(minimum, min(x, maximum))

def length(minimum, maximum):

    return maximum - minimum

def sigmoid(x):

    return 1/(1 + exp(-x))

def isigmoid(x):

    return -log(1/x - 1)

