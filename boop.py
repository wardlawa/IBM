import random, doctest


'''
desc
'''
def thingy(N, rands=None):
    '''
>>> thingy(5, rands=[2, 3])
3.3333333333333335

>>> thingy(7, rands=[3, 0])
'NaN'

>>> thingy(7, rands=[3, 7])
3

'''
    rand_mult = rands.pop(0) if rands else random.randint(0,10)
    rand_div = rands.pop(0) if rands else random.randint(0,10)

    if rand_div == 0:
        return "NaN"

    return (float(N)*rand_mult)/rand_div

doctest.testmod()#verbose=True)


