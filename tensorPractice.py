def split_x_into_chunks(x):
    #split tensor x into 4 equal chunks along the first dimension
    return x.chunk(chunks=4, dim=0)

def split_y_into_chunks(y):
    #split tensor y into 4 equal chunks
    return y.chunk(chunks=4)

def split_x_custom(x):
    #split tensor x into chunks with the following number of rows: 5 and 3 along its 1st dimension
    return x.split([5, 3], dim=0)

def split_y_custom(y):
    #split tensor y into chunks with the lengths 4, 6, and 6
    return y.split([4, 6, 6])