import numpy as np

def gauss_jordan(A_in):
    A = np.array(A_in, dtype=float)
    A = np.concatenate((A, np.eye(A_in.shape[0])), axis=1)
    A = np.array(A, dtype=float)
    for k in range(A.shape[0]):
        max_value=-1
        max_index=-1
       # index =  np.argmax(np.abs(A), axis=0)[k]
        for index in range(k, A.shape[0]):
            if abs(A[index, k]) > max_value:
                max_value = abs(A[index, k])
                max_index = index
            
        if A[max_index, k] == 0:
            return None
        
        #swap rows max_index and k
        for i in range(A.shape[1]):
            temp = A[max_index, i]
            A[max_index, i] = A[k, i]
            A[k, i] = temp
        
        for j in range(k+1, A.shape[0]):
            f = A[j, k]/A[k, k]
            for p in range(A.shape[1]):
                A[j, p] = A[j, p]- (f*A[k, p])

    #back substitution
    for k in reversed(range(A.shape[0])):
        temp = A[k, k]
        for l in range(A.shape[1]):     
            A[k, l] = A[k, l]/temp #normalizing
        
        for j in reversed(range(k)):
            f = A[j, k]/A[k, k]
            for p in range(A.shape[1]):
                A[j, p] = A[j, p]- (f*A[k, p])
    
    return np.delete(A, [i for i in range(A_in.shape[0])], axis=1) #remove the identity at the left
    
