import numpy as np
import unittest
import sys

def check_load(filename,fun_name):
    """
    A wrapper function to load modules and functions to test. Produces error 
    if file or function is not found.
    
    Inputs: -> Filename that contains the functions we want to call without the ".py" extensions
            -> Function name that is inside filename that we want to call
            
    Output: -> Desired function is returned
    """
    try:
        #Importing the file source code
        user_module = __import__(filename) 
    except ModuleNotFoundError: 
        print("%s FILE NOT FOUND"%filename)
        return None
    
    """
     Everything in python is an object and importing this way will make it so that 
     user_module has a __dict__ which is a dictionary that contains all of the 
     relevant information about filename in it including the desired function
    """
    if hasattr(user_module,fun_name) and callable(user_module.__dict__[fun_name]):
        print("%s was succesfully loaded"%fun_name)
        return user_module.__dict__[fun_name] #return the function pointer for usage
    else: #did not find the function
        print("%s is not defined"%fun_name)
        return None

class TestGaussJordan(unittest.TestCase):
    fn = "demo"
    
    def test(self):
        print("*********** Testing gauss_jordan ***********")
        gauss_jordan = check_load(self.fn,"gauss_jordan")
        self._test1(gauss_jordan,"test 1")
        self._test2(gauss_jordan,"test 2")
        self._test_non_invertable(gauss_jordan,"test non-invertible")
    
    def _test1(self,gauss_jordan,test_name):
        print("---- %s ----"%test_name)
        A = np.array([[1,3],[2,5]])
        A_inv = np.array([[-5,3],[2,-1]])
        got = gauss_jordan(A)
        print("input:")
        print(A)
        print("expected:")
        print(A_inv)
        print("got:")
        print(got)

        if np.all(np.array_equal(got,A_inv)):
            print("result: Correct\n")
        else:
            print("result: Incorrect!\n")
        
    def _test4(self):
        from demo import gauss_jordan
        A = np.array([[1,3],[2,5]])
        A_inv = np.array([[-5,3],[2,-1]])
        got = gauss_jordan(A)
        print("input:")
        print(A)
        print("expected:")
        print(A_inv)
        print("got:")
        print(got)
    
        if np.all(np.isclose(got,A_inv)):
            print("result: Correct\n")
        else:
            print("result: Incorrect!\n")
            
    def _test2(self,gauss_jordan,test_name):
        print("---- %s ----"%test_name)
        A = np.array([[2,3,0],[1,-2,-1],[2,0,-1]])
        A_inv = np.array([[2,3,-3],[-1,-2,2],[4,6,-7]])
        got = gauss_jordan(A)
        print("input:")
        print(A)
        print("expected:")
        print(A_inv)
        print("got:")
        print(got)

        if np.all(np.isclose(got,A_inv)):
            print("result: Correct\n")
        else:
            print("result: Incorrect!\n")
        
    def _test_non_invertable(self,gauss_jordan,test_name):
        """
            A non-invertible example.
        """
        print("---- %s ----"%test_name)
        A = np.array([[2,3,0],[1,-2,-1],[2,-4,-2]])
        got = gauss_jordan(A)
        print("input:")
        print(A)
        print("expected:")
        print(None)
        print("got:")
        print(got)

        if got is None:
            print("result: Correct\n")
        else:
            print("result: Incorrect!\n")

if __name__=="__main__":
    unittest.main()
