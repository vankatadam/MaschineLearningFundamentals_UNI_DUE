import numpy as np


# numeric
def add(a: float, b: float):
    """Add a to b and return result."""
    return a + b


def power(a: float, e: float):
    """Returns a to the power of e."""
    return a**e


def modulo(a: int, b: int):
    """Returns a modulo b."""
    return a % b


# strings
def string_len(string: str):
    """Returns length of string."""
    return len(string)


def char_at(string: str, i: int):
    """Returns character at position i of string."""
    return string[i]


def get_sub_string(string: str, start: int, end: int):
    """Returns substring from index start to end of string."""
    return string[start:end]


def contains_sub_string(string: str, sub_string: str):
    """Returns true if sub_string is part of string."""
    return sub_string in string


def get_string_date(day: int, month: int, year: int):
    """Returns the date as a string in the format DD-MM-YYYY.
    Example: 13-01-2020"""
    return f"{day:02d}-{month:02d}-{year:02d}"


# list
def list_len(a: list):
    """Returns the length of a list."""
    return len(a)


def last_element(a: list):
    """Returns the last element of a list.
    The list won't be empty."""
    return a[-1]


def contains_element(a: list, e):
    """Returns True if e is an element of list a."""
    return e in a


def sum_(a: list):
    """Returns the sum of all elements of list a.
    List a only contains integers and floats."""
    return sum(a)


def mean(a: list):
    """Returns the mean over all elements of list a.
    List a only contains integers and floats."""
    return sum(a) / len(a)


def pairwise_add(a: list, b: list):
    """Returns a list where the i-th element is the sum of the i-th element of a and b.
    List a and b only contain integers and floats.
    List a and b have the same length.
    Hint: take a look at the in-built python function zip()"""
    return list(map(add, a, b))
    """ oder dies: return(list(x+y for x,y in zip(a, b)))"""


# numpy
def np_pairwise_add(a: np.ndarray, b: np.ndarray):
    """Returns a numpy array where the i-th element is the sum of the i-th element of a and b.
    Numpy array a and b only contain integers and floats.
    Numpy array a and b have the same shape."""
    # zwei 1 dimensionale arrays der gleichen länge
    return np.add(a, b)
    # or a+b da => siehe unten

    """wenn "numpy arrays" übergeben werden, dann überschreibt numpy die "+" (__add__)methode
    falls dimensionen ungleich, dann wird der kleinere Array/Vektor erweitert (es wird
    so erweitert, dass der kleinere Array von anfang weiederholt wird. Zb.: (1,5,7) + 5
    führt zu (6,10,13) da 5 auf (5,5,5) erweitert wird) ... nach Testen funktioniert es nur mit einer Zahl
    oder, z.B.: print(a + [[1], [2], [10]])"""


def np_get_column_at(a: np.ndarray, i: int):
    """Returns the i-th column of the matrix a.
    a has at least 2 dimensions; shape (M, N, ...)"""
    # print(a) # : heißt nur, dass man alle elemente der Dimension haben will (hier die Zeile)
    return a[:, i]


def np_mean(a: np.ndarray):
    """Returns the mean over all elements of numpy array a regardless of shape."""
    return np.mean(a)


def np_mean_per_row(a: np.ndarray):
    """Returns the mean for each row of matrix a.
    a has at least 2 dimensions; shape (M, N, ...)."""
    # zeilenweise
    return np.mean(a, axis=1)


def np_mean_per_column(a: np.ndarray):
    """Returns the mean for each column of matrix a.
    a has at least 2 dimensions; shape (M, N, ...)."""
    # spaltenweise
    return np.mean(a, axis=0)


def np_to_row_vector(a: np.ndarray):
    """Returns a numpy array of shape (1, N) where N is the number of elements in a.
    a will always be of shape (N).  (zeilen = 1, spalten = N)"""
    #print("A:", a)
    # ich will 1 Zeile und den rest fillen in die spalte
    return a.reshape(1, -1)


def np_row_vector_to_column_vector(a: np.ndarray):
    """Returns a numpy array of shape (N, 1).
     a will always be a row vector with shape (1, N)."""
    return a.T


def np_column_vector_to_row_vector(a: np.ndarray):
    """Returns a numpy array of shape (1, N).
    a will always be a row vector with shape (N, 1)."""
    return a.T


def np_auto_column_row_vector_conversion(a: np.ndarray):
    """Returns a numpy array of shape (1, N) if a is of shape (N, 1)
    or a numpy array of shape (N, 1) if a is of shape (1, N).
    a will always be of shape (N, 1) or (1, N).
    Hint: take a look at matrix operation transpose https://en.wikipedia.org/wiki/Transpose"""
    return a.T


def np_dot_product(a: np.ndarray, b: np.ndarray):
    """Returns the scalar product (also called: dot product) of a and b.
    a and b are numpy arrays of shape (N)."""
    return np.dot(a, b)


def np_matrix_product(a: np.ndarray, b: np.ndarray):
    """Returns the matrix product of matrices a and b.
    a is a numpy array of shape (N, M).
    b is a numpy array of shape (M, N)."""
    return np.matmul(a,b)


if __name__ == "__main__":
    a = np.arange(1, 10)
    b = np.arange(6, 15)
    c = np.arange(1, 10).reshape(3, 3)

    print("--- numeric ---\n")
    print("add: ", add(5.7, 1.3))
    print("power: ", power(2, 3))
    print("modulo: ", modulo(18, 4))
    print("\n")

    print("--- strings ---\n")
    print("string_len: ", string_len("hello"))
    print("char_at: ", char_at("hello", 1))
    print("get_sub_string: ", get_sub_string("hello", 2, 4))
    print("contains_sub_string: ", contains_sub_string("hello", "ll"))
    print("get_string_date: ", get_string_date(28, 4, 2021))
    print("\n")

    print("--- list ---\n")
    print("list_len: ", list_len([1, 2, 3]))
    print("last_element: ", last_element([1, 2, 1]))
    print("contains_element: ", contains_element([1, 2, 7], 7))
    print("sum_: ", sum_([1, 2, 3]))
    print("mean: ", mean([1, 2, 3]))
    print("pairwise_add: ", pairwise_add([1, 2, 3], [4, 5, 6]))
    print("\n")

    print("--- numpy ---\n")
    print("pairwise_add: ", np_pairwise_add(a, b))
    print("np_get_column_at: ", np_get_column_at(c, 1))
    print("np_mean: ", np_mean(c))
    print("np_mean_per_row: ", np_mean_per_row(c))
    print("np_mean_per_column: ", np_mean_per_column(c))
    aRV = np_to_row_vector(a)
    print("np_to_row_vector: ", aRV)
    aCV = np_row_vector_to_column_vector(aRV)
    print("np_row_vector_to_column_vector: ", aCV)
    print("np_column_vector_to_row_vector: ",
          np_column_vector_to_row_vector(aCV))
    print("np_auto_column_row_vector_conversion: ",
          np_auto_column_row_vector_conversion(aRV))
    print("np_auto_column_row_vector_conversion: ",
          np_auto_column_row_vector_conversion(aCV))
    print("np_dot_product: ", np_dot_product(a, b))
    print("np_matrix_product: ", np_matrix_product(aRV, aCV))
