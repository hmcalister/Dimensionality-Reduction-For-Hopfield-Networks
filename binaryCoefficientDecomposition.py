from typing import Tuple
import numpy as np

def BinaryCoefficientDecomposition(data: np.ndarray, iterationDataProportion:float, numComponents: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose the items of the data matrix (a (n,d) dim matrix of items) into a set of components such that
    the coefficients of those components are binary in value (0,1).

    This is done by, iteratively, computing the direction r with maximum variance of the data,
    then (using only the items most similar with that direction r, the number of items determined by `iterationDataProportion`),
    finding the scalar |R| such that r|R| best describes those most similar items. this value r|R| is the component at this iteration,
    and the most similar vectors have the component subtracted from them. The coefficient matrix 
    (stating as the zero matrix of size (n, numComponents)) has the i-th column filled with 1's at the indices
    of the most similar items. This process is run again with the newly reduced data.

    In a geometric explanation, we take a large collection of vectors, calculate the direction of maximum variance,
    then choose a number of vectors that are most aligned with that direction. Taking the average of those vectors
    (r|R|) as the components and subtracting that from the chosen set results in vectors that are smaller (closer to 0).
    Hence the data becomes more explained over time (variance decreases) and coefficient of components are all binary. 

    Returns the coefficient matrix (n, numComponents) and components matrix (numComponents, d) factorization
    """

    numItems, originalDimension = data.shape
    data = np.copy(data)
    components = np.zeros(shape=(numComponents, originalDimension))
    coefficients = np.zeros(shape=(numItems, numComponents))

    for componentIndex in range(numComponents):
        pass
        # Determine the direction of maximum variance

        # Determine the subset of most similar items

        # Calculate component

        # Reduce those items and update data matrix

        # Add calculated component to components matrix and update coefficient matrix

    return coefficients, components
