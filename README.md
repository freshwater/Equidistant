
# Equidistant

A program that finds equidistant points. Given the number of points and the number of dimensions, the program tries to find a configuration of points that equalizes all distances between them. This constraint is specified as a loss function and optimized using PyTorch. I like the engineering simplicity.

    model = gc.GeometricConstraints(points=3, dimensions=2, constraints=[
        lambda points, distances: distances.var()
    ])

    plt.scatter(*model.points().T);

![](images/3p2d_001.svg)